"""
CD_10_Batch.py
===============================================================================
10-Shift Agent-Based Simulation Engine: Batched Centralized Dispatch (CD_BATCHED)
-------------------------------------------------------------------------------
Manuscript Reference:
    "Matching Technology, Fleet Size, and Passenger Wait in Ride-Hailing Markets"
    - (Main_Document): paper_minor_2.tex, Section 5.1 ("Simulation Design and
      Calibration"), Section 5.2 ("Empirical Validation"), Table 3
      (`tab:scaling_audit_combined`), and Figure 5 (`fig:cd_batched_structure`).
    - (Appendix): EC_appendix_minor_2.tex, Section EC.2.3 ("Robustness Check:
      Central Dispatch With Batching").

Purpose:
    Evaluates the robustness of the Central Dispatch scaling and distributional
    approximations when the platform accumulates incoming passenger requests over
    a fixed batching window (`BATCH_WINDOW = 2` periods = 0.6 minutes) before
    performing global matching across accumulated requests and available drivers.

    Two-Mode Execution Design (Time-Saving Feature):
        - By default (`FORCE_RERUN_SCALING = False`, `FORCE_RERUN_WAITS = False`),
          this script checks whether the pre-computed simulation outputs
          (`Data/sim_scaling_CD_batched_w2.csv` and
          `Data/sim_waits_CD_batched_w2.csv`) already exist and loads them
          immediately without re-running the multi-hour simulation loops.
        - Setting `FORCE_RERUN_SCALING = True` and/or `FORCE_RERUN_WAITS = True`
          regenerates the CSV files from scratch across 10 independent shifts
          (seeds 0 through 9).
===============================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# =============================================================================
# PATH RESOLUTION & BATCH SIMULATION CONFIGURATION
# =============================================================================
# Batching interval in discrete periods (1 period = 0.3 min -> 2 periods = 0.6 min)
BATCH_WINDOW = 2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = SCRIPT_DIR if os.path.exists(os.path.join(SCRIPT_DIR, "Data")) else os.path.dirname(SCRIPT_DIR)

SCALING_FILE = os.path.join(PACKAGE_ROOT, "Data", f"sim_scaling_CD_batched_w{BATCH_WINDOW}.csv")
WAITS_FILE   = os.path.join(PACKAGE_ROOT, "Data", f"sim_waits_CD_batched_w{BATCH_WINDOW}.csv")

# Number of independent simulation shifts (replications) per fleet size
NUM_SEEDS = 10

# Fleet size sweep range covering the En-Route-Dominated Regime
N_D_RANGE = range(5300, 7600, 100)
TARGET_N_D_DIST = 6000  # Representative operating point in the en-route-dominated regime

# Operating Regime Cutoffs (En-Route-Dominated Regime)
N_START, N_SAT, N_END = 5300, 7200, 7600

# Safety Flags: Set to True ONLY if you wish to regenerate the CSV files from
# scratch (Warning: running 240 full grid shifts takes several hours).
FORCE_RERUN_SCALING = False  # Set to True to regenerate scaling data
FORCE_RERUN_WAITS   = False  # Set to True to regenerate distribution data


# =============================================================================
# 1. BATCHED CENTRALIZED DISPATCH SIMULATION ENGINE
#    (See Section EC.2.3 of (Appendix))
# =============================================================================
def sim_batched_dispatch_strict(
    n_d,
    num_of_periods=1000,
    seed_init=12,
    params_sim=(80, 10, 2, 0),
    idle_cruise_prob=0.9,
    abandon_after_periods=40,
    burn_in=200,
    return_individuals=False,
    batch_window=1
):
    """
    Executes a single discrete-time shift of the toroidal Centralized Dispatch
    simulation under periodic batch matching (`batch_window` periods per batch).

    Arrivals, cruising, and abandonment occur in every period t = 0, ..., T - 1,
    while the global matching step (Step 4) executes every `batch_window`
    periods (`if t % batch_window == 0`).

    Burn-In:
        The simulation initializes with an empty backlog ("cold start"). The
        first `burn_in = 200` periods (200 * 0.3 = 60 minutes = 1 hour) are
        discarded so that only steady-state waits (t >= burn_in) are recorded.

    Args:
        n_d (int): Number of centrally dispatched drivers in the fleet.
        num_of_periods (int): Total number of discrete simulation periods.
        seed_init (int): Integer seed for the NumPy random number generator.
        params_sim (tuple): (num_blocks, speed_mph, trip_miles, _).
        idle_cruise_prob (float): Probability an idle driver advances 1 block per period.
        abandon_after_periods (int): Maximum passenger waiting tolerance K in periods.
        burn_in (int): Number of initial periods discarded as burn-in (200 periods = 1 hr).
        return_individuals (bool): If True, returns array of all served passenger waits.
        batch_window (int): Number of periods between batch matching dispatches.

    Returns:
        dict: Contains 'mean_total_wait_min' (float) and optionally
              'individual_waits' (np.ndarray).
    """
    rng = np.random.default_rng(seed_init)
    num_blocks, speed_mph, trip_miles, _ = params_sim
    time_per_block_min = 0.3     # 0.3 minutes (18 seconds) per city block
    lambda_per_period = 117.96   # 23,592 requests/hr * (0.3 / 60) hr/period
    mean_trip_blocks = 40.0      # Average trip distance in blocks (1/tau = 40)

    # Initialize driver positions uniformly across the G x G grid (Cold Start)
    cabs_x = rng.integers(0, num_blocks, size=n_d)
    cabs_y = rng.integers(0, num_blocks, size=n_d)
    cabs_busy_until = np.zeros(n_d, dtype=int)
    cabs_ids = np.arange(n_d)

    backlog_x = []
    backlog_y = []
    backlog_arr = []

    total_wait_sum = 0.0
    served_count = 0
    individual_waits = []

    def get_dist_matrix(px, py, dx, dy, limit):
        """Computes pairwise toroidal Manhattan distances between passengers and idle drivers."""
        diff_x = np.abs(px[:, None] - dx[None, :])
        diff_x = np.minimum(diff_x, limit - diff_x)
        diff_y = np.abs(py[:, None] - dy[None, :])
        diff_y = np.minimum(diff_y, limit - diff_y)
        return diff_x + diff_y

    # --- Main Discrete-Time Simulation Loop ---
    for t in range(num_of_periods):
        # Step 1: Idle Driver Cruising (Random walk on toroidal grid)
        idle_mask = cabs_busy_until <= t
        idle_indices = np.flatnonzero(idle_mask)
        if len(idle_indices) > 0 and idle_cruise_prob > 0:
            moves = rng.random(size=len(idle_indices)) < idle_cruise_prob
            if np.any(moves):
                movers = idle_indices[moves]
                axis = rng.integers(0, 2, size=len(movers))
                direction = rng.choice([-1, 1], size=len(movers))
                cabs_x[movers] = np.where(axis == 0, (cabs_x[movers] + direction) % num_blocks, cabs_x[movers])
                cabs_y[movers] = np.where(axis == 1, (cabs_y[movers] + direction) % num_blocks, cabs_y[movers])

        # Step 2: Passenger Arrivals (Accumulated in central platform backlog B)
        k = int(rng.poisson(lambda_per_period))
        if k > 0:
            backlog_x.extend(rng.integers(0, num_blocks, size=k))
            backlog_y.extend(rng.integers(0, num_blocks, size=k))
            backlog_arr.extend([t] * k)

        # Step 3: Customer Abandonment (Remove requests waiting longer than K periods)
        if abandon_after_periods and len(backlog_x) > 0:
            keep = [(t - arr) < abandon_after_periods for arr in backlog_arr]
            if not all(keep):
                backlog_x = [x for x, k in zip(backlog_x, keep) if k]
                backlog_y = [y for y, k in zip(backlog_y, keep) if k]
                backlog_arr = [a for a, k in zip(backlog_arr, keep) if k]

        # Step 4: Global Batched Matching (Executed every `batch_window` periods)
        if t % batch_window == 0:
            idle_mask = cabs_busy_until <= t
            idle_indices = np.flatnonzero(idle_mask)
            n_pax = len(backlog_x)
            n_idle = len(idle_indices)

            if n_pax > 0 and n_idle > 0:
                dists = get_dist_matrix(
                    np.array(backlog_x), np.array(backlog_y),
                    cabs_x[idle_indices], cabs_y[idle_indices], num_blocks
                )
                flat_dists = dists.ravel()

                # Computational optimization: use `argpartition` to pre-filter the
                # top candidate pairs (`100 * n_pax` shortest distances) before full
                # lexicographical sorting, avoiding O(N log N) sorting overhead on
                # millions of distant pairs when the batch pool is large.
                n_pairs = len(flat_dists)
                k_top = min(n_pairs, n_pax * 100)

                if k_top < n_pairs:
                    candidate_indices = np.argpartition(flat_dists, k_top)[:k_top]
                else:
                    candidate_indices = np.arange(n_pairs)

                # Map flat candidate indices back to (passenger_index, idle_driver_index)
                cand_p_indices = candidate_indices // n_idle
                cand_d_indices = candidate_indices % n_idle

                cand_dists = flat_dists[candidate_indices]
                cand_arr   = np.array([backlog_arr[p] for p in cand_p_indices])
                cand_ids   = cabs_ids[idle_indices[cand_d_indices]]

                # Lexicographical sort on candidates: primary = distance, secondary = arrival time, tertiary = driver ID
                local_order = np.lexsort((cand_ids, cand_arr, cand_dists))
                sorted_candidates = candidate_indices[local_order]

                matched_p = set()
                matched_d = set()
                rem_p_indices = []

                # Step 5: Service Execution & State Updates
                for k_flat in sorted_candidates:
                    if len(matched_p) == n_pax or len(matched_d) == n_idle:
                        break

                    pidx = k_flat // n_idle
                    didx = k_flat % n_idle

                    if pidx in matched_p or didx in matched_d:
                        continue

                    matched_p.add(pidx)
                    matched_d.add(didx)

                    # Total wait = batch/queue wait (t - t_arr) + en-route pickup travel time
                    wait_time = (t - backlog_arr[pidx]) * time_per_block_min + flat_dists[k_flat] * time_per_block_min
                    if t >= burn_in:
                        total_wait_sum += wait_time
                        served_count += 1
                        if return_individuals:
                            individual_waits.append(wait_time)

                    # Sample trip length L and relocate driver to passenger's destination
                    real_d_idx = idle_indices[didx]
                    L = int(rng.geometric(1.0 / mean_trip_blocks))
                    move_x = rng.integers(0, L + 1)
                    move_y = L - move_x
                    sx = rng.choice([-1, 1])
                    sy = rng.choice([-1, 1])

                    cabs_x[real_d_idx] = (backlog_x[pidx] + sx * move_x) % num_blocks
                    cabs_y[real_d_idx] = (backlog_y[pidx] + sy * move_y) % num_blocks
                    cabs_busy_until[real_d_idx] = t + int(flat_dists[k_flat]) + L
                    rem_p_indices.append(pidx)

                # Remove matched passengers from the central backlog
                if rem_p_indices:
                    rem_set = set(rem_p_indices)
                    backlog_x = [x for i, x in enumerate(backlog_x) if i not in rem_set]
                    backlog_y = [y for i, y in enumerate(backlog_y) if i not in rem_set]
                    backlog_arr = [a for i, a in enumerate(backlog_arr) if i not in rem_set]

    res = {"mean_total_wait_min": total_wait_sum / max(1, served_count)}
    if return_individuals:
        res["individual_waits"] = np.array(individual_waits)
    return res


# =============================================================================
# 2A. LOOP 1: GENERATE OR LOAD SCALING DATA (`Data/sim_scaling_CD_batched_w2.csv`)
# =============================================================================
if not FORCE_RERUN_SCALING and os.path.exists(SCALING_FILE):
    print("Found existing Scaling Data. Loading...")
    df_scaling = pd.read_csv(SCALING_FILE)
    print(f"Scaling data loaded from {SCALING_FILE}")
else:
    print(f"\n{'='*60}")
    print(f"LOOP 1: Generating Scaling Data (Batched w={BATCH_WINDOW})")
    print(f"{'='*60}")
    print(f"Running {NUM_SEEDS} seeds across n_d range {N_D_RANGE.start}-{N_D_RANGE.stop-1}...")

    scaling_data = []
    for seed in range(NUM_SEEDS):
        print(f"  ... Scaling Sweep Seed {seed+1}/{NUM_SEEDS}")
        for n_d in N_D_RANGE:
            res = sim_batched_dispatch_strict(n_d=n_d, seed_init=seed, batch_window=BATCH_WINDOW)
            scaling_data.append({
                "seed": seed, "n_d": n_d, "wait": res["mean_total_wait_min"]
            })

    df_scaling = pd.DataFrame(scaling_data)
    df_scaling.to_csv(SCALING_FILE, index=False)
    print(f"✓ Scaling data saved to {SCALING_FILE}")


# =============================================================================
# 2B. LOOP 2: GENERATE OR LOAD DISTRIBUTION DATA (`Data/sim_waits_CD_batched_w2.csv`)
# =============================================================================
if not FORCE_RERUN_WAITS and os.path.exists(WAITS_FILE):
    print("Found existing Distribution Data. Loading...")
    df_waits = pd.read_csv(WAITS_FILE)
    print(f"Distribution data loaded from {WAITS_FILE}")
else:
    print(f"\n{'='*60}")
    print(f"LOOP 2: Generating Distribution Data (Batched w={BATCH_WINDOW})")
    print(f"{'='*60}")
    print(f"Running {NUM_SEEDS} seeds at n_d={TARGET_N_D_DIST} (2000 periods each)...")

    all_waits_6000 = []
    for seed in range(NUM_SEEDS):
        print(f"  ... Distribution Deep Dive Seed {seed+1}/{NUM_SEEDS} (n_d={TARGET_N_D_DIST})")
        res_dist = sim_batched_dispatch_strict(
            n_d=TARGET_N_D_DIST, num_of_periods=2000,
            seed_init=seed, return_individuals=True, batch_window=BATCH_WINDOW
        )
        seed_waits = pd.DataFrame({"seed": seed, "wait_min": res_dist["individual_waits"]})
        all_waits_6000.append(seed_waits)

    df_waits = pd.concat(all_waits_6000, ignore_index=True)
    df_waits.to_csv(WAITS_FILE, index=False)
    print(f"✓ Distribution data saved to {WAITS_FILE}")

print(f"\n{'='*60}")
print("Data Ready for Analysis")
print(f"{'='*60}")
