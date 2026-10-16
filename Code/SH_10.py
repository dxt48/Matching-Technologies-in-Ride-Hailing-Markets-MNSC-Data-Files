"""
SH_10.py
===============================================================================
10-Shift Agent-Based Simulation Engine: Decentralized Street-Hailing (SH)
-------------------------------------------------------------------------------
Manuscript Reference:
    "Matching Technology, Fleet Size, and Passenger Wait in Ride-Hailing Markets"
    - (Main_Document): paper_minor_2.tex, Section 5.1 ("Simulation Design and
      Calibration") and Section 5.2 ("Empirical Validation").
    - (Appendix): EC_appendix_minor_2.tex, Section EC.2.1 ("Street Hailing
      Simulation Model") and Algorithm 1 ("Toroidal Street Hailing Simulation
      with Queue Dynamics").

Purpose:
    Simulates a decentralized street-hailing ride-hailing market on an 80 x 80
    wrap-around (toroidal) grid (J = 6,400 corners) across 10 independent
    simulation shifts (seeds 0 through 9).

    Two-Mode Execution Design (Time-Saving Feature):
        - By default (`FORCE_RERUN_SCALING = False`, `FORCE_RERUN_WAITS = False`),
          this script checks whether the pre-computed simulation outputs
          (`Data/sim_scaling_SH.csv` and `Data/sim_waits_SH.csv`) already exist
          and loads them immediately without re-running the multi-hour simulation
          loops.
        - Setting `FORCE_RERUN_SCALING = True` and/or `FORCE_RERUN_WAITS = True`
          regenerates the CSV files from scratch by executing:
            * Loop 1 (Scaling Sweep): 10 shifts across 24 fleet sizes
              n_s in {6400, 6500, ..., 8700} (240 full simulation shifts).
            * Loop 2 (Distributional Deep Dive): 10 extended shifts of 2,000
              periods each collecting individual passenger wait times.
===============================================================================
"""

import logging
import os
import warnings
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# =============================================================================
# PATH RESOLUTION & SIMULATION CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = SCRIPT_DIR if os.path.exists(os.path.join(SCRIPT_DIR, "Data")) else os.path.dirname(SCRIPT_DIR)

SCALING_FILE = os.path.join(PACKAGE_ROOT, "Data", "sim_scaling_SH.csv")
WAITS_FILE   = os.path.join(PACKAGE_ROOT, "Data", "sim_waits_SH.csv")

# Number of independent simulation shifts (replications) per fleet size
NUM_SEEDS = 10

# Fleet size sweep range covering the Matching-Dominated Regime (Approximation 1)
N_S_RANGE = range(6400, 8800, 100)
TARGET_N_S_DIST = 7200  # Representative operating point in the matching-dominated regime

# Operating Regime Cutoffs (Matching-Dominated Regime)
N_START, N_SAT, N_END = 6600, 8500, 8800

# Safety Flags: Set to True ONLY if you wish to regenerate the CSV files from
# scratch (Warning: running 250 full grid shifts takes several hours).
FORCE_RERUN_SCALING = False  # Set to True to regenerate Data/sim_scaling_SH.csv
FORCE_RERUN_WAITS   = False  # Set to True to regenerate Data/sim_waits_SH.csv


# =============================================================================
# 1. STREET-HAILING SIMULATION ENGINE (Algorithm 1 in Section EC.2.1 of (Appendix))
# =============================================================================
CITY_SIDE_MILES = 4.0          # 4 miles x 4 miles = 16 sq. mile Manhattan study area
BASE_ARRIVALS_PER_MIN = 393.2  # 23,592 passenger ride requests per hour / 60 min

def sim_street_hailing_with_queue(
    n_s,
    num_of_periods=1000,
    seed_init=12,
    params_sim=(80, 10, 2, 0),
    demand_rate_scale=1,
    mean_trip_blocks=40.0,
    idle_cruise_prob=0.5,
    abandon_after_periods=40,
    return_individual_waits=False,
    burn_in=200
):
    """
    Executes a single discrete-time shift of the toroidal Street-Hailing
    simulation with local FIFO queues at each city corner (Algorithm 1 in
    Section EC.2.1 of (Appendix)).

    In each discrete time period t = 0, ..., num_of_periods - 1 (where each
    period represents 0.3 minutes):
        Step 1 (Demand Realization):
            Sample total city arrivals k ~ Poisson(lam) and assign each arriving
            passenger uniformly at random to a corner FIFO queue Q_{x,y}.
        Step 2 (Customer Abandonment):
            Remove any waiting passenger whose waiting time (t - t_arr) exceeds
            the impatience threshold K = `abandon_after_periods`.
        Step 3 (Pre-Cruising Pickups):
            Idle drivers (busy_until <= t) check their current corner queue; if
            non-empty, they immediately pick up the head-of-line passenger,
            sample a trip length L ~ Geometric(1 / mean_trip_blocks), relocate
            to the drop-off corner, and remain busy until t + L.
        Step 4 (Idle Driver Cruising):
            Remaining idle drivers move one block in a random cardinal direction
            (North, South, East, West) with probability `idle_cruise_prob`,
            subject to wrap-around (toroidal) boundary conditions.
        Step 5 (Post-Cruising Pickups):
            Idle drivers who just cruised check the queue at their new corner
            and pick up the head-of-line passenger if present.

    Burn-In:
        The simulation initializes with an empty grid ("cold start"). The first
        `burn_in = 200` periods (200 * 0.3 = 60 minutes = 1 hour) are discarded
        so that only steady-state waits (t >= burn_in) are recorded.

    Args:
        n_s (int): Number of street-hailing drivers in the fleet.
        num_of_periods (int): Total number of discrete simulation periods.
        seed_init (int): Integer seed for the NumPy random number generator.
        params_sim (tuple): (num_blocks, speed_mph, trip_len_miles, _).
        demand_rate_scale (float): Multiplier on baseline passenger arrival rate.
        mean_trip_blocks (float): Average trip distance in city blocks (1/tau = 40).
        idle_cruise_prob (float): Probability an idle driver advances 1 block per period.
        abandon_after_periods (int): Maximum passenger waiting tolerance K in periods.
        return_individual_waits (bool): If True, returns array of all served passenger waits.
        burn_in (int): Number of initial periods discarded as burn-in (200 periods = 1 hr).

    Returns:
        dict: Contains 'mean_total_wait_min' (float) and optionally
              'individual_waits' (np.ndarray).
    """
    # --- RNG & Grid Calibration Setup ---
    rng = np.random.default_rng(int(seed_init))
    num_blocks, speed_mph, trip_len_miles, _ = params_sim

    city_side_miles   = CITY_SIDE_MILES
    block_len_miles   = city_side_miles / num_blocks                  # 0.05 miles per block
    time_per_blockmin = 60.0 * (block_len_miles / speed_mph)          # 0.3 minutes per period

    mu_blocks = float(mean_trip_blocks)
    mu_blocks = max(mu_blocks, 1e-6)

    # Expected city-wide passenger arrivals per 0.3-minute period (~117.96)
    lam = float(demand_rate_scale) * time_per_blockmin * BASE_ARRIVALS_PER_MIN

    # --- Coordinate & Sampling Helper Functions ---
    def wrap_step(pos, step, limit):
        """Toroidal (wrap-around) coordinate update modulo grid dimension G."""
        return (int(pos) + int(step)) % int(limit)

    def sample_trip_blocks():
        """Samples passenger trip distance L (in blocks) from a Geometric distribution."""
        p = min(1.0, max(1e-6, 1.0 / mu_blocks))
        return int(rng.geometric(p))

    def cell_index(x, y):
        """Maps 2D corner coordinates (x, y) to a 1D array index in [0, G^2 - 1]."""
        return int(x) * num_blocks + int(y)

    # --- State Initialization (Cold Start) ---
    xC = rng.integers(0, num_blocks, size=int(n_s), dtype=np.int64)
    yC = rng.integers(0, num_blocks, size=int(n_s), dtype=np.int64)
    busy_until = np.zeros(int(n_s), dtype=np.int64)
    corner_queues = [deque() for _ in range(num_blocks * num_blocks)]

    served = 0
    sum_queue_min = 0.0
    individual_waits = []

    # --- Main Discrete-Time Simulation Loop ---
    for t in range(int(num_of_periods)):
        # Step 1: Passenger Arrivals (Poisson process distributed uniformly across corners)
        k = int(rng.poisson(lam))
        if k > 0:
            xP = rng.integers(0, num_blocks, size=k)
            yP = rng.integers(0, num_blocks, size=k)
            for xi, yi in zip(xP, yP):
                corner_queues[cell_index(xi, yi)].append(int(t))

        # Step 2: Customer Abandonment (Impatience threshold K)
        if abandon_after_periods is not None:
            thresh = int(t) - int(abandon_after_periods)
            if thresh >= 0:
                for q in corner_queues:
                    while q and int(q[0]) <= thresh:
                        q.popleft()

        # Step 3: Pre-Cruising Pickups at Drivers' Current Corners
        idle_mask = (busy_until <= t)
        idle_idx  = np.flatnonzero(idle_mask)

        # Randomly shuffle idle driver order to prevent index-priority bias
        if len(idle_idx) > 0:
            rng.shuffle(idle_idx)

        def try_pickup(cab, use_move=False):
            """Checks the FIFO queue at `cab`'s current corner and serves the head-of-line passenger."""
            nonlocal served, sum_queue_min
            q = corner_queues[cell_index(xC[cab], yC[cab])]
            if q:
                a = q.popleft()
                L = sample_trip_blocks()
                busy_until[cab] = t + L

                # Relocate driver to trip destination (partition Manhattan distance L into dx, dy)
                move_x = rng.integers(0, L + 1)
                move_y = L - move_x
                sx = rng.choice([-1, 1])
                sy = rng.choice([-1, 1])
                xC[cab] = wrap_step(wrap_step(xC[cab], sx * move_x, num_blocks), 0, num_blocks)
                yC[cab] = wrap_step(wrap_step(yC[cab], sy * move_y, num_blocks), 0, num_blocks)

                # Record passenger waiting time only after the 1-hour burn-in (t >= 200)
                if t >= burn_in:
                    wait = float((t - a) * time_per_blockmin)
                    if return_individual_waits:
                        individual_waits.append(wait)
                    sum_queue_min += wait
                    served += 1
                return True
            return False

        if idle_idx.size:
            for cab in idle_idx:
                try_pickup(cab)

        # Step 4: Idle Driver Cruising (Random walk on the toroidal grid)
        idle_mask = (busy_until <= t)
        idle_idx  = np.flatnonzero(idle_mask)

        if idle_idx.size and idle_cruise_prob > 0.0:
            will_move = rng.random(idle_idx.size) < idle_cruise_prob
            axes  = rng.integers(0, 2, size=idle_idx.size)
            steps = rng.choice([-1, 1], size=idle_idx.size)
            for k, cab in enumerate(idle_idx):
                if will_move[k]:
                    if axes[k] == 0:
                        xC[cab] = wrap_step(xC[cab], steps[k], num_blocks)
                    else:
                        yC[cab] = wrap_step(yC[cab], steps[k], num_blocks)

        # Step 5: Post-Cruising Pickups at Drivers' Newly Reached Corners
        idle_mask = (busy_until <= t)
        idle_idx  = np.flatnonzero(idle_mask)
        if idle_idx.size:
            rng.shuffle(idle_idx)
            for cab in idle_idx:
                try_pickup(cab)

    mean_w = sum_queue_min / max(1, served)
    res = {"mean_total_wait_min": mean_w}
    if return_individual_waits:
        res["individual_waits"] = np.array(individual_waits)
    return res


# =============================================================================
# 2A. LOOP 1: GENERATE OR LOAD SCALING DATA (`Data/sim_scaling_SH.csv`)
# =============================================================================
if not FORCE_RERUN_SCALING and os.path.exists(SCALING_FILE):
    print("Found existing Scaling Data. Loading...")
    df_scaling = pd.read_csv(SCALING_FILE)
    print(f"Scaling data loaded from {SCALING_FILE}")
else:
    print(f"\n{'='*60}")
    print("LOOP 1: Generating Scaling Data")
    print(f"{'='*60}")
    print(f"Running {NUM_SEEDS} seeds across n_s range {N_S_RANGE.start}-{N_S_RANGE.stop-1}...")

    scaling_data = []
    for seed in range(NUM_SEEDS):
        print(f"  ... Scaling Sweep Seed {seed+1}/{NUM_SEEDS}")
        for n_s in N_S_RANGE:
            res = sim_street_hailing_with_queue(n_s=n_s, seed_init=seed)
            scaling_data.append({
                "seed": seed, "n_s": n_s, "wait": res["mean_total_wait_min"]
            })

    df_scaling = pd.DataFrame(scaling_data)
    df_scaling.to_csv(SCALING_FILE, index=False)
    print(f"✓ Scaling data saved to {SCALING_FILE}")


# =============================================================================
# 2B. LOOP 2: GENERATE OR LOAD DISTRIBUTION DATA (`Data/sim_waits_SH.csv`)
# =============================================================================
if not FORCE_RERUN_WAITS and os.path.exists(WAITS_FILE):
    print("Found existing Distribution Data. Loading...")
    df_waits = pd.read_csv(WAITS_FILE)
    print(f"Distribution data loaded from {WAITS_FILE}")
else:
    print(f"\n{'='*60}")
    print("LOOP 2: Generating Distribution Data")
    print(f"{'='*60}")
    print(f"Running {NUM_SEEDS} seeds at n_s={TARGET_N_S_DIST} (2000 periods each)...")

    all_waits_dist = []
    for seed in range(NUM_SEEDS):
        print(f"  ... Distribution Deep Dive Seed {seed+1}/{NUM_SEEDS} (n_s={TARGET_N_S_DIST})")
        res_dist = sim_street_hailing_with_queue(
            n_s=TARGET_N_S_DIST, num_of_periods=2000,
            seed_init=seed, return_individual_waits=True
        )
        seed_waits = pd.DataFrame({"seed": seed, "wait_min": res_dist["individual_waits"]})
        all_waits_dist.append(seed_waits)

    df_waits = pd.concat(all_waits_dist, ignore_index=True)
    df_waits.to_csv(WAITS_FILE, index=False)
    print(f"✓ Distribution data saved to {WAITS_FILE}")

print(f"\n{'='*60}")
print("Data Ready for Analysis")
print(f"{'='*60}")
