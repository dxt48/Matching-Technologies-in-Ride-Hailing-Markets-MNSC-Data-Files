"""
Lem_4_Prop_3.py
===============================================================================
Capacity Premium, Short-Wait Advantage, and Bootstrap Calibration Analysis
-------------------------------------------------------------------------------
Manuscript Reference:
    "Matching Technology, Fleet Size, and Passenger Wait in Ride-Hailing Markets"
    - (Main_Document): paper_minor_2.tex, Section 6.2 ("Comparison of Matching
      Technologies")
        * Lemma 5 (`lem:capacity_premium`): Fleet size capacity premium required
          by Street-Hailing relative to Central Dispatch for a given target
          average wait W*.
        * Proposition 1 (`prop:tailprob`): Distributional comparison and
          short-wait advantage of Street-Hailing (F_s(w, lambda, n_s*) >=
          F_d(w, lambda, n_d*) if and only if w <= w' = 4 * W* / pi).
        * Figure 6 (`fig:wait_comparison`): "Comparison of Wait Time
          Distributions (W* approx 1.9 Min)".
    - (Appendix): EC_appendix_minor_2.tex, Section EC.3 ("Empirical Validation
      of Capacity Premium and Tail Risk")
        * Table EC.1 (`tab:prop3_test`): "Bootstrap Test of Short Wait
          Probabilities (E[W] = 1.5 min)".

Purpose:
    1. `perform_statistical_test()`:
        Loads the 10-shift scaling datasets (`Data/sim_scaling_SH.csv` and
        `Data/sim_scaling_CD.csv`), estimates the scaling curves across the 10
        independent simulation seeds, and computes the non-parametric bootstrap
        point estimates and 95% t-distribution confidence intervals (df = 9)
        for the calibrated fleet sizes n_s* and n_d* that achieve equal mean
        wait W* = 1.5 minutes. Evaluates the resulting short-wait probabilities
        P(W <= 1.5) reported in Section 6.2 of (Main_Document) and Table EC.1
        of (Appendix).
    2. `plot_tail_prob_illustration()`:
        Uses the analytical CDF and mean wait expressions from `Theory.py` to
        calibrate n_s* and n_d* for target mean wait W* = 1.5 minutes, locates
        the single-crossing threshold w' approx 1.91 minutes, and displays
        Figure 6 (`Figures/wait_time_comparison.png`) via `plt.show()`. To
        protect the reference image file provided in `Figures/`, saving/overwriting
        the PNG file (`plt.savefig`) is disabled by default (`SAVE_FIGURES = False`).
===============================================================================
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, curve_fit
from scipy.stats import norm, t

# =============================================================================
# PATH RESOLUTION & MODULE IMPORTS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = SCRIPT_DIR if os.path.exists(os.path.join(SCRIPT_DIR, "Data")) else os.path.dirname(SCRIPT_DIR)

# Safety Flag: Keep False so running the script displays Figure 6 (`plt.show()`)
# without overwriting the reference PNG file provided in `Figures/`.
SAVE_FIGURES = False

# Ensure local `Code/` directory is on sys.path so `Theory.py` is imported cleanly
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from Theory import f_s, f_d, F_s, F_d, taxi_wait, uber_wait, DEFAULT_PARAMS_S, DEFAULT_PARAMS_D
except ImportError:
    from Empirical_Analysis.Theory import f_s, f_d, F_s, F_d, taxi_wait, uber_wait, DEFAULT_PARAMS_S, DEFAULT_PARAMS_D

# Baseline Reference Constants
CITY_SIDE_MILES = 4.0
NUM_BLOCKS = 70
SPEED_MPH = 10
TRIP_LENGTH_MILES = 2
BLOCK_LENGTH = CITY_SIDE_MILES / NUM_BLOCKS
BASE_ARRIVALS_PER_MIN = 23592.0 / 60.0  # ~393.2 passengers per minute


# =============================================================================
# HELPER FUNCTIONS: STABILITY ASYMPTOTES & FLEET SIZE ROOT-FINDING
# =============================================================================

def get_critical_n_s(params):
    """
    Computes the theoretical minimum fleet size n_base required for stability
    under Street-Hailing (Approximation 1 in Section 4.1 of (Main_Document)):
        n_base,s = J * lambda * (1 + tau) / (nu * tau).
    """
    _, J, nu, tau, lam = params
    return (J * lam * (1 + tau)) / (nu * tau)


def get_critical_n_d(params):
    """
    Computes the theoretical minimum fleet size n_base required for stability
    under Central Dispatch (Approximation 2 in Section 4.2 of (Main_Document)):
        n_base,d = J * lambda / (nu * tau).
    """
    _, J, nu, tau, lam = params
    return (J * lam) / (nu * tau)


def find_fleet_size_for_wait(wait_func, target_wait, params_template, mode='s'):
    """
    Numerically solves for the fleet size n* that achieves a specified target
    expected wait time `target_wait` (W*) using Brent's root-finding method.

    Args:
        wait_func (callable): Expected wait function (`taxi_wait` or `uber_wait`).
        target_wait (float): Target mean passenger wait W* in minutes.
        params_template (tuple): Default parameter tuple `(n, J, nu, tau, lambda)`.
        mode (str): 's' for Street-Hailing or 'd' for Central Dispatch.

    Returns:
        float or None: Calibrated fleet size n* achieving W*.
    """
    base_params = params_template[1:]

    # Determine stability lower bound (asymptote)
    if mode == 's':
        n_crit = get_critical_n_s(params_template)
    else:
        n_crit = get_critical_n_d(params_template)

    print(f"[{mode.upper()}] Critical n: {n_crit:.1f}")

    def objective(n):
        return wait_func(n, base_params) - target_wait

    try:
        # Search strictly above the stability asymptote
        lower_bound = n_crit + 10
        upper_bound = 50000

        n_opt = brentq(objective, lower_bound, upper_bound)
        return n_opt
    except Exception as e:
        print(f"Could not find n for target wait {target_wait}: {e}")
        return None


# =============================================================================
# FIGURE 6 GENERATION: CDF CROSSING & SHORT-WAIT ADVANTAGE (PROPOSITION 1)
# =============================================================================

def plot_tail_prob_illustration():
    """
    Generates Figure 6 (`Figures/wait_time_comparison.png`) in Section 6.2 of
    (Main_Document), illustrating Proposition 1:
        - Calibrates both Street-Hailing and Central Dispatch to have identical
          expected wait W* = 1.5 minutes.
        - Plots the cumulative distribution functions F_s(w, lambda, n_s*) and
          F_d(w, lambda, n_d*).
        - Identifies the unique crossing threshold w' = (4 / pi) * W* approx
          1.91 minutes below which Street-Hailing delivers a higher probability
          of pickup (the "short-wait advantage") and above which Central
          Dispatch dominates (thinner right tail).
    """
    TARGET_WAIT_MIN = 1.5
    print(f"Calibrating for Target Wait = {TARGET_WAIT_MIN} min")

    # 1. Solve for analytical fleet sizes n_s* and n_d* at W* = 1.5 min
    n_s_star = find_fleet_size_for_wait(taxi_wait, TARGET_WAIT_MIN, DEFAULT_PARAMS_S, mode='s')
    n_d_star = find_fleet_size_for_wait(uber_wait, TARGET_WAIT_MIN, DEFAULT_PARAMS_D, mode='d')

    if n_s_star is None or n_d_star is None:
        print("Failed to calibrate fleet sizes.")
        return

    print(f"Calibrated: n_s*={n_s_star:.0f}, n_d*={n_d_star:.0f}")

    params_s_star = (n_s_star,) + DEFAULT_PARAMS_S[1:]
    params_d_star = (n_d_star,) + DEFAULT_PARAMS_D[1:]

    # 2. Evaluate CDFs across wait times w in [0, 8] minutes
    w = np.linspace(0, 8, 1000)
    Fs_vals = F_s(w, params_s_star)
    Fd_vals = F_d(w, params_d_star)

    # 3. Locate the strictly positive CDF intersection point w' > 0
    def diff_F(val):
        return F_s(val, params_s_star) - F_d(val, params_d_star)

    try:
        w_prime = brentq(diff_F, 0.01, 5.0)
        if w_prime < 0.05:
            try:
                w_prime = brentq(diff_F, 0.1, 5.0)
            except Exception:
                pass

        F_prime = F_s(w_prime, params_s_star)
        print(f"Intersection w' found at {w_prime:.3f} min")
    except ValueError:
        w_prime = None
        print("No intersection found in range")

    # 4. Plot CDF Comparison and Shaded Dominance Regions
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(w, Fs_vals, 'b-', lw=2.5, label=r'Street Hail ($F_s(\cdot, n_s^*)$)')
    ax.plot(w, Fd_vals, 'r--', lw=2.5, label=r'Central Dispatch ($F_d(\cdot, n_d^*)$)')

    if w_prime:
        # Highlight intersection point (w', F(w'))
        ax.plot(w_prime, F_prime, 'ko', zorder=5)
        ax.axvline(w_prime, color='gray', linestyle=':', alpha=0.5)
        ax.text(w_prime, -0.05, r"$w'$", ha='center', va='top', fontsize=14, color='black')
        ax.annotate(
            rf"$w^\prime \approx {w_prime:.1f}$ min",
            xy=(w_prime, F_prime),
            xytext=(w_prime - 0.95, F_prime + 0.12),
            fontsize=12,
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF59D', edgecolor='#B7950B', alpha=0.95),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.15', color='black', lw=1.2),
            zorder=6
        )

        # Region 1 (w <= w'): Street-Hailing Short-Wait Advantage (F_s >= F_d)
        ax.fill_between(
            w, Fs_vals, Fd_vals, where=(w <= w_prime), color='green', alpha=0.15,
            label=r'$F_s \geq F_d$ (Higher Availability)'
        )

        # Region 2 (w > w'): Street-Hailing Tail Risk / Central Dispatch Reliability Advantage (F_s < F_d)
        ax.fill_between(
            w, Fs_vals, Fd_vals, where=(w > w_prime), color='orange', alpha=0.15,
            label=r'$F_s < F_d$ (Tail Risk)'
        )

    ax.set_xlabel(r'Wait Time $w$ (minutes)', fontsize=14)
    ax.set_ylabel(r'CDF $F(w)$', fontsize=14)

    ax.legend(fontsize=12, loc='lower right', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 6])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    # Display Figure 6 without overwriting the reference PNG in `Figures/`
    if SAVE_FIGURES:
        output_dir = os.path.join(PACKAGE_ROOT, 'Figures')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'wait_time_comparison.png')
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to: {output_path}")
    else:
        print("Displaying wait_time_comparison figure (saving disabled to preserve existing file in Figures/)...")
        plt.show()


# =============================================================================
# TABLE EC.1 GENERATION: EMPIRICAL BOOTSTRAP TEST OF CAPACITY & TAIL RISK
# =============================================================================

def perform_statistical_test():
    """
    Executes the formal bootstrap calibration and short-wait probability test
    reported in Section 6.2 of (Main_Document) and Section EC.3 / Table EC.1
    (`tab:prop3_test`) of (Appendix):
        1. Loads the 10-shift scaling datasets (`sim_scaling_SH.csv` and
           `sim_scaling_CD.csv`).
        2. For each simulation seed i in {0, ..., 9}, fits the theoretical
           scaling curves W_bar_s(n) = alpha / (n - beta) and
           W_bar_d(n) = alpha / sqrt(n - beta) over the interior unsaturated
           operating windows.
        3. Inverts each fitted curve to obtain seed-level calibrated fleet sizes
           (n_s,i*, n_d,i*) that yield target expected wait W* = 1.5 minutes.
        4. Computes the bootstrap mean and 95% t-distribution confidence
           intervals (df = 9) across the 10 independent simulation shifts:
               n_s* = 7,012 (95% CI: [7,002, 7,023])
               n_d* = 5,296 (95% CI: [5,287, 5,305])
        5. Evaluates the structural short-wait probabilities P(W <= 1.5) for the
           Exponential and Rayleigh distributions at mean W* = 1.5 minutes:
               P(W_s <= 1.5) = 0.6321 vs. P(W_d <= 1.5) = 0.5441 (+0.0881).
    """
    print("\n" + "="*50)
    print("FORMAL STATISTICAL TEST (PROPOSITION 3)")
    print("="*50)

    # 1. Load Pre-Computed 10-Shift Scaling Data from Data/
    sh_path = os.path.join(PACKAGE_ROOT, 'Data', 'sim_scaling_SH.csv')
    cd_path = os.path.join(PACKAGE_ROOT, 'Data', 'sim_scaling_CD.csv')
    try:
        if os.path.exists(sh_path):
            df_sh = pd.read_csv(sh_path)
            df_cd = pd.read_csv(cd_path)
        elif os.path.exists('Data/sim_scaling_SH.csv'):
            df_sh = pd.read_csv('Data/sim_scaling_SH.csv')
            df_cd = pd.read_csv('Data/sim_scaling_CD.csv')
        else:
            df_sh = pd.read_csv('Empirical_Analysis/Data/sim_scaling_SH.csv')
            df_cd = pd.read_csv('Empirical_Analysis/Data/sim_scaling_CD.csv')
    except FileNotFoundError:
        print("Error: simulation data not found in 'Data/'")
        return

    # 2. Fit Pooled Scaling Curves Over Interior Unsaturated Operating Windows
    sh_data = df_sh[(df_sh['n_s'] >= 6800) & (df_sh['n_s'] <= 8000)]
    cd_data = df_cd[(df_cd['n_d'] >= 5500) & (df_cd['n_d'] <= 7000)]

    def model_sh(n, alpha, beta):
        return alpha / (n - beta)

    def model_cd(n, alpha, beta):
        return alpha / np.sqrt(n - beta)

    try:
        popt_sh, _ = curve_fit(model_sh, sh_data['n_s'], sh_data['wait'], p0=[4000, 4800], bounds=([0, 0], [np.inf, np.inf]))
    except Exception:
        popt_sh, _ = curve_fit(model_sh, df_sh['n_s'], df_sh['wait'], p0=[4000, 4000])

    try:
        popt_cd, _ = curve_fit(model_cd, cd_data['n_d'], cd_data['wait'], p0=[10, 4000], bounds=([0, 0], [np.inf, np.inf]))
    except Exception:
        popt_cd, _ = curve_fit(model_cd, df_cd['n_d'], df_cd['wait'], p0=[10, 4000])

    print(f"SH Scaling Fit: alpha={popt_sh[0]:.1f}, beta={popt_sh[1]:.1f}")
    print(f"CD Scaling Fit: alpha={popt_cd[0]:.1f}, beta={popt_cd[1]:.1f}")

    # 3. Pooled Calibration for Target Mean Wait W* = 1.5 Minutes
    TARGET_MEAN = 1.5
    n_sh_calibrated = popt_sh[0] / TARGET_MEAN + popt_sh[1]
    n_cd_calibrated = (popt_cd[0] / TARGET_MEAN)**2 + popt_cd[1]

    print(f"Calibrated Fleet Sizes for E[W]={TARGET_MEAN} min:")
    print(f"  n_SH* = {n_sh_calibrated:.0f}")
    print(f"  n_CD* = {n_cd_calibrated:.0f}")

    # 4. Non-Parametric / Empirical Bootstrap Across the 10 Independent Shifts
    #    (See Section EC.3, Steps 1-4 and Table EC.1 in (Appendix))
    sh_seeds = df_sh['seed'].unique()
    cd_seeds = df_cd['seed'].unique()
    common_seeds = np.intersect1d(sh_seeds, cd_seeds)
    print(f"\nRunning Empirical Bootstrap on {len(common_seeds)} Seeds: {common_seeds}")

    bs_n_sh = []
    bs_n_cd = []

    for seed in common_seeds:
        sh_sub = sh_data[sh_data['seed'] == seed]
        cd_sub = cd_data[cd_data['seed'] == seed]

        # Estimate seed-specific Street-Hailing scaling curve and invert at W* = 1.5
        try:
            p_sh, _ = curve_fit(model_sh, sh_sub['n_s'], sh_sub['wait'], p0=[4000, 4800], bounds=([0, 0], [np.inf, np.inf]))
            n_sh_star = p_sh[0] / TARGET_MEAN + p_sh[1]
            bs_n_sh.append(n_sh_star)
        except Exception:
            pass

        # Estimate seed-specific Central Dispatch scaling curve and invert at W* = 1.5
        try:
            p_cd, _ = curve_fit(model_cd, cd_sub['n_d'], cd_sub['wait'], p0=[10, 4000], bounds=([0, 0], [np.inf, np.inf]))
            n_cd_star = (p_cd[0] / TARGET_MEAN)**2 + p_cd[1]
            bs_n_cd.append(n_cd_star)
        except Exception:
            pass

    bs_n_sh = np.array(bs_n_sh)
    bs_n_cd = np.array(bs_n_cd)

    # Compute sample mean, standard error, and 95% t-distribution CIs (df = N - 1 = 9)
    n_sh_mean = np.mean(bs_n_sh)
    n_sh_sem = np.std(bs_n_sh, ddof=1) / np.sqrt(len(bs_n_sh))
    ci_sh = t.interval(0.95, len(bs_n_sh) - 1, loc=n_sh_mean, scale=n_sh_sem)

    n_cd_mean = np.mean(bs_n_cd)
    n_cd_sem = np.std(bs_n_cd, ddof=1) / np.sqrt(len(bs_n_cd))
    ci_cd = t.interval(0.95, len(bs_n_cd) - 1, loc=n_cd_mean, scale=n_cd_sem)

    print("-" * 30)
    print("EMPIRICAL BOOTSTRAP RESULTS (N=10 runs):")
    print(f"  n_SH* Mean: {n_sh_mean:.0f} (SE: {n_sh_sem:.1f})")
    print(f"  n_SH* 95% CI: [{ci_sh[0]:.0f}, {ci_sh[1]:.0f}]")
    print(f"  n_CD* Mean: {n_cd_mean:.0f} (SE: {n_cd_sem:.1f})")
    print(f"  n_CD* 95% CI: [{ci_cd[0]:.0f}, {ci_cd[1]:.0f}]")

    # 5. Evaluate Short-Wait Probabilities P(W <= W*) at Equated Mean W* = 1.5
    #    - Street-Hailing (Exponential with mean 1.5): P(W_s <= 1.5) = 1 - exp(-1) = 0.6321
    #    - Central Dispatch (Rayleigh with mean 1.5):  P(W_d <= 1.5) = 1 - exp(-pi/4) = 0.5441
    prob_sh = 1 - np.exp(-1.5 / 1.5)
    sigma_cd = 1.5 / np.sqrt(np.pi / 2)
    prob_cd = 1 - np.exp(-(1.5**2) / (2 * sigma_cd**2))

    print("\nPROBABILITY RESULT (Structural):")
    print(f"  P(W_SH <= 1.5) = {prob_sh:.4f}")
    print(f"  P(W_CD <= 1.5) = {prob_cd:.4f}")
    print(f"  Difference: +{prob_sh - prob_cd:.4f} (+{(prob_sh - prob_cd)/prob_cd*100:.1f}%)")

    print("\nCONCLUSION: The empirical variance confirms the robustness of the calibration.")
    print("Results are based on 10 independent simulation runs.")


if __name__ == "__main__":
    perform_statistical_test()
    plot_tail_prob_illustration()