"""
run_all.py
===============================================================================
One-Click Master Replication Script (Non-Destructive: Preserves Data & Figures)
-------------------------------------------------------------------------------
Manuscript Reference:
    "Matching Technology, Fleet Size, and Passenger Wait in Ride-Hailing Markets"
    Authors: Kaitlin M. Daniels and Danko Turcic
    Journal: Management Science

Purpose:
    Executes the complete empirical validation, statistical estimation, and
    figure drawing pipeline using the EXISTING pre-computed 10-shift
    simulation CSV files located in `Data/`, while preserving all provided
    reference files in `Data/` and `Figures/` without overwriting them.

    Specifically, running `python run_all.py` from `Empirical_Analysis_Code_MNSC/`
    performs the following steps:
        1. Verifies that all 6 pre-computed simulation CSV files are present in
           `Data/` (and loads them via `SH_10.py`, `CD_10.py`, and `CD_10_Batch.py`
           with `FORCE_RERUN_SCALING = False` and `FORCE_RERUN_WAITS = False`).
        2. Executes `Code/Master_Analysis.py`:
            - Prints Table 3 (`tab:scaling_audit_combined`) in (Main_Document).
            - Draws and displays (`plt.show()`) Figures 3, 4, and 5 without
              overwriting the existing PNG files in `Figures/` (`SAVE_FIGURES = False`).
        3. Executes `Code/Lem_4_Prop_3.py`:
            - Prints the 10-seed bootstrap fleet calibration (n_s* = 7,012;
              n_d* = 5,296) and short-wait probability test results reported in
              Section 6.2 of (Main_Document) and Table EC.1 (`tab:prop3_test`)
              of (Appendix).
            - Draws and displays (`plt.show()`) Figure 6 without overwriting the
              existing PNG file in `Figures/` (`SAVE_FIGURES = False`).
===============================================================================
"""

import os
import runpy
import sys
import time


def main():
    start_time = time.time()

    # Resolve package root and subdirectories
    package_root = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(package_root, "Code")
    data_dir = os.path.join(package_root, "Data")
    fig_dir = os.path.join(package_root, "Figures")

    os.chdir(package_root)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    print("=" * 78)
    print("MANAGEMENT SCIENCE REPLICATION PACKAGE: MASTER EXECUTION")
    print("Paper: Matching Technology, Fleet Size, and Passenger Wait in Ride-Hailing Markets")
    print("Mode : Non-Destructive Audit (Uses Existing CSVs in Data/ & Preserves Figures/)")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # Step 1: Verify Existing Pre-Computed Simulation CSV Files in Data/
    # -------------------------------------------------------------------------
    required_csvs = [
        "sim_scaling_SH.csv",
        "sim_waits_SH.csv",
        "sim_scaling_CD.csv",
        "sim_waits_CD.csv",
        "sim_scaling_CD_batched_w2.csv",
        "sim_waits_CD_batched_w2.csv",
    ]
    print("\n[Step 1/3] Verifying existing pre-computed simulation CSV files in Data/...")
    for fname in required_csvs:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required data file missing: {fpath}")
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  ✓ Found Data/{fname} ({size_mb:.2f} MB)")

    # Run simulation scripts in default load-only mode (FORCE_RERUN_* = False)
    for sim_script in ["SH_10.py", "CD_10.py", "CD_10_Batch.py"]:
        runpy.run_path(os.path.join(code_dir, sim_script), run_name="__main__")

    # -------------------------------------------------------------------------
    # Step 2: Run Master_Analysis.py (Prints Table 3 & Displays Figures 3, 4, 5)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("[Step 2/3] Running Code/Master_Analysis.py (Table 3 & Displaying Figures 3, 4, 5)...")
    print("=" * 78)
    runpy.run_path(os.path.join(code_dir, "Master_Analysis.py"), run_name="__main__")

    # -------------------------------------------------------------------------
    # Step 3: Run Lem_4_Prop_3.py (Prints Table EC.1, Section 6.2 & Displays Figure 6)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("[Step 3/3] Running Code/Lem_4_Prop_3.py (Table EC.1, Section 6.2 & Displaying Figure 6)...")
    print("=" * 78)
    runpy.run_path(os.path.join(code_dir, "Lem_4_Prop_3.py"), run_name="__main__")

    elapsed = time.time() - start_time
    print("\n" + "=" * 78)
    print(f"REPLICATION COMPLETE in {elapsed:.2f} seconds.")
    print(f"Note: Existing CSV files in {data_dir} and PNG figures in {fig_dir} were preserved without overwriting.")
    print("=" * 78)


if __name__ == "__main__":
    main()
