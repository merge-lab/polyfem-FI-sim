"""
Preprocess real tensile-tester poke data so the 9 first-trial CSVs can be
replayed sequentially inside the Newton simulation.

For each poke site (idx 1-9), trial 1:
  - Detects when compression actually starts (first sample where force > threshold)
  - Shifts timestamps so pokes run one after another
  - Converts position_mm to simulation z-coordinates (metres)
  - Saves an adjusted CSV: newton/data/adjusted_pokes/adjusted_idx{i}_1.csv
  - Saves a poke_windows.csv with t_start / t_end / t_compression_start per poke

Outputs are consumed by sim_thighpad_realistic.py.

Run from the project root:
    python newton/process_poke_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── constants ───────────────────────────────────────────────────────────────
DATA_DIR        = Path("../Data/Gilbert/Thigh_pad/pokes")
OUT_DIR         = Path("../Data/Gilbert/Thighpad/pokes/adjusted")
N_POKES         = 9
TRIAL           = 1
FORCE_THRESHOLD = 0.01      # N — same threshold used in the notebook
POKE1_START_S   = 10.0      # desired sim-time (s) for start of first poke CSV
INTER_POKE_GAP  = 5.0       # s of "dead time" between the end of one CSV and start of next

# Simulation contact-surface reference position (metres).
# x_compression_start in the real data maps to this z value in the sim.
Z_ZERO_M = -0.086

# Real poke numbering:
# | 1 | 2 | 3 |
# |---+---+---|
# | 4 | 5 | 6 |
# |---+---+---|
# | 7 | 8 | 9 |
# Simulated poke numbering
# | 2 | 6 | 7 |
# |---+---+---|
# | 1 | 5 | 8 |
# |---+---+---|
# | 3 | 4 | 9 |
dict_poke_idx_real2sim = {
    1: 2,
    2: 6,
    3: 7,
    4: 1,
    5: 5,
    6: 8,
    7: 3,
    8: 4,
    9: 9
}


def load_csv(i_poke: int, trial: int) -> pd.DataFrame:
    path = DATA_DIR / f"experiment_idx{i_poke}_{trial}_results.csv"
    return pd.read_csv(path)


def find_compression_start(df: pd.DataFrame) -> tuple[int, float, float]:
    """Return (row_index, t_compression_start, x_compression_start_mm)."""
    mask = df["force_N"] > FORCE_THRESHOLD
    i = int(np.nonzero(mask.values)[0][0])
    return i, float(df["time_s"].iloc[i]), float(df["position_mm"].iloc[i])


def pos_mm_to_z_m(position_mm: np.ndarray, x_contact_mm: float) -> np.ndarray:
    """
    Map tensile-tester crosshead position (mm) to simulation z-coordinate (m).

    Contact surface  →  z = Z_ZERO_M
    Deeper (smaller position_mm)  →  more negative z  (probe pressing in)
    """
    return Z_ZERO_M + (position_mm - x_contact_mm) / 1000.0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    windows_rows = []
    next_start_s = POKE1_START_S  # desired shifted start time of the current CSV

    fig, ax = plt.subplots(figsize=(14, 4))

    for i_poke in range(1, N_POKES + 1):
        df = load_csv(i_poke, TRIAL)

        i_comp, t_comp_raw, x_comp_mm = find_compression_start(df)
        x_bottom_mm = float(df["position_mm"].min())
        dx_mm = x_bottom_mm - x_comp_mm  # negative (compression)

        print(
            f"Real poke {i_poke}: compression starts at raw t={t_comp_raw:.3f}s, "
            f"x_contact={x_comp_mm:.3f}mm, x_bottom={x_bottom_mm:.3f}mm, "
            f"dx={dx_mm:.3f}mm"
        )

        # ── time shifting ────────────────────────────────────────────────────
        t_csv_start = float(df["time_s"].iloc[0])
        time_offset  = next_start_s - t_csv_start
        shifted_time = df["time_s"].values + time_offset

        t_start_shifted = shifted_time[0]
        t_end_shifted   = shifted_time[-1]
        t_comp_shifted  = t_comp_raw + time_offset

        # ── position conversion ──────────────────────────────────────────────
        z_m = pos_mm_to_z_m(df["position_mm"].values, x_comp_mm)

        # ── save adjusted CSV ────────────────────────────────────────────────
        df_out = pd.DataFrame({
            "time_s":       shifted_time,
            "z_position_m": z_m,
            "position_mm":  df["position_mm"].values,
            "force_N":      df["force_N"].values,
            "pressure_atm": df["pressure_atm"].values,
        })
        i_poke_sim = dict_poke_idx_real2sim[i_poke]
        out_path = OUT_DIR / f"adjusted_idx{i_poke_sim}_1.csv"
        df_out.to_csv(out_path, index=False)
        print(f"  → saved {out_path}  (shifted t=[{t_start_shifted:.2f}, {t_end_shifted:.2f}]s)")

        windows_rows.append({
            "i_poke":            i_poke - 1,   # 0-based index used by the sim
            "t_start":           t_start_shifted,
            "t_end":             t_end_shifted,
            "t_compression_start": t_comp_shifted,
        })

        # ── plot this poke ───────────────────────────────────────────────────
        ax.plot(shifted_time, z_m * 1000, label=f"Poke {i_poke}")   # z in mm for readability

        # ── advance start time for next poke ─────────────────────────────────
        next_start_s = t_end_shifted + INTER_POKE_GAP

    # ── save poke windows lookup ─────────────────────────────────────────────
    df_windows = pd.DataFrame(windows_rows)
    windows_path = OUT_DIR / "poke_windows.csv"
    df_windows.to_csv(windows_path, index=False)
    print(f"\nSaved poke windows → {windows_path}")
    print(df_windows.to_string(index=False))

    # ── finalise plot ────────────────────────────────────────────────────────
    ax.set_xlabel("Shifted simulation time [s]")
    ax.set_ylabel("Probe z position [mm]")
    ax.set_title("Real poke trajectories — all 9 pokes sequenced")
    ax.legend(ncols=3, fontsize=8)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
