import argparse
import glob as glob_mod
import os
from pathlib import Path

import numpy as np
import pandas as pd

# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument("--plotter", default="plotly", choices=["plotly", "matplotlib"])
parser.add_argument("--mode", default="volumes", choices=["volumes", "pressures"])
# Option 1: explicit list of files, e.g. --csv a.csv b.csv c.csv
parser.add_argument("--csv", nargs="+", default=None, help="One or more CSV paths to plot")
# Option 2: glob pattern that auto-collects matching files, e.g. --glob "./logs/run_*/*.csv"
parser.add_argument("--glob", default=None, help='Glob pattern to select CSVs, e.g. "./logs/run_*/*.csv"')
args = parser.parse_args()

y_col = {"volumes": "volumes_m3", "pressures": "pressures_atm"}[args.mode]

# --- Data ---
# Collect CSV paths from --csv, --glob, or both. If neither is given, fall back to the
# most recently modified file in ./logs/ (original single-file behavior).
csv_paths = []
if args.csv:
    csv_paths.extend(args.csv)
if args.glob:
    # Sort glob matches by modification time so the legend order is chronological.
    csv_paths.extend(sorted(glob_mod.glob(args.glob), key=os.path.getmtime))
if not csv_paths:
    logs = sorted(glob_mod.glob("./logs/*.csv"), key=os.path.getmtime)
    csv_paths = [logs[-1]]


def load_df(path):
    df = pd.read_csv(path, index_col=0)
    if "volumes_cm3" in df:
        df.rename(columns={"volumes_cm3": "volumes_m3"}, inplace=True)
    df["volumes_m3"] *= 100**3  # Convert cubic meters to cubic cm, for readability
    mask_nonzero = df["volumes_m3"] > 0  # Cut out the initial wait
    df = df.iloc[mask_nonzero]
    return df


def label_for(path):
    return Path(path).stem


# Each dataset is keyed by its filename stem, used as the legend label.
datasets = {label_for(p): load_df(p) for p in csv_paths}

# Mapping: i_poke -> (row, col), 0-indexed
POKE_TO_SUBPLOT = {
    0: (1, 0),
    1: (0, 0),
    2: (2, 0),
    3: (2, 1),
    4: (1, 1),
    5: (0, 1),
    6: (0, 2),
    7: (1, 2),
    8: (2, 2),
}

# --- Plotly ---
if args.plotter == "plotly":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig1 = go.Figure()
    for label, df in datasets.items():
        fig1.add_trace(go.Scatter(x=df["sim_times_s"], y=df[y_col], mode="lines", name=label))
    fig1.update_layout(
        title=f"All pokes: {y_col} over time",
        xaxis_title="sim_time_s",
        yaxis_title=y_col,
    )
    fig1.show()

    fig2 = make_subplots(
        rows=3,
        cols=3,
        shared_xaxes=False,
        shared_yaxes="all",
        subplot_titles=[f"subplot ({r},{c})" for r in range(3) for c in range(3)],
        x_title="sim_time_s",
        y_title=y_col,
    )
    for label, df in datasets.items():
        for i_poke, (row, col) in POKE_TO_SUBPLOT.items():
            subset = df[df["i_poke"] == i_poke]
            fig2.add_trace(
                go.Scatter(
                    x=subset["sim_times_s"],
                    y=subset[y_col],
                    mode="lines",
                    name=f"{label} i_poke={i_poke}",
                    # Group traces by dataset so the legend shows one entry per CSV,
                    # not one per poke, when comparing multiple files.
                    legendgroup=label,
                    showlegend=(i_poke == 0),
                ),
                row=row + 1,  # plotly subplots are 1-indexed
                col=col + 1,
            )
    fig2.update_layout(title=f"{y_col} over time by poke index")
    fig2.show()

# --- Matplotlib ---
else:
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots()
    for label, df in datasets.items():
        ax1.plot(df["sim_times_s"], df[y_col], label=label)
    ax1.set_xlabel("sim_time_s")
    ax1.set_ylabel(y_col)
    ax1.set_title("All pokes: volume over time")
    if len(datasets) > 1:
        ax1.legend()

    fig2, axes = plt.subplots(3, 3, sharey="all", figsize=(12, 9), layout="constrained")
    fig2.suptitle(f"{y_col} over time by poke index")
    for label, df in datasets.items():
        for i_poke, (row, col) in POKE_TO_SUBPLOT.items():
            subset = df[df["i_poke"] == i_poke]
            ax = axes[row, col]
            # When comparing multiple CSVs, use the filename as the label; otherwise label by poke index.
            ax.plot(subset["sim_times_s"], subset[y_col], label=label if len(datasets) > 1 else f"i_poke={i_poke}")
            ax.set_title(f"subplot ({row},{col})", fontsize=6)
            ax.legend(fontsize="small")
    for ax in axes.flat:
        ax.set_xlabel("sim_time_s", fontsize=6)
        ax.set_ylim([1 - 0.0008, 1 + 0.0118])
        ax.set_ylabel(y_col)
        ax.grid(visible=True)

    plt.show()
