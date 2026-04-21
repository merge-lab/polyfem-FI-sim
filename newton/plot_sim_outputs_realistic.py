import argparse

import numpy as np
import pandas as pd

# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument("--plotter", default="plotly", choices=["plotly", "matplotlib"])
parser.add_argument("--mode", default="volumes", choices=["volumes", "pressures"])
parser.add_argument("--csv", default=None, help="Path to CSV file; defaults to latest file in ./logs/")
args = parser.parse_args()

y_col = {"volumes": "volumes_m3", "pressures": "pressures_atm"}[args.mode]

# --- Data ---
if args.csv is not None:
    csv_path = args.csv
else:
    import glob, os
    logs = sorted(glob.glob("./logs/*.csv"), key=os.path.getmtime)
    csv_path = logs[-1]
df = pd.read_csv(csv_path, index_col=0)

if "volumes_cm3" in df:
    df.rename(columns={"volumes_cm3": "volumes_m3"}, inplace=True)

# Preprocess the csv
df["volumes_m3"] *= 100**3  # Convert cubic meters to cubic cm, for readability
mask_nonzero = df["volumes_m3"] > 0 # Cut out the initial wait
df = df.iloc[mask_nonzero]

# df = df.iloc[args.i_start:]

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
    fig1.add_trace(go.Scatter(x=df["sim_times_s"], y=df[y_col], mode="lines"))
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
    for i_poke, (row, col) in POKE_TO_SUBPLOT.items():
        subset = df[df["i_poke"] == i_poke]
        fig2.add_trace(
            go.Scatter(
                x=subset["sim_times_s"],
                y=subset[y_col],
                mode="lines",
                name=f"i_poke={i_poke}",
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
    ax1.plot(df["sim_times_s"], df[y_col])
    ax1.set_xlabel("sim_time_s")
    ax1.set_ylabel(y_col)
    ax1.set_title("All pokes: volume over time")

    fig2, axes = plt.subplots(3, 3, sharey="all", figsize=(12, 9), layout="constrained")
    fig2.suptitle(f"{y_col} over time by poke index")
    for i_poke, (row, col) in POKE_TO_SUBPLOT.items():
        subset = df[df["i_poke"] == i_poke]
        ax = axes[row, col]
        ax.plot(subset["sim_times_s"], subset[y_col], label=f"i_poke={i_poke}")
        ax.set_title(f"subplot ({row},{col})", fontsize=6)
        ax.legend(fontsize="small")
    for ax in axes.flat:
        ax.set_xlabel("sim_time_s", fontsize=6)
        ax.set_ylim([1 - 0.0008, 1 + 0.0118])
        ax.set_ylabel(y_col)
        ax.grid(visible=True)

    plt.show()
