import argparse

import numpy as np
import pandas as pd

# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument("--plotter", default="plotly", choices=["plotly", "matplotlib"])
args = parser.parse_args()

# --- Data ---
# df = pd.read_csv("sim_outputs_realistic.csv", index_col=0)
df = pd.read_csv("sim-outputs_13-04-2026_21:25:43.csv", index_col=0)

df["volumes_cm3"] *= 100**3  # Oops it's actually outputting it in cubic meters...
df = df.iloc[10:]

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
    fig1.add_trace(go.Scatter(x=df["sim_times_s"], y=df["volumes_cm3"], mode="lines"))
    fig1.update_layout(
        title="All pokes: volume over time",
        xaxis_title="sim_time_s",
        yaxis_title="volumes_cm3",
    )
    fig1.show()

    fig2 = make_subplots(
        rows=3,
        cols=3,
        shared_xaxes=False,
        shared_yaxes=True,
        subplot_titles=[f"subplot ({r},{c})" for r in range(3) for c in range(3)],
        x_title="sim_time_s",
        y_title="volumes_cm3",
    )
    for i_poke, (row, col) in POKE_TO_SUBPLOT.items():
        subset = df[df["i_poke"] == i_poke]
        fig2.add_trace(
            go.Scatter(
                x=subset["sim_times_s"],
                y=subset["volumes_cm3"],
                mode="lines",
                name=f"i_poke={i_poke}",
            ),
            row=row + 1,  # plotly subplots are 1-indexed
            col=col + 1,
        )
    fig2.update_layout(title="Volume over time by poke index")
    fig2.show()

# --- Matplotlib ---
else:
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots()
    ax1.plot(df["sim_times_s"], df["volumes_cm3"])
    ax1.set_xlabel("sim_time_s")
    ax1.set_ylabel("volumes_cm3")
    ax1.set_title("All pokes: volume over time")

    fig2, axes = plt.subplots(3, 3, sharey=True, figsize=(12, 9))
    fig2.suptitle("Volume over time by poke index")
    for i_poke, (row, col) in POKE_TO_SUBPLOT.items():
        subset = df[df["i_poke"] == i_poke]
        ax = axes[row, col]
        ax.plot(subset["sim_times_s"], subset["volumes_cm3"], label=f"i_poke={i_poke}")
        ax.set_title(f"subplot ({row},{col})")
        ax.legend(fontsize="small")
    for ax in axes.flat:
        ax.set_xlabel("sim_time_s")
        ax.set_ylabel("volumes_cm3")
    fig2.tight_layout()

    plt.show()
