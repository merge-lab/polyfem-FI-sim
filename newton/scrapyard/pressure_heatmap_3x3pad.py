import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

z_pad_surface = 0.010

data = pd.read_csv("../logs/sim-outputs_03-07-2026_13:02:21.csv")

mask_poking = data["robot_ee_z_m"] <= z_pad_surface + 0.0005

x = data["robot_ee_x_m"].iloc[mask_poking]
y = data["robot_ee_y_m"].iloc[mask_poking]

channel_ids = [1, 2, 3, 4, 5, 6]

fig, axs = plt.subplots(2,3, sharex=True, sharey=True, layout="constrained", figsize=(12, 8))
i_row = 0
for i, id_channel in enumerate(channel_ids):
    i_col = i%3

    ax_i = axs[i_row, i_col]

    z = data[f"p_{id_channel}"].iloc[mask_poking]
    ax_i.plot(x, y, 'o', markersize=1, color="grey")
    ax_i.tricontourf(x, y, z)
    ax_i.set_aspect("equal", "box")
    ax_i.set_title(f"Channel {id_channel} pressure vs robot EE location")
    if i == 2:
        i_row += 1


fig2, axs2 = plt.subplots()
axs2.hist2d(x, y)
axs2.set_title("Histogram of robot EE position")
axs2.set_aspect("equal", "box")
plt.show()