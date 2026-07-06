import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
import matplotlib.colorizer as mcolorizer
import matplotlib.colors as mcolors

z_pad_surface = 0.010

data = pd.read_csv("../logs/EXP5_i01_sim-outputs_03-07-2026_14:10:02.csv")
SUBTRACT_MEDIAN = True

mask_poking = data["robot_ee_z_m"] <= z_pad_surface + 0.0005

x = data["robot_ee_x_m"].iloc[mask_poking]
y = data["robot_ee_y_m"].iloc[mask_poking]

x_range = [np.min(x), np.max(x)]
y_range = [np.min(y), np.max(y)]

n_bins = 200
x_interp = np.linspace(x_range[0], x_range[1], n_bins)
y_interp = np.linspace(y_range[0], y_range[1], n_bins)
binsize_x = x_interp[1] - x_interp[0]
binsize_y = y_interp[1] - y_interp[0]

X_interp, Y_interp = np.meshgrid(x_interp, y_interp)

channel_ids = [1, 2, 3, 4, 5, 6]

fig, axs = plt.subplots(2,3, sharex=True, sharey=True, layout="constrained", figsize=(12, 8))
i_row = 0

df_all_pressures = data.filter(regex="p_.*")
if SUBTRACT_MEDIAN:
    mat_all_ps = df_all_pressures.values
    mat_all_ps_rebalanced = mat_all_ps - np.tile(np.median(mat_all_ps, axis=0), (mat_all_ps.shape[0], 1))
    cmap_range = [np.min(mat_all_ps_rebalanced), np.max(mat_all_ps_rebalanced)]
else:
    cmap_range = [np.min(df_all_pressures), np.max(df_all_pressures)]

max_extent = np.max(np.abs(cmap_range))
cmap_range_sym = [-max_extent, max_extent]

norm = mcolors.Normalize(vmin=cmap_range_sym[0], vmax=cmap_range_sym[1])
colorizer = mcolorizer.Colorizer(norm=norm, cmap="RdBu")

images = []
img_extent = [x_interp[0] - binsize_x, x_interp[-1] + binsize_x, y_interp[0] - binsize_y, y_interp[-1] + binsize_y]

for i, id_channel in enumerate(channel_ids):
    i_col = i%3

    ax_i = axs[i_row, i_col]

    # Fetch channel_i's sensitivity map (resulting pressure from pokes at different places)
    z = data[f"p_{id_channel}"].iloc[mask_poking]
    if SUBTRACT_MEDIAN:
        z = z - np.median(z)

    # Interpolate sensitivity map over channel surface
    f_interp = NearestNDInterpolator(list(zip(x, y)), z)
    Z_interp = f_interp(X_interp, Y_interp)

    ax_i.plot(x, y, ',', markersize=0.1, color="grey", alpha=0.2)
    image_i = ax_i.imshow(Z_interp, origin="lower", colorizer=colorizer, extent=img_extent)

    # ax_i.set_xticks(range(n_bins), labels=x_interp)
    # ax_i.set_yticks(range(n_bins), labels=y_interp)

    # ax_i.tricontourf(x, y, z)
    ax_i.set_aspect("equal", "box")
    ax_i.set_title(f"Channel {id_channel} Δp")

    images.append(image_i)

    if i == 2:
        i_row += 1

fig.supxlabel("Indentor x position [m]")
fig.supylabel("Indentor y position [m]")
cbar = fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1)
cbar.ax.tick_params(axis="y", labelrotation=90)
cbar.ax.set_ylabel("Δp [atm]")


fig2, axs2 = plt.subplots()
axs2.hist2d(x, y, bins=n_bins)
axs2.set_title("Histogram of robot EE position")
axs2.set_aspect("equal", "box")
plt.show()