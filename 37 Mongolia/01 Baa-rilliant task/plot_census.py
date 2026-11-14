"""Place each survey count at its plot position (survey_plots.csv) and draw the census map.

The plots lie on a regular 50 m grid, 37 x 9, rotated by about 27 degrees. The flight log is
rotated back so the grid is axis-aligned (a proper rotation, no mirroring), every plot is snapped
to its cell and the cell is shaded by its sheep count.

    python plot_census.py --data ../task_mongolia --counts counts.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("../task_mongolia"))
ap.add_argument("--counts", type=Path, default=Path("counts.csv"))
ap.add_argument("--col", default="count")
ap.add_argument("--out", type=Path, default=Path("census_map.png"))
a = ap.parse_args()

plots = pd.read_csv(a.data / "survey_plots.csv")
df = plots.merge(pd.read_csv(a.counts), on="image")
XY = df[["x_m", "y_m"]].values - df[["x_m", "y_m"]].values.mean(0)

# grid direction from the principal axis, turned into a proper rotation
u, s, vt = np.linalg.svd(XY, full_matrices=False)
ang = np.arctan2(vt[0, 1], vt[0, 0])
if np.cos(ang) < 0:
    ang += np.pi
rot = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
R = XY @ rot.T
step = 50.0
col = np.round((R[:, 0] - R[:, 0].min()) / step).astype(int)
row = np.round((R[:, 1] - R[:, 1].min()) / step).astype(int)
G = np.full((row.max() + 1, col.max() + 1), np.nan)
G[row, col] = df[a.col].values
G = G[::-1]  # north (larger y) on top
print(f"grid rotation {np.degrees(ang):.1f} deg, {G.shape[1]} x {G.shape[0]} cells, "
      f"{np.isfinite(G).sum()} filled")

fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [1.2, 1, 1]})
ax = axes[0]
sc = ax.scatter(df.x_m, df.y_m, c=df[a.col], s=60, cmap="Greys", marker="s")
plt.colorbar(sc, ax=ax, label="sheep counted")
ax.set_aspect("equal"); ax.set_title("Counts at the flight-log positions")
ax = axes[1]
im = ax.imshow(G, cmap="Greys", interpolation="nearest")
plt.colorbar(im, ax=ax, label="sheep counted")
ax.set_title(f"Grid rotated by {-np.degrees(ang):.0f} deg, one cell per plot")
ax.set_xticks([]); ax.set_yticks([])
ax = axes[2]
thr = 47
ax.imshow(G > thr, cmap="Greys", interpolation="nearest")
ax.set_title(f"Cells with more than {thr} sheep")
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(a.out, dpi=100)
np.savetxt(a.out.with_suffix(".grid.csv"), G, fmt="%.1f", delimiter=",")
print("saved", a.out)
