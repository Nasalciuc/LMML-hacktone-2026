"""Draw the album sheets from the saved page heatmaps.

For every album two views on the same 1600x1200 page frame:
  * top-1: the strongest region of every page (as in boxes.csv), dot size by confidence;
  * overlay: the sum over the 26 pages of each page's heatmap, normalised to its own peak.
    Real edits fall on the strokes of the letter and add up; false alarms scatter.

    python sheets.py --dir outputs
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", type=Path, default=Path("outputs"))
a = ap.parse_args()

H = np.load(a.dir / "heatmaps_400x300.npz")
boxes = pd.read_csv(a.dir / "boxes.csv")
albums = sorted(boxes.album.unique())
top, over = [], []
for al in albums:
    s = np.full((300, 400, 3), 255, np.uint8)
    for r in boxes[boxes.album == al].itertuples():
        g = int(200 * (1 - min(1.0, r.score)))
        cv2.circle(s, (int(r.cx / 4), int(r.cy / 4)), 7, (g, g, g), -1)
    cv2.putText(s, al, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    top.append(s)
    acc = np.zeros((300, 400), np.float32)
    for k in H.files:
        if k.startswith(al + "__"):
            h = cv2.GaussianBlur(H[k].astype(np.float32), (0, 0), 2)
            acc += h / (h.max() + 1e-6)
    acc = cv2.GaussianBlur(acc, (0, 0), 2)
    v = 255 - (np.clip(acc / acc.max(), 0, 1) * 255).astype(np.uint8)
    over.append(cv2.cvtColor(v, cv2.COLOR_GRAY2BGR))
for t in top + over:
    cv2.rectangle(t, (0, 0), (399, 299), (180, 180, 180), 1)
cv2.imwrite(str(a.dir / "album_sheets_top1.png"), np.hstack(top))
cv2.imwrite(str(a.dir / "album_sheets_overlay.png"), np.hstack(over))
print("wrote", a.dir / "album_sheets_top1.png", a.dir / "album_sheets_overlay.png")
