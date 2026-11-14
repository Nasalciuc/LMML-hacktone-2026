"""Count each crate's glyph in every rubbing of its five sets with the trained detector.

    python count_crates.py --data ../task_china --model detector.pt --thr 0.3

Writes counts.csv (crate, set, glyph_id, count), detections.csv (one row per detected glyph)
and counts_by_threshold.csv (the same counts at several thresholds, to show they are stable).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from detector import Net, detect, load_gray

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("../task_china"))
ap.add_argument("--model", type=Path, default=Path("detector.pt"))
ap.add_argument("--thr", type=float, default=0.3)
a = ap.parse_args()

ck = torch.load(a.model, map_location="cuda")
classes = ck["classes"]
model = Net(len(classes)).cuda()
model.load_state_dict(ck["model"])
model.eval()

crates = pd.read_csv(a.data / "crates.csv")
det_rows, rows = [], []
for crate, gid in zip(crates.crate, crates.glyph_id):
    k = classes.index(gid)
    for set_dir in sorted((a.data / "crates" / crate).iterdir()):
        for f in sorted(set_dir.glob("*.jpg")):
            dets, _ = detect(model, load_gray(f), [k], thr=0.05)
            for _, s, cx, cy, w, h in dets:
                det_rows.append([crate, set_dir.name, f.name, gid, round(s, 4), round(cx, 1), round(cy, 1), round(w, 1), round(h, 1)])
            if not dets:
                det_rows.append([crate, set_dir.name, f.name, gid, 0.0, -1, -1, -1, -1])
det = pd.DataFrame(det_rows, columns=["crate", "set", "image", "glyph_id", "score", "cx", "cy", "w", "h"])
det.to_csv("detections.csv", index=False)

table = []
for thr in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
    c = det[det.score >= thr].groupby(["crate", "set", "glyph_id"]).size()
    for crate, gid in zip(crates.crate, crates.glyph_id):
        for s in sorted(det[det.crate == crate].set.unique()):
            table.append([thr, crate, s, gid, int(c.get((crate, s, gid), 0))])
tab = pd.DataFrame(table, columns=["thr", "crate", "set", "glyph_id", "count"])
tab.to_csv("counts_by_threshold.csv", index=False)
final = tab[np.isclose(tab.thr, a.thr)].drop(columns="thr")
final.to_csv("counts.csv", index=False)
print(tab.pivot_table(index=["crate", "glyph_id"], columns=["thr", "set"], values="count").to_string())
