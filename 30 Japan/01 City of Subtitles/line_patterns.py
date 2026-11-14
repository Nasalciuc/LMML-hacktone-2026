"""Row / column patterns of the noise residual (the part the PRNU pipeline throws away with zero-meaning).
For each photographer, average over its photos (sensor orientation) of the column means and row means of the
raw wavelet residual, per colour channel. Saves cache/line_patterns.npz and prints a summary."""
import csv
from pathlib import Path

import numpy as np
import torch

import prnu
import wavelet
from render_sensor import info, names, dev

HERE = Path(__file__).resolve().parent
H6 = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
acc = {h: {"col": 0, "row": 0, "n": 0} for h in H6}
per_photo = {}
for i, n in enumerate(names):
    h, r = info(i)
    folder, f = n.split("/")
    path = prnu.DATA / ("anonymous" if folder == "anonymous" else f"credited/{folder}") / f
    rgb = prnu.load_rgb(path)
    m = torch.from_numpy(prnu.caption_mask(rgb)).to(dev).float()
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(dev)
    res = wavelet.residual(x, 3.0)[0] * m
    if r:
        res = torch.flip(res, (-2, -1)); m = torch.flip(m, (-2, -1))
    col = (res.sum(1) / m.sum(0).clamp(min=1)).cpu().numpy()  # 3 x W
    row = (res.sum(2) / m.sum(1).clamp(min=1)).cpu().numpy()  # 3 x H
    per_photo[n] = (col, row)
    acc[h]["col"] = acc[h]["col"] + col
    acc[h]["row"] = acc[h]["row"] + row
    acc[h]["n"] += 1
np.savez_compressed(HERE / "cache" / "line_patterns.npz",
                    **{f"{h}_col": acc[h]["col"] / acc[h]["n"] for h in H6},
                    **{f"{h}_row": acc[h]["row"] / acc[h]["n"] for h in H6},
                    **{f"p_{k.replace('/', '_')}_col": v[0] for k, v in per_photo.items()},
                    **{f"p_{k.replace('/', '_')}_row": v[1] for k, v in per_photo.items()})
for h in H6:
    c = acc[h]["col"] / acc[h]["n"]
    r = acc[h]["row"] / acc[h]["n"]
    print(f"{h:12s} col sd {c.std(1).round(3)}  row sd {r.std(1).round(3)}  col max {np.abs(c).max():.3f} row max {np.abs(r).max():.3f}")
