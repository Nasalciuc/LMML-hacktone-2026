"""PRNU gain per photo: beta = <W, K I> / <K I, K I> with the photographer's leave-one-out sensor fingerprint.
A sender could modulate the strength of the pattern; this lists it in posting order."""
import csv
from pathlib import Path

import numpy as np
import torch

from render_sensor import info, d, names, dev, rot, FE

HERE = Path(__file__).resolve().parent
H6 = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
S = {}
for h in H6:
    num = 0; den = 0
    for i in range(len(names)):
        if info(i)[0] != h:
            continue
        w = d["W"][i].float().to(dev); li = d["I"][i].float().to(dev)
        if info(i)[1]:
            w, li = rot(w), rot(li)
        num = num + w * li; den = den + li * li
    S[h] = (num, den)
out = []
for i in range(len(names)):
    h, r = info(i)
    num, den = S[h]
    w = d["W"][i].float().to(dev); li = d["I"][i].float().to(dev)
    if r:
        w, li = rot(w), rot(li)
    k = FE.zm((num - w * li) / (den - li * li).clamp(min=1))
    s = k * li
    beta = float((w * s).sum() / (s * s).sum())
    out.append((names[i], h, r, beta))
with open(HERE / "out" / "prnu_gain.csv", "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["photo", "photographer", "rotated180", "gain"])
    for o in out:
        wr.writerow([o[0], o[1], o[2], f"{o[3]:.4f}"])
for h in H6:
    g = np.array([o[3] for o in out if o[1] == h])
    print(h, len(g), np.round(np.percentile(g, [0, 10, 25, 50, 75, 90, 100]), 3))
