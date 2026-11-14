"""Every photo against its photographer's sensor fingerprint under the four orientations of a 1280x960 frame
(as shot, turned 180, mirrored left-right, mirrored top-bottom). z-scores, leave-one-out.
Writes out/orient4.csv."""
import csv
from pathlib import Path

import numpy as np
import torch

from render_sensor import info, d, names, dev, rot, FE

HERE = Path(__file__).resolve().parent
SHIFTS = [(97, 131), (211, 353), (-157, 419), (331, -277), (-283, -199), (149, 557), (-401, 89), (53, -613)]
OPS = {"id": lambda t: t, "rot180": lambda t: torch.flip(t, (-2, -1)), "flipLR": lambda t: torch.flip(t, (-1,)),
       "flipUD": lambda t: torch.flip(t, (-2,))}
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


def z(a, b):
    a = a - a.mean(); b = b - b.mean()
    raw = float((a * b).sum() / (a.norm() * b.norm()))
    nl = [float((a * torch.roll(b, s, (-2, -1))).sum() / (a.norm() * b.norm())) for s in SHIFTS]
    return raw / np.std(nl)


rows = []
for i in range(len(names)):
    h, r = info(i)
    num, den = S[h]
    w = d["W"][i].float().to(dev); li = d["I"][i].float().to(dev)
    ws, ls = (rot(w), rot(li)) if r else (w, li)
    k = FE.zm((num - ws * ls) / (den - ls * ls).clamp(min=1))
    zs = {nm: z(w, op(k) * li) for nm, op in OPS.items()}
    best = max(zs, key=zs.get)
    rows.append([names[i], h, best] + [f"{zs[o]:.2f}" for o in OPS])
    if best in ("flipLR", "flipUD"):
        print("MIRRORED?", rows[-1], flush=True)
with open(HERE / "out" / "orient4.csv", "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["photo", "photographer", "best"] + list(OPS)); wr.writerows(rows)
from collections import Counter
print(Counter((r[1], r[2]) for r in rows))
