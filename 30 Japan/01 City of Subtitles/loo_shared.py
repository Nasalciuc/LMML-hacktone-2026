"""BIASED, kept as a record: Pooled (powered) test for a second pattern shared by a subset of photos. Cleaned residuals
(cache/clean_res.pt, own sensor removed). For each group and frame, each photo is correlated with the mean of
the OTHER photos of the group (leave-one-out, no self bias), z-scored against the same correlation at 16 large
circular shifts. Frames: posted (as published) and sensor (photo turned back if rotated180).
Groups: each photographer (all photos / anonymous only) and all 253 photos. A pattern carried by a subset shows
as a tail of high z in that group. Also prints toorisugari's anonymous z in posting order (bit pattern).
Output out/loo_shared.csv."""
import csv
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
D = torch.load(HERE / "cache" / "clean_res.pt"); names, who, rot = D["names"], D["who"], D["rot"]
R = D["R"].cuda(); N = len(names)
SH = [(97 * k, 131 * k) for k in range(1, 17)]
def frame(i, f):
    r = R[i].float()
    return torch.flip(r, (0, 1)) if (f == "sensor" and rot[i]) else r
def z(a, b):
    c0 = float((a * b).sum())
    cs = np.array([float((a * torch.roll(b, s, (0, 1))).sum()) for s in SH])
    return (c0 - cs.mean()) / (cs.std() + 1e-9)
groups = {"ALL": list(range(N))}
for h in sorted(set(who)):
    groups[h] = [i for i in range(N) if who[i] == h]
    groups[h + "_anon"] = [i for i in groups[h] if names[i].startswith("anonymous")]
rows = []
for g, ix in groups.items():
    for f in ("posted", "sensor"):
        S = sum(frame(i, f) for i in ix)
        zs = [z(frame(i, f), (S - frame(i, f)) / (len(ix) - 1)) for i in ix]
        for i, zz in zip(ix, zs): rows.append((g, f, names[i], who[i], rot[i], round(zz, 2)))
        zs = np.array(zs)
        print(f"{g:16s} {f:6s} n={len(ix):3d}  mean z {zs.mean():5.2f}  sd {zs.std():4.2f}  max {zs.max():5.2f}  #z>3 {int((zs>3).sum()):2d}  #z<-3 {int((zs<-3).sum()):2d}", flush=True)
with open(HERE / "out" / "loo_shared.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["group", "frame", "photo", "photographer", "rotated180", "z"]); w.writerows(rows)
for f in ("posted", "sensor"):
    zz = [r for r in rows if r[0] == "toorisugari_anon" and r[1] == f]
    zz.sort(key=lambda r: r[2])
    print(f"toorisugari anon {f}:", " ".join(f"{r[5]:.1f}" for r in zz))
