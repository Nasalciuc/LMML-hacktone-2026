"""Bias-free pooled tests on RAW residuals (no fingerprint subtraction, so no cleaning bias and no leftover):
T1 posted-frame pattern within a photographer: photo vs mean of the same photographer's photos with the
   OPPOSITE rotation flag, posted frame. Their sensor patterns are 180 degrees apart there (uncorrelated,
   corr 0.0005), so only something added in posted coordinates can score.
T2 pattern shared across photographers: photo vs mean of all OTHER photographers' photos, in posted frame and
   in sensor frame (everyone turned back). Different sensors are independent.
z against 16 circular shifts. The shift-based z is inflated (null sd about 1.5-2, read off the groups), so
only a tail clearly above z 4-5 in one group would mean something.
Output out/loo_shared3.csv; prints toorisugari anonymous T1 z in posting order."""
import csv, random
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
d = torch.load(HERE / "cache" / "feats.pt", mmap=True)
D = torch.load(HERE / "cache" / "clean_res.pt"); names, who, rot = D["names"], D["who"], D["rot"]
N = len(names); dev = "cuda"; flip = lambda t: torch.flip(t, (0, 1))
R = torch.empty(N, 960, 1280, dtype=torch.float16, device=dev)
for i in range(N):
    w = d["W"][i].float().mean(0).to(dev); m = d["M"][i].float().to(dev); r = w * m
    R[i] = (r / r[m > 0].std()).half()
fr = lambda i, f: flip(R[i].float()) if (f == "sensor" and rot[i]) else R[i].float()
SH = [(97 * k, 131 * k) for k in range(1, 17)]
def z(a, b):
    c0 = float((a * b).sum()); cs = np.array([float((a * torch.roll(b, s, (0, 1))).sum()) for s in SH])
    return (c0 - cs.mean()) / (cs.std() + 1e-9)
rows = []
def report(tag, zs):
    zs = np.array(zs); print(f"{tag:34s} n={len(zs):3d} mean {zs.mean():5.2f} sd {zs.std():4.2f} max {zs.max():5.2f} #z>3 {int((zs>3).sum()):2d} #z>4 {int((zs>4).sum())}", flush=True)
H = sorted(set(who))
for h in H:
    ix = [i for i in range(N) if who[i] == h]
    M = {fl: sum(fr(i, "posted") for i in ix if rot[i] == fl) / max(1, sum(1 for i in ix if rot[i] == fl)) for fl in (0, 1)}
    zs = []
    for i in ix:
        v = z(fr(i, "posted"), M[1 - rot[i]]); zs.append(v); rows.append(("T1", "posted", names[i], h, rot[i], round(v, 2)))
    report(f"T1 {h} vs opposite-flag mean", zs)
for f in ("posted", "sensor"):
    tot = sum(fr(i, f) for i in range(N)); per = {h: sum(fr(i, f) for i in range(N) if who[i] == h) for h in H}
    cnt = {h: sum(1 for i in range(N) if who[i] == h) for h in H}
    for h in H:
        ref = (tot - per[h]) / (N - cnt[h]); zs = []
        for i in range(N):
            if who[i] != h: continue
            v = z(fr(i, f), ref); zs.append(v); rows.append(("T2", f, names[i], h, rot[i], round(v, 2)))
        report(f"T2 {f} {h} vs other photographers", zs)
with open(HERE / "out" / "loo_shared3.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["test", "frame", "photo", "photographer", "rotated180", "z"]); w.writerows(rows)
t = sorted([r for r in rows if r[0] == "T1" and r[3] == "toorisugari" and r[2].startswith("anonymous")], key=lambda r: r[2])
print("toorisugari anon T1 z (posting order):", " ".join(f"{r[5]:.1f}" for r in t))
