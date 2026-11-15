"""LEFTOVER-DOMINATED, kept as a record: Unbiased pooled test for a second pattern shared by a subset of photos (fixes the self-bias of
loo_shared.py / hidden_pattern_z.csv). Each photographer's photos are split into halves H1/H2 (alternating in
name order). Photos of H1 are cleaned with the sensor fingerprint estimated from H2 only (and vice versa), so a
cleaned photo carries no trace of the other photos in its own half. Within a half, each photo is correlated with
the leave-one-out mean of the others (posted or sensor frame) and z-scored against 16 circular shifts.
Groups: per photographer, per photographer anonymous-only, and all photographers pooled (half-wise).
Calibration: the z of a photo against an unrelated photographer's half should be ~N(0,1).
Output out/loo_shared2.csv."""
import csv
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
d = torch.load(HERE / "cache" / "feats.pt", mmap=True)
D = torch.load(HERE / "cache" / "clean_res.pt"); names, who, rot = D["names"], D["who"], D["rot"]
N = len(names); dev = "cuda"
flip = lambda t: torch.flip(t, (0, 1))
half = {}
for h in set(who):
    ix = sorted([i for i in range(N) if who[i] == h], key=lambda i: names[i])
    for k, i in enumerate(ix): half[i] = k % 2
def raw(i):
    return d["W"][i].float().mean(0).to(dev), d["I"][i].float().mean(0).to(dev), d["M"][i].float().to(dev)
K = {}
for h in set(who):
    for hf in (0, 1):
        num = 0; den = 0
        for i in range(N):
            if who[i] != h or half[i] != hf: continue
            w, I, m = raw(i)
            if rot[i]: w, I, m = flip(w), flip(I), flip(m)
            num = num + w * I * m; den = den + I * I * m
        k = num / den.clamp(min=1); k = k - k.mean(1, keepdim=True); K[(h, hf)] = k - k.mean(0, keepdim=True)
C = torch.empty(N, 960, 1280, dtype=torch.float16, device=dev)
for i in range(N):
    w, I, m = raw(i); k = K[(who[i], 1 - half[i])]; k = flip(k) if rot[i] else k
    s = k * I * m; a = (w * s).sum() / (s * s).sum().clamp(min=1e-9); r = (w - a * s) * m
    C[i] = (r / r[m > 0].std()).half()
SH = [(97 * k, 131 * k) for k in range(1, 17)]
def fr(i, f):
    r = C[i].float(); return flip(r) if (f == "sensor" and rot[i]) else r
def z(a, b):
    c0 = float((a * b).sum()); cs = np.array([float((a * torch.roll(b, s, (0, 1))).sum()) for s in SH])
    return (c0 - cs.mean()) / (cs.std() + 1e-9)
groups = {}
for hf in (0, 1):
    groups[f"ALL_h{hf}"] = [i for i in range(N) if half[i] == hf]
    for h in sorted(set(who)):
        groups[f"{h}_h{hf}"] = [i for i in range(N) if who[i] == h and half[i] == hf]
        groups[f"{h}_anon_h{hf}"] = [i for i in groups[f"{h}_h{hf}"] if names[i].startswith("anonymous")]
rows = []
for g, ix in groups.items():
    for f in ("posted", "sensor"):
        S = sum(fr(i, f) for i in ix)
        zs = np.array([z(fr(i, f), (S - fr(i, f)) / (len(ix) - 1)) for i in ix])
        rows += [(g, f, names[i], who[i], rot[i], round(float(v), 2)) for i, v in zip(ix, zs)]
        print(f"{g:20s} {f:6s} n={len(ix):3d} mean z {zs.mean():5.2f} sd {zs.std():4.2f} max {zs.max():5.2f} #z>3 {int((zs > 3).sum())}", flush=True)
# calibration: toorisugari photos vs yakin half mean (unrelated) in posted frame
ix_t = [i for i in range(N) if who[i] == "toorisugari"]; Sy = sum(fr(i, "posted") for i in range(N) if who[i] == "yakin") / 38
cal = np.array([z(fr(i, "posted"), Sy) for i in ix_t]); print(f"calibration toori vs yakin mean: mean z {cal.mean():.2f} sd {cal.std():.2f}")
with open(HERE / "out" / "loo_shared2.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["group", "frame", "photo", "photographer", "rotated180", "z"]); w.writerows(rows)
