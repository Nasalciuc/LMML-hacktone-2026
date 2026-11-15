"""Why are real other-brand photos rejected less often than the package's null set?

    python -I soles_test3.py <model.pt> <hf_sneakers folder> <packaged oof csv> <out.npz>

Per-image scores for unknown-brand photos resized straight to 256, split by native size, and the
trade-off between known accuracy (the package's out-of-fold table) and rejection for other thresholds.
"""
import collections
import csv
import glob
import io
import os
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

MODEL, HF, OOF, OUT = sys.argv[1:5]
N_PER = 300
UNKNOWN = ["puma", "reebok", "vans", "asics"]
torch.set_num_threads(8)
rng = np.random.default_rng(1)
m = torch.jit.load(MODEL, map_location="cpu").eval()
sem_cal, tr_cal = m.sem_cal.numpy(), m.tr_cal.numpy()
THR_J, W = float(m.thr_joint), float(m.w_trace)


def pval(cal, s):
    return (np.searchsorted(cal, s, side="right") + 1.0) / (len(cal) + 2.0)


def scores(x):
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).contiguous()
    ss, st = [], []
    with torch.no_grad():
        for i in range(0, len(t), 64):
            _, _, c, d = m.analyse(t[i:i + 64])
            ss.append(c), st.append(d)
    return torch.cat(ss).numpy(), torch.cat(st).numpy()


def jpeg(a, q=95):
    b = io.BytesIO()
    Image.fromarray(a).save(b, "JPEG", quality=q, subsampling=2)
    b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"))


files = sorted(glob.glob(os.path.join(HF, "*.parquet")))
imgs, brand, side, wh = [], [], [], []
cnt = collections.Counter()
for f in files:
    t = pq.read_table(f, columns=["brand", "image"])
    br = [b.strip().lower() for b in t["brand"].to_pylist()]
    for i in rng.permutation(len(br)):
        b = br[i]
        if b not in UNKNOWN or cnt[b] >= N_PER:
            continue
        try:
            im = Image.open(io.BytesIO(t["image"][int(i)].as_py())).convert("RGB")
        except Exception:
            continue
        if min(im.size) < 150:
            continue
        imgs.append(np.asarray(im)), brand.append(b), side.append(min(im.size)), wh.append(im.size)
        cnt[b] += 1
brand, side = np.array(brand), np.array(side)
print("unknown-brand photos:", dict(cnt), "| native shorter side: median", int(np.median(side)),
      "| share >= 400 px:", round(float((side >= 400).mean()), 3), flush=True)

res = {}
for name, f in (("pil_bicubic", lambda a: np.asarray(Image.fromarray(a).resize((256, 256), Image.BICUBIC))),
                ("pil_bilinear", lambda a: np.asarray(Image.fromarray(a).resize((256, 256), Image.BILINEAR))),
                ("cv_area", lambda a: cv2.resize(a, (256, 256), interpolation=cv2.INTER_AREA))):
    x = np.stack([jpeg(f(a)) for a in imgs])
    ss, st = scores(x)
    ps, pt = pval(sem_cal, ss), pval(tr_cal, st)
    stat = np.log(ps) + W * np.log(pt)
    rej = stat < THR_J
    res[name] = (ps, pt, stat)
    print(f"\n== straight to 256 with {name}: rejected {rej.mean():.3f} of {len(rej)}", flush=True)
    print("   per brand:", {b: round(float(rej[brand == b].mean()), 3) for b in UNKNOWN})
    for lo, hi in ((150, 230), (230, 236), (236, 245), (245, 256), (256, 300), (300, 400), (400, 10000)):
        k = (side >= lo) & (side < hi)
        if k.sum():
            print(f"   native shorter side {lo:4d}-{hi:<5d} n={int(k.sum()):4d}  rejected {rej[k].mean():.3f}  "
                  f"median p_trace {np.median(pt[k]):.4f}  share p_trace<0.0154 {np.mean(pt[k] < 0.0154):.3f}")
    sq = np.array([w == h for w, h in wh])
    print(f"   square originals n={int(sq.sum())} rejected {rej[sq].mean():.3f} | non-square n={int((~sq).sum())} rejected {rej[~sq].mean():.3f}")

# trade-off with other thresholds: knowns from the package's out-of-fold table
rows = list(csv.DictReader(open(OOF)))
kps = np.array([float(r["p_semantic"]) for r in rows])
kpt = np.array([float(r["p_trace"]) for r in rows])
kok = np.array([int(r["oof_correct"]) for r in rows]).astype(bool)
ups, upt, _ = res["pil_bicubic"]
print("\n== other thresholds (knowns: package out-of-fold table, n=735; unknowns: these photos, pil_bicubic)")
print("   rule                 keep   known_acc  unknown_rejected")
for w in (1.0, 2.0, 3.0, 5.0):
    ks, us = np.log(kps) + w * np.log(kpt), np.log(ups) + w * np.log(upt)
    for keep in (0.98, 0.97, 0.95, 0.93, 0.90):
        thr = np.quantile(ks, 1 - keep)
        print(f"   sem + {w:.0f} x trace     {keep:.2f}   {np.mean((ks >= thr) & kok):.3f}      {np.mean(us < thr):.3f}")
for keep in (0.98, 0.97, 0.95, 0.93, 0.90):
    thr = np.quantile(kpt, 1 - keep)
    print(f"   trace only           {keep:.2f}   {np.mean((kpt >= thr) & kok):.3f}      {np.mean(upt < thr):.3f}")
np.savez(OUT, brand=brand, side=side, **{f"{k}_{n}": v for k, t3 in res.items() for n, v in zip(("ps", "pt", "stat"), t3)})
print("saved", OUT)
