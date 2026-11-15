"""What each candidate change does: the archive's search re-run with options.

    python ablation.py <cache folder>       (after build_cases.py)

run() repeats detect() from arhiva/main.py on the cached features, with a
choice of features and of the longest block allowed. The first row has to
match the archive column of compare.py and the row marked "= ../main.py" the
fixed column.
"""
import os
import sys

import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import archive, passed, load_cases

MELODY = ("f0log", "mfcc", "centroid_f0", "flux")              # follow the notes played
CHAIN = ("lf_", "subf0", "infra", "dc", "hum", "nz_", "floor",  # follow how it was recorded
         "hf_rel", "flat2", "skew", "crest")


def run(c, exclude=(), only=None, max_s=40.0):
    tt, F, dur = c["tt"], c["F"], c["dur"]
    N = len(tt)
    hop = tt[1] - tt[0]
    V = []
    for k in sorted(F):
        if only is not None and not any(k.startswith(p) for p in only):
            continue
        if any(k.startswith(p) for p in exclude):
            continue
        g = archive.gauss_rank(median_filter(F[k], size=5, mode="nearest"))
        if np.std(g) > 1e-3:
            V.append(g)
    V = np.array(V).T
    V = V - V.mean(0)
    C = np.atleast_2d(np.cov(V.T))
    C = 0.9 * C + 0.1 * np.diag(np.diag(C))
    w, U = np.linalg.eigh(C)
    Xw = V @ (U @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-6))) @ U.T)
    min_len = max(2, int(round(1.5 / hop)))
    max_len = max(min_len + 1, min(int(0.5 * N), int(round(max_s / hop))))
    _, a, b = archive.best_block(Xw, min_len, max_len)
    return float(np.clip(tt[a] - hop / 2, 0, dur)), float(np.clip(tt[b - 1] + hop / 2, 0, dur))


VARIANTS = {
    "archive as is (max 40 s)": {},
    "max 20 s": dict(max_s=20),
    "max 15 s": dict(max_s=15),
    "max 12 s": dict(max_s=12),
    "no melody features": dict(exclude=MELODY),
    "no melody + max 20 s = ../main.py": dict(exclude=MELODY, max_s=20),
    "no melody + max 15 s": dict(exclude=MELODY, max_s=15),
    "no melody + max 12 s": dict(exclude=MELODY, max_s=12),
    "chain features only": dict(only=CHAIN),
}

cases = [c for c in load_cases(sys.argv[1]) if c["truth"] is not None]
cols = ["first 140"]
for c in cases:
    if c["set"] != "first 140" and c["group"] not in cols:
        cols.append(c["group"])


def col(c):
    return "first 140" if c["set"] == "first 140" else c["group"]


print("variant".ljust(36) + " ".join(f"{k[:17]:>17s}" for k in cols) + "   ALL")
print("recordings".ljust(36) + " ".join(f"{sum(col(c) == k for c in cases):>17d}" for k in cols) + f"   {len(cases)}")
for name, opt in VARIANTS.items():
    res = dict.fromkeys(cols, 0)
    for c in cases:
        res[col(c)] += passed(c["truth"], *run(c, **opt))
    print(name.ljust(36) + " ".join(f"{res[k]:>17d}" for k in cols) + f"   {sum(res.values())}", flush=True)
