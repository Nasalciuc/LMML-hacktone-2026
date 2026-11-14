"""PRNU attribution with z-scores and iterative refinement.

z-score of a photo against a fingerprint = correlation at zero shift divided by the spread of the correlation
over many non-zero shifts (null distribution). Fingerprints are re-estimated with confidently attributed
anonymous photos (EM style), always leaving the scored photo out of its own fingerprint.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import feats as FE

HERE = Path(__file__).resolve().parent
dev = FE.dev
d = FE.load_all()
names = d["names"]
handles = sorted({n.split("/")[0] for n in names if not n.startswith("anonymous")})
N = len(names)
lab = np.array([handles.index(n.split("/")[0]) if not n.startswith("anonymous") else -1 for n in names])
W = d["W"]
I = d["I"]


def zm(a):
    a = a - a.mean(-1, keepdim=True)
    return a - a.mean(-2, keepdim=True)


def sums(assign, weights=None):
    """per-person numerator / denominator of the MLE fingerprint for a given assignment (-1 = unused)."""
    num = torch.zeros(6, *W.shape[1:], device=dev)
    den = torch.zeros(6, *W.shape[1:], device=dev)
    for i in range(N):
        c = assign[i]
        if c < 0:
            continue
        w = 1.0 if weights is None else float(weights[i])
        wi = W[i].to(dev).float()
        li = I[i].to(dev).float()
        num[c] += w * wi * li
        den[c] += w * li * li
    return num, den


SHIFTS = [(97, 131), (211, 353), (-157, 419), (331, -277), (-283, -199), (149, 557), (-401, 89), (53, -613),
          (389, 241), (-71, -463), (263, 29), (-337, 509)]


def score_one(i, num, den, assign, weights=None, demean=True):
    wi = W[i].to(dev).float()
    li = I[i].to(dev).float()
    ks = []
    for c in range(6):
        nu, de = num[c], den[c]
        if assign[i] == c:
            w = 1.0 if weights is None else float(weights[i])
            nu = nu - w * wi * li
            de = de - w * li * li
        ks.append(zm(nu / de.clamp(min=1)))
    K = torch.stack(ks)
    if demean:
        K = K - K.mean(0, keepdim=True)
    a = wi - wi.mean()
    b = K * li[None]
    dims = tuple(range(1, b.dim()))
    b = b - b.mean(dims, keepdim=True)
    na = a.norm()
    raw = (b * a).sum(dims) / (b.flatten(1).norm(dim=1) * na)
    nulls = []
    for dy, dx in SHIFTS:
        bs = torch.roll(b, (dy, dx), (-2, -1))
        nulls.append((bs * a).sum(dims) / (bs.flatten(1).norm(dim=1) * na))
    nulls = torch.stack(nulls)
    sd = nulls.std(0).mean()
    return (raw / sd).cpu().numpy(), raw.cpu().numpy()


def run(n_iter=4, z_add=3.0, verbose=True):
    assign = lab.copy()
    for it in range(n_iter + 1):
        num, den = sums(assign)
        Z = np.zeros((N, 6))
        for i in range(N):
            Z[i], _ = score_one(i, num, den, assign)
        # LOO accuracy on credited
        cr = lab >= 0
        acc = (Z[cr].argmax(1) == lab[cr]).mean()
        an = ~cr
        best = Z[an].argmax(1)
        srt = np.sort(Z[an], 1)
        margin = srt[:, -1] - srt[:, -2]
        if verbose:
            print(f"iter {it}: credited LOO acc {acc:.3f}; anon counts {np.bincount(best, minlength=6)}; "
                  f"anon with margin>{z_add}: {(margin > z_add).sum()}", flush=True)
        new = lab.copy()
        idx = np.where(an)[0]
        for j, i in enumerate(idx):
            if margin[j] > z_add:
                new[i] = best[j]
        assign = new
    return Z, assign


if __name__ == "__main__":
    z_add = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    Z, assign = run(n_iter, z_add)
    np.save(HERE / "cache" / "Z.npy", Z)
    an = np.where(lab < 0)[0]
    with open(HERE / "attribution.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["file", "photographer", "z_best", "z_margin"] + [f"z_{h}" for h in handles])
        for i in an:
            z = Z[i]
            o = np.argsort(z)[::-1]
            wr.writerow([names[i].split("/")[1], handles[o[0]], f"{z[o[0]]:.2f}", f"{z[o[0]]-z[o[1]]:.2f}"] + [f"{v:.2f}" for v in z])
    cr = np.where(lab >= 0)[0]
    conf = np.zeros((6, 6), int)
    for i in cr:
        conf[lab[i], Z[i].argmax()] += 1
    print("credited LOO confusion (rows true):\n", conf)
