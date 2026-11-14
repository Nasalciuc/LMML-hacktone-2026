"""Pairwise residual correlation between all 253 photos (GPU). Saves cache/gram.npy.

G[i, j] = NCC(W_i * I_i, W_j * I_j) over pixels usable in both; z = G / sd, sd from the median absolute
off-diagonal value of the matrix (most pairs come from different cameras)."""
from pathlib import Path

import numpy as np
import torch

import feats as FE

HERE = Path(__file__).resolve().parent
d = FE.load_all()
names = d["names"]
N = len(names)
dev = FE.dev
# signal per photo: residual weighted by intensity (matches the multiplicative PRNU model)
X = []
for i in range(N):
    w = d["W"][i].float().to(dev)
    li = d["I"][i].float().to(dev)
    li = li / (li.pow(2).mean().sqrt() + 1e-6)
    s = w * li
    s = s - s.mean()
    X.append((s / (s.norm() + 1e-9)).half())
X = torch.stack(X).flatten(1)
G = torch.zeros(N, N, device=dev)
B = 16
for a in range(0, N, B):
    G[a:a + B] = (X[a:a + B].float() @ X.float().T) if False else torch.cat(
        [(X[a:a + B].float() @ X[c:c + 64].float().T) for c in range(0, N, 64)], 1)
G = G.cpu().numpy()
off = G[~np.eye(N, dtype=bool)]
sd = 1.4826 * np.median(np.abs(off - np.median(off)))
Z = G / sd
np.fill_diagonal(Z, 0)
np.save(HERE / "cache" / "gram_z.npy", Z)
with open(HERE / "cache" / "gram_names.txt", "w") as fh:
    fh.write("\n".join(names))
print("sd", sd)
lab = [n.split("/")[0] for n in names]
hs = sorted({l for l in lab if l != "anonymous"})
for h in hs:
    idx = [i for i in range(N) if lab[i] == h]
    row = []
    for g in hs:
        jdx = [j for j in range(N) if lab[j] == g]
        sub = Z[np.ix_(idx, jdx)]
        row.append(sub[sub != 0].mean())
    print(f"{h:12s} " + " ".join(f"{v:6.2f}" for v in row))
