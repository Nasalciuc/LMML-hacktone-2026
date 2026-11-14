"""Hunt for a seventh camera hidden inside the toorisugari group.

Seed: anonymous photos that match the anonymous toorisugari group but not the credited toorisugari photos.
Build a fingerprint K_x from the seed, score every photo (z, leave-one-out), grow the set, repeat.
"""
import sys
from pathlib import Path

import numpy as np
import torch

import attribute2 as A

HERE = Path(__file__).resolve().parent
seed = sys.argv[1].split(",") if len(sys.argv) > 1 else ["w10_018", "w08_023", "w10_010", "w08_006", "w08_010", "w10_022"]
members = [A.names.index(f"anonymous/{s}.jpg") for s in seed]


def zscore(i, num, den, inside):
    wi = A.W[i].float().to(A.dev)
    li = A.I[i].float().to(A.dev)
    if inside:
        num = num - wi * li
        den = den - li * li
    k = A.zm(num / den.clamp(min=1))
    a = wi - wi.mean()
    b = k * li
    b = b - b.mean()
    raw = float((a * b).sum() / (a.norm() * b.norm()))
    nulls = [float((a * torch.roll(b, s, (-2, -1))).sum() / (a.norm() * b.norm())) for s in A.SHIFTS]
    return raw / np.std(nulls)


for it in range(4):
    num = sum(A.W[i].float().to(A.dev) * A.I[i].float().to(A.dev) for i in members)
    den = sum(A.I[i].float().to(A.dev) ** 2 for i in members)
    Z = np.array([zscore(i, num, den, i in members) for i in range(A.N)])
    new = [i for i in range(A.N) if Z[i] > 5]
    print(f"iter {it}: {len(members)} members; photos with z>5: {len(new)}")
    order = np.argsort(-Z)
    print("  " + " ".join(f"{A.names[i].replace('anonymous/','')}:{Z[i]:.1f}" for i in order[:45]))
    if set(new) == set(members):
        break
    members = new
np.save(HERE / "cache" / "Z_seventh.npy", Z)
cr = A.lab >= 0
print("credited photos with z>4:", [A.names[i] for i in np.where(cr & (Z > 4))[0]])
