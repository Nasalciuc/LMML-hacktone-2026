"""Is the group attributed to one handle one camera? Score each member against a fingerprint built only from the
credited photos and against one built only from the other anonymous members (leave-one-out)."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

import attribute2 as A

HERE = Path(__file__).resolve().parent
handle = sys.argv[1] if len(sys.argv) > 1 else "toorisugari"
c = A.handles.index(handle)
rows = list(csv.DictReader(open(HERE / "attribution.csv")))
grp = [A.names.index("anonymous/" + r["file"]) for r in rows if r["photographer"] == handle]
cred = [i for i in range(A.N) if A.lab[i] == c]


def fp(idx):
    num = sum(A.W[i].float().to(A.dev) * A.I[i].float().to(A.dev) for i in idx)
    den = sum(A.I[i].float().to(A.dev) ** 2 for i in idx)
    return num, den


def z(i, num, den, self_in):
    wi = A.W[i].float().to(A.dev)
    li = A.I[i].float().to(A.dev)
    if self_in:
        num = num - wi * li
        den = den - li * li
    k = A.zm(num / den.clamp(min=1))
    a = wi - wi.mean()
    b = k * li
    b = b - b.mean()
    raw = float((a * b).sum() / (a.norm() * b.norm()))
    nulls = []
    for dy, dx in A.SHIFTS:
        bs = torch.roll(b, (dy, dx), (-2, -1))
        nulls.append(float((a * bs).sum() / (a.norm() * bs.norm())))
    return raw / np.std(nulls)


nc, dc = fp(cred)
na, da = fp(grp)
print(f"{'photo':12s} z_vs_credited z_vs_anon_group")
for i in grp + cred:
    print(f"{A.names[i]:28s} {z(i, nc, dc, i in cred):7.1f} {z(i, na, da, i in grp):7.1f}")
