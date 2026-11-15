"""Step 3: read cache/pce_pairs.npz (or the file given as 2nd argument, e.g. pce_pairs_phat.npz). Per transform: top pairs, hits PCE > THR, connected clusters of hits.
A hit between two photos of the same photographer at zero shift in the orientation that aligns their sensors
(id if same rotation flag, rot180 if different) is leftover of their own fingerprint, flagged 'own'."""
import sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 60
NPZ = sys.argv[2] if len(sys.argv) > 2 else "pce_pairs.npz"
Z = np.load(HERE / "cache" / NPZ)
D = torch.load(HERE / "cache" / "clean_res.pt")
names, who, rot = D["names"], D["who"], D["rot"]
N = len(names)
short = lambda i: names[i].replace("anonymous/", "A:").replace("/", ":")
hits = []
for v in ["id", "rot180", "flipLR", "flipUD", "rot90", "rot270"]:
    if v + "_pce" not in Z: continue
    P = Z[v + "_pce"]; iu = np.triu_indices(N, 1); order = np.argsort(-P[iu])
    print(f"\n== {v}: top 12 ==")
    for t in order[:12]:
        i, j = iu[0][t], iu[1][t]
        dy, dx, pk = Z[v + "_dy"][i, j], Z[v + "_dx"][i, j], Z[v + "_peak"][i, j]
        own = who[i] == who[j] and dy == 0 and dx == 0 and ((v == "id" and rot[i] == rot[j]) or (v == "rot180" and rot[i] != rot[j]))
        print(f"  PCE {P[i,j]:8.1f} sign {'+' if pk>0 else '-'} shift ({dy:4d},{dx:5d})  {short(i):24s}[{who[i]},r{rot[i]}]  {short(j):24s}[{who[j]},r{rot[j]}]{'  own' if own else ''}")
        if P[i, j] > THR and not own: hits.append((v, i, j, float(P[i, j]), int(dy), int(dx)))
    for i, j in zip(*np.where(np.triu(P, 1) > THR)):
        dy, dx = Z[v + "_dy"][i, j], Z[v + "_dx"][i, j]
        own = who[i] == who[j] and dy == 0 and dx == 0 and ((v == "id" and rot[i] == rot[j]) or (v == "rot180" and rot[i] != rot[j]))
        if not own and (v, i, j) not in [(h[0], h[1], h[2]) for h in hits]:
            hits.append((v, int(i), int(j), float(P[i, j]), int(dy), int(dx)))
print(f"\nnon-own hits with PCE > {THR}: {len(hits)}")
for h in sorted(hits, key=lambda h: -h[3])[:60]:
    v, i, j, p, dy, dx = h; print(f"  {v:7s} {p:8.1f} ({dy},{dx}) {short(i)}[{who[i]}] - {short(j)}[{who[j]}]")
# clusters
par = list(range(N))
def f(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for v, i, j, *_ in hits: par[f(i)] = f(j)
cl = {}
for i in range(N): cl.setdefault(f(i), []).append(i)
big = [c for c in cl.values() if len(c) > 1]
print("clusters:", [(len(c), sorted({who[i] for i in c})) for c in sorted(big, key=len, reverse=True)][:10])
for c in sorted(big, key=len, reverse=True)[:3]:
    print("  members:", " ".join(short(i) for i in sorted(c, key=lambda i: names[i])))
