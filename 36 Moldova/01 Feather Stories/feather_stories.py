# Feather Stories: ring of the unregistered White Stork
# python feather_stories.py [folder with observations.csv and ring_register.csv]
import os
import sys
from collections import Counter, defaultdict
from itertools import product

import numpy as np
import pandas as pd
from scipy.special import betaln, comb

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "task_moldova")
obs = pd.read_csv(os.path.join(DATA, "observations.csv"), dtype={"ring_read": str})
reg = pd.read_csv(os.path.join(DATA, "ring_register.csv"))

# misreads stay inside these look-alike groups
GROUPS = ["AKRX", "BGSZ", "CDOQ", "EFLP", "HMNW", "IJTY"]
GRP = {c: i for i, g in enumerate(GROUPS) for c in g}
HALF = {c: "H1" if GRP[c] < 3 else "H2" for c in GRP}

# the three letter swaps inside a group (a*b = g)
SWAPS = {
    "a": "AR KX BZ GS CD OQ EL FP HN MW IJ TY",
    "b": "AK RX BG SZ CO DQ EF LP HW MN JT IY",
    "g": "AX KR BS GZ CQ DO EP FL HM NW IT JY",
}
INV = {"e": {c: c for c in GRP}}
for k, s in SWAPS.items():
    INV[k] = {}
    for p in s.split():
        INV[k][p[0]], INV[k][p[1]] = p[1], p[0]
MUL = {(x, y): next(z for z in "eabg" if all(INV[z][c] == INV[x][INV[y][c]] for c in GRP))
       for x in "eabg" for y in "eabg"}


def swap_of(true, read):
    if read == "?":
        return None
    return next((k for k in "eabg" if INV[k][true] == read), "X")


pattern = lambda s: "".join("?" if c == "?" else str(GRP[c]) for c in s)
fits = lambda p, q: all(a == "?" or a == b for a, b in zip(p, q))

reads = obs[obs.ring_read.notna()].copy()
reads["pat"] = reads.ring_read.map(pattern)
reg["pat"] = reg.ring.map(pattern)
by_sp = {sp: g for sp, g in reg.groupby("species")}
reads["cand"] = [[r for r, q in zip(by_sp[sp].ring, by_sp[sp].pat) if fits(p, q)] if sp in by_sp else []
                 for sp, p in zip(reads.species, reads.pat)]

known = reads[reads.cand.map(len) == 1].copy()
known["ring"] = known.cand.str[0]
known["sw"] = [[swap_of(t, x) for t, x in zip(r, rd)] for r, rd in zip(known.ring, known.ring_read)]
n_let = sum(c is not None for s in known.sw for c in s)
n_cross = sum(c == "X" for s in known.sw for c in s)
print(f"{len(reads)} reads, {len(known)} matched to one registered bird, {n_cross}/{n_let} letters cross a group")

stork = reads[reads.cand.map(len) == 0].sort_values(["date", "time"])
print(f"unmatched reads: {len(stork)}, all {stork.species.unique()} at {stork.locality_id.unique()}")

# swap counts per locality and per half of the alphabet
prof = defaultdict(Counter)
for loc, ring, sw in zip(known.locality_id, known.ring, known.sw):
    for t, c in zip(ring, sw):
        if c is not None:
            prof[(loc, HALF[t])][c] += 1

clear_a = clear_n = 0
for cnt in prof.values():
    n = sum(cnt.values())
    top = max("eabg", key=lambda k: cnt[k])
    if n >= 15 and top == "e" and cnt["e"] / n >= 0.55 and max("abg", key=lambda k: cnt[k]) != "a":
        clear_a += cnt["a"]
        clear_n += n
print(f"clear sites: double misreads {clear_a}/{clear_n}")

# the stork reads
pos = [Counter(r[k] for r in stork.ring_read if r[k] != "?") for k in range(6)]
major = "".join(c.most_common(1)[0][0] for c in pos)
for k, c in enumerate(pos):
    print(k + 1, dict(c.most_common()))
print("majority read:", major)

rel = {"H1": Counter(), "H2": Counter()}
for r in stork.ring_read:
    for k, x in enumerate(r):
        if x != "?":
            rel[HALF[major[k]]][swap_of(major[k], x)] += 1

# is the Lower Prut a clear site? too many double misreads says no
k_a = rel["H1"]["a"] + rel["H2"]["a"]
n_s = sum(rel["H1"].values()) + sum(rel["H2"].values())
rate = (clear_a + 0.5) / (clear_n + 1)
a0, b0 = rate * 500, (1 - rate) * 500
p = sum(comb(n_s, k) * np.exp(betaln(a0 + k, b0 + n_s - k) - betaln(a0, b0)) for k in range(k_a, n_s + 1))
print(f"double misreads in stork reads {k_a}/{n_s}, P(that many | clear site) = {p:.1e}")

# which swap? weigh each hypothesis by how well it fits the registered sites
sites = []
for loc in {l for l, _ in prof}:
    c1, c2 = prof[(loc, "H1")], prof[(loc, "H2")]
    n1, n2 = sum(c1.values()), sum(c2.values())
    if n1 >= 8 and n2 >= 8:
        sites.append(({k: (c1[k] + 0.1) / (n1 + 0.4) for k in "eabg"},
                      {k: (c2[k] + 0.1) / (n2 + 0.4) for k in "eabg"}))

ring_for = lambda x1, x2: "".join(INV[x1 if HALF[c] == "H1" else x2][c] for c in major)
score = {}
for x1, x2 in product("eabg", repeat=2):
    s1 = Counter({MUL[(r, x1)]: v for r, v in rel["H1"].items()})
    s2 = Counter({MUL[(r, x2)]: v for r, v in rel["H2"].items()})
    ll = np.array([sum(s1[k] * np.log(w1[k]) for k in "eabg") + sum(s2[k] * np.log(w2[k]) for k in "eabg")
                   for w1, w2 in sites])
    score[(x1, x2)] = ll.max() + np.log(np.mean(np.exp(ll - ll.max())))
top = max(score.values())
z = sum(np.exp(v - top) for v in score.values())
post = {k: np.exp(v - top) / z for k, v in score.items()}

both = Counter()
for w1, w2 in sites:
    d1, d2 = max(w1, key=w1.get), max(w2, key=w2.get)
    if d1 == d2 != "e" and w1[d1] >= 0.6 and w2[d2] >= 0.6:
        both[d1] += 1
print(f"{len(sites)} sites, both halves swapped the same way: {dict(both)}")
for (x1, x2), pr in sorted(post.items(), key=lambda kv: -kv[1])[:5]:
    print(f"  {x1}|{x2}  {ring_for(x1, x2)}  {pr:.3f}")

ring = ring_for(*max(post, key=post.get))
assert ring not in set(reg.ring) and pattern(ring) == pattern(major)
print("flag: SIGMOID_" + ring)
