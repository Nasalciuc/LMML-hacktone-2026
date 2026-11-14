"""Read the word from counts.csv.

Every set count is either high (the crate glyph was carved in many rubbings of that set) or
close to zero. High = 1, low = 0; the five sets of a crate, set_1 first, form a 5-bit number,
and that number is the letter's position in the alphabet (A = 1).

    python decode.py --counts counts.csv
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--counts", default="counts.csv")
ap.add_argument("--by_thr", default="counts_by_threshold.csv")
a = ap.parse_args()

c = pd.read_csv(a.counts)
P = c.pivot(index="crate", columns="set", values="count")
vals = np.sort(P.values.ravel())
gap = np.argmax(np.diff(vals))
cut = (vals[gap] + vals[gap + 1]) / 2
print(f"counts split into low <= {vals[gap]} and high >= {vals[gap + 1]}; cut at {cut}")
word = ""
for crate, row in P.iterrows():
    bits = "".join("1" if v > cut else "0" for v in row.values)
    n = int(bits, 2)
    word += chr(ord("A") + n - 1)
    print(crate, row.values.tolist(), bits, n, chr(ord("A") + n - 1))
print("WORD:", word)

# stability over detector thresholds
t = pd.read_csv(a.by_thr)
for thr, g in t.groupby("thr"):
    Q = g.pivot(index="crate", columns="set", values="count")
    w = "".join(chr(ord("A") + int("".join("1" if v > cut else "0" for v in r.values), 2) - 1) for _, r in Q.iterrows())
    print(f"threshold {thr}: {w}")

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(P))
for k, s in enumerate(P.columns):
    ax.bar(x + (k - 2) * 0.16, P[s].values, width=0.15, label=s)
ax.axhline(cut, color="k", ls="--", lw=1, label=f"cut {cut:.1f}")
ax.set_xticks(x)
ax.set_xticklabels([f"{cr}\n{''.join('1' if v > cut else '0' for v in P.loc[cr].values)} = {ch}"
                    for cr, ch in zip(P.index, word)])
ax.set_ylabel("detected crate glyphs in the set (30 rubbings)")
ax.set_title(f"Crate glyph counts per set -> 5-bit letters -> {word}")
ax.legend(ncol=6, fontsize=8)
plt.tight_layout()
plt.savefig("counts_bits.png", dpi=110)
print("saved counts_bits.png")
