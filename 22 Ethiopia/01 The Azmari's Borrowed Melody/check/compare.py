"""Runs arhiva/main.py and ../main.py on every cached recording and compares them.

    python compare.py <cache folder>        (after build_cases.py)

Both files' own detect() is called. Only compute_features is answered from the
cache; the script first checks that this function is the same text in both
files. A recording passes when both edges are within 1.5 s of the truth.
"""
import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import archive, fixed, passed, load_cases

assert inspect.getsource(archive.compute_features) == inspect.getsource(fixed.compute_features), \
    "compute_features differs between the two files: rebuild a cache per file"

cases = load_cases(sys.argv[1])


def call(mod, c):
    mod.compute_features = lambda x, sr, hop_s=0.05, win_s=0.093: (c["tt"], c["F"])
    return mod.detect(np.zeros(int(round(c["dur"] * c["sr"]))), c["sr"])


rows = []
for c in cases:
    ra, rf = call(archive, c), call(fixed, c)
    if c["truth"] is None:
        print(f"control with nothing replaced: archive answers {ra[0]:.2f}-{ra[1]:.2f}, fixed answers {rf[0]:.2f}-{rf[1]:.2f}")
        continue
    rows.append((c, ra, rf, passed(c["truth"], *ra), passed(c["truth"], *rf)))


def table(key, title):
    print(f"\n{title:36s} cases  archive  fixed")
    keys = []
    for r in rows:
        if key(r[0]) not in keys:
            keys.append(key(r[0]))
    for k in keys:
        sel = [r for r in rows if key(r[0]) == k]
        print(f"  {k:34s} {len(sel):5d}  {sum(r[3] for r in sel):7d}  {sum(r[4] for r in sel):5d}")
    print(f"  {'ALL':34s} {len(rows):5d}  {sum(r[3] for r in rows):7d}  {sum(r[4] for r in rows):5d}")


def length(c):
    return c["truth"][1] - c["truth"][0]


def length_bin(c):
    L = length(c)
    return "under 2.5 s" if L < 2.5 else "2.5-5 s" if L < 5 else "5-10 s" if L < 10 else "10-20 s"


table(lambda c: c["set"] + " / " + c["group"][:22], "by test group")
table(lambda c: c["sc"], "by kind of replacement")
table(length_bin, "by length of the replaced part")

print("\nfixed by the change:", sum((not r[3]) and r[4] for r in rows),
      "| broken by the change:", sum(r[3] and not r[4] for r in rows))
for c, ra, rf, pa, pf in rows:
    if pa and not pf:
        print(f"  BROKEN  {c['set']} {c['group'][:22]:22s} {c['sc'][:22]:22s} truth {c['truth'][0]:6.2f}-{c['truth'][1]:6.2f}"
              f" | archive {ra[0]:6.2f}-{ra[1]:6.2f} | fixed {rf[0]:6.2f}-{rf[1]:6.2f}")

print("\nstill failing with the fixed file:")
for c, ra, rf, pa, pf in rows:
    if not pf:
        a, b = c["truth"]
        s, e = rf
        if max(0.0, min(b, e) - max(a, s)) == 0:
            kind = "elsewhere in the file"
        else:
            kind = " and ".join(n for n, bad in (("start wrong", abs(s - a) > 1.5), ("end wrong", abs(e - b) > 1.5)) if bad)
        print(f"  {c['set']} {c['group'][:22]:22s} {c['sc'][:22]:22s} len {b - a:4.1f} truth {a:6.2f}-{b:6.2f}"
              f" | fixed {s:6.2f}-{e:6.2f} (start {s - a:+6.2f}, end {e - b:+6.2f}) {kind}")

err = np.array([max(abs(rf[0] - c["truth"][0]), abs(rf[1] - c["truth"][1])) for c, ra, rf, pa, pf in rows if pf])
print(f"\nedge error among passed recordings, fixed file: median {np.median(err):.2f} s, "
      f"95% {np.percentile(err, 95):.2f} s, worst {err.max():.2f} s, over 1.0 s in {np.mean(err > 1.0) * 100:.1f}%")
