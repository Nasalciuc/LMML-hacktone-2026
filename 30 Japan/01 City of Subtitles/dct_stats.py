"""Histogram-pair statistics of the recovered luma AC coefficients per photo (JSteg-style embedding equalises
the counts of 2k and 2k+1). Prints, per photo, h(2)/h(3), h(-2)/h(-3), h(4)/h(5) and the photographer."""
import csv
from pathlib import Path

import numpy as np

import prnu
from dct_coeffs import coeffs

HERE = Path(__file__).resolve().parent
att = {r["file"]: r["photographer"] for r in csv.DictReader(open(HERE / "attribution.csv"))}
out = []
for p in sorted((prnu.DATA / "anonymous").glob("*.jpg")):
    cq, _ = coeffs(p)
    v = np.round(cq).astype(int)
    ac = v.reshape(-1, 64)[:, 1:].ravel()
    h = lambda k: max(1, int((ac == k).sum()))
    r = (h(2) / h(3), h(-2) / h(-3), h(4) / h(5), h(1) / h(2))
    out.append((p.name, att[p.name], r))
    print(p.name, f"{att[p.name]:12s}", " ".join(f"{x:5.2f}" for x in r), flush=True)
