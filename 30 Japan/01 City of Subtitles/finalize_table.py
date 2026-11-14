"""Final attribution table.

The unconstrained attribution (attribute_final.py) gives exactly 9 toorisugari photos every week and 4 for
nearly every regular every week (29 = 9 + 5 x 4). Two weeks are off by one (a weak amayadori photo taken by
the next-best regular). Here each week is solved as an assignment problem with those quotas, maximising the
summed z-scores; changes vs. the free assignment are listed. Writes attribution_table.csv.
"""
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
H6 = ["amayadori", "jihanki", "rojiura", "shuuden", "toorisugari", "yakin"]
QUOTA = {"amayadori": 4, "jihanki": 4, "rojiura": 4, "shuuden": 4, "toorisugari": 9, "yakin": 4}
rows = list(csv.DictReader(open(HERE / "attribution.csv")))
out = []
for wk in sorted({r["file"][:3] for r in rows}):
    R = [r for r in rows if r["file"].startswith(wk)]
    Z = np.array([[max(float(r[f"z_{h}"]), float(r[f"z_{h}_rot180"])) for h in H6] for r in R])
    slots = [h for h in H6 for _ in range(QUOTA[h])]
    cost = -np.stack([Z[:, H6.index(h)] for h in slots], 1)
    ri, ci = linear_sum_assignment(cost)
    for i, j in zip(ri, ci):
        r = R[i]
        h = slots[j]
        rot = int(float(r[f"z_{h}_rot180"]) > float(r[f"z_{h}"]))
        z = Z[i]
        o = np.sort(z)[::-1]
        zh = z[H6.index(h)]
        other = np.max(np.delete(z, H6.index(h)))
        if h != r["photographer"]:
            print(f"{r['file']}: free choice {r['photographer']} (z {float(r['z_best']):.2f}) -> {h} (z {zh:.2f}) by the weekly quota")
        out.append([r["file"], h, rot, f"{zh:.2f}", f"{zh - other:.2f}", "quota" if h != r["photographer"] else ""])
out.sort()
with open(HERE / "attribution_table.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["file", "photographer", "rotated180", "z", "z_margin_vs_next", "note"])
    w.writerows(out)
from collections import Counter
print(Counter(r[1] for r in out))
print("toorisugari:", " ".join(r[0][:-4] for r in out if r[1] == "toorisugari"))
