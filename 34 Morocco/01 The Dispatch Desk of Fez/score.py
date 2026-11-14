"""Score main.py's routing on the training requests.

    python score.py ../task_morroco

Runs the router on train-input.csv, then reports quality and cost, per
problem family too, and a stress test where the family mix is reweighted.
"""
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import main as router

data = Path(sys.argv[1] if len(sys.argv) > 1 else "../task_morroco")
prompts = {r["request_id"]: r["prompt"] for r in csv.DictReader(open(data / "train-input.csv", encoding="utf-8-sig"))}
ans = {r["request_id"]: r for r in csv.DictReader(open(data / "train-answers.csv", encoding="utf-8-sig"))}


def family(p):
    for key in ("Walking", "stall fee", "takings", "pauses", "stall opens", "sells whatever", "board lists",
                "sells", "loads", "heavier", "gate charges"):
        if key in p:
            return key
    return "other"


def correct(rid, model):
    a = ans[rid]
    return a[model + "_answer"].strip().lower() == a["reference_answer"].strip().lower()


route = {rid: router.route(p) for rid, p in prompts.items()}
n = len(route)
q = sum(correct(rid, m) for rid, m in route.items()) / n
cost = sum(25 if m == "expensive" else 1 for m in route.values())
print(f"train: quality {q:.4f} (gate 0.80)  cost {cost} (gate {20 * n})  avg {cost / n:.2f}/req")

fam = defaultdict(list)
for rid, p in prompts.items():
    fam[family(p)].append(rid)
for k, ids in sorted(fam.items(), key=lambda x: -len(x[1])):
    ok = sum(correct(r, route[r]) for r in ids)
    exp = sum(route[r] == "expensive" for r in ids)
    print(f"  {k:15s} n={len(ids):3d} correct={ok:3d} expensive={exp:3d}")

rng = random.Random(0)
qs, cs = [], []
for _ in range(2000):
    w = {k: rng.uniform(0.6, 1.4) for k in fam}
    pool = [rid for k, ids in fam.items() for rid in ids]
    weights = [w[family(prompts[r])] for r in pool]
    sample = rng.choices(pool, weights=weights, k=500)
    qs.append(sum(correct(r, route[r]) for r in sample) / 500)
    cs.append(sum(25 if route[r] == "expensive" else 1 for r in sample))
passed = sum(a >= 0.8 and b <= 10000 for a, b in zip(qs, cs)) / 2000
qs.sort(); cs.sort()
print(f"500-req resamples, family weights x0.6-1.4: quality p5 {qs[100]:.3f}  cost p95 {cs[1900]}  "
      f"pass rate {passed:.3f}")
