"""Report for the outlier-exposure variant that never trains on the four test brands
(oe_cv.py --holdout_test 1). Rule: 4-class softmax, `bias` added to the unknown logit.

python oe2_report.py <tag>   -> results/oe2_report_<tag>.json
"""
import csv, json, sys

import numpy as np
import torch

import oe_lib as L
from oe_cv import HERE, RES_DIR, TEST_BRANDS

tag = sys.argv[1]
rows = list(csv.DictReader(open(HERE.parent / "task_vietnam/train-output.csv")))
y = np.array([L.CLASSES.index(r["label"]) for r in rows])
folds = [dict(np.load(RES_DIR / f"oe_cv_{tag}_k{k}.npz")) | {"k": k} for k in range(5) if (RES_DIR / f"oe_cv_{tag}_k{k}.npz").exists()]
print(f"== {tag}: {len(folds)} folds")
print(f"{'bias':>5s} | task acc  rej | web-known acc  rej | " + " ".join(f"{b:>6s}" for b in TEST_BRANDS) + " | other brands (seen in training, new photos)")
res = {}
for bias in [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1]:
    T = lambda n, d: L.decide(torch.from_numpy(d[f"k{d['k']}_{n}"]), 0.0, bias).numpy()
    pk = np.concatenate([T("lk", d) for d in folds]); yk = np.concatenate([y[d[f"k{d['k']}_va"]] for d in folds])
    r = dict(task_acc=float((pk == yk).mean()), task_rej=float((pk == 3).mean()),
             web_acc=float(np.mean([(T("lb", d) == d[f"k{d['k']}_yb"]).mean() for d in folds])),
             web_rej=float(np.mean([(T("lb", d) == 3).mean() for d in folds])),
             rej={b: float(np.mean([(T("lh", d)[d[f"k{d['k']}_hbr"] == b] == 3).mean() for d in folds])) for b in TEST_BRANDS},
             rej_seen=float(np.mean([(T("ls", d) == 3).mean() for d in folds])))
    res[bias] = r
    print(f"{bias:5} |   {r['task_acc']:.3f} {r['task_rej']:.3f} |         {r['web_acc']:.3f} {r['web_rej']:.3f} | " +
          " ".join(f"{r['rej'][b]:6.3f}" for b in TEST_BRANDS) + f" | {r['rej_seen']:.3f}")
json.dump(res, open(RES_DIR / f"oe2_report_{tag}.json", "w"), indent=1)
