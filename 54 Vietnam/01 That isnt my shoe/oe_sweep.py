"""Sweep the open-set rule (unknown-logit bias, tau) over saved CV logits.
python oe_sweep.py b0_oe [more tags]"""
import json, sys

import numpy as np
import torch

import oe_lib as L
from oe_cv import HOLDOUT, RES_DIR



def main():
    import csv
    from pathlib import Path
    rows = list(csv.DictReader(open(Path(__file__).resolve().parent.parent / "task_vietnam/train-output.csv")))
    y = np.array([L.CLASSES.index(r["label"]) for r in rows])
    for tag in sys.argv[1:]:
        d = np.load(RES_DIR / f"oe_cv_{tag}.npz")
        ks = sorted({int(n[1]) for n in d.files if n.endswith("_lk")})
        LK = torch.from_numpy(np.concatenate([d[f"k{k}_lk"] for k in ks])); YK = np.concatenate([y[d[f"k{k}_va"]] for k in ks])
        print(f"== {tag}  folds {ks}  closed-set acc {(LK[:, :3].argmax(1).numpy() == YK).mean():.3f}")
        best = []
        for bias in [-6, -5, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1]:
            for tau in [0.0, 0.5, 0.6, 0.7]:
                acc = (L.decide(LK, tau, bias).numpy() == YK).mean()
                rej = {HOLDOUT[k]: (L.decide(torch.from_numpy(d[f"k{k}_lh"]), tau, bias).numpy() == 3).mean() for k in ks}
                rs = np.mean([(L.decide(torch.from_numpy(d[f"k{k}_ls"]), tau, bias).numpy() == 3).mean() for k in ks])
                rmin = min(rej.values())
                re_ = np.mean([(L.decide(torch.from_numpy(d[f"k{k}_le"]), tau, bias).numpy() == 3).mean() for k in ks]) if f"k{ks[0]}_le" in d.files else float("nan")
                if f"k{ks[0]}_lb" in d.files:
                    pb = [L.decide(torch.from_numpy(d[f"k{k}_lb"]), tau, bias).numpy() for k in ks]
                    wa = np.mean([(p == d[f"k{k}_yb"]).mean() for p, k in zip(pb, ks)]); wr = np.mean([(p == 3).mean() for p in pb])
                else:
                    wa = wr = float("nan")
                best.append((min(acc - 0.85, rmin - 0.70), bias, tau, acc, rej, rs))
                if tau == 0.0 or (bias in (-3, -2, -1, 0)):
                    print(f"bias {bias:5} tau {tau:.1f}: known acc {acc:.3f} | held-out rejection " +
                          " ".join(f"{b} {v:.3f}" for b, v in rej.items()) + f" (min {rmin:.3f}) | seen-brand new photos {rs:.3f} | doctored val photos {re_:.3f} | web known-brand photos acc {wa:.3f} rejected {wr:.3f}")
        b = max(best, key=lambda t: t[0])
        print(f"best margin {b[0]:.3f}: bias {b[1]} tau {b[2]} acc {b[3]:.3f} rej min {min(b[4].values()):.3f}")


if __name__ == "__main__":
    main()
