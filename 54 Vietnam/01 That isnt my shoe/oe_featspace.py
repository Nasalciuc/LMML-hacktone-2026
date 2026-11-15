"""Feature-space open-set scores on the CV models, next to the 4th-class score.

python oe_featspace.py b0_hfk_r256_e25
Per fold: knowns = task validation photos / unseen web photos of the three brands,
unknowns = the outlier brand that fold never trained on. Higher score = more 'known'.
Reports AUROC and the share of held-out-brand photos rejected when 10 % of the knowns are rejected.
-> results/oe_featspace_<tag>.json
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from oe_cv import HOLDOUT, RES_DIR

HERE = Path(__file__).resolve().parent


def norm(f):
    return f / f.norm(dim=1, keepdim=True).clamp(min=1e-6)


def scores(d, f):
    """dict name -> (task val, web knowns, held-out brand) score tensors"""
    k = d["k"]
    lk, lb, lh = [torch.from_numpy(d[f"k{k}_{n}"]) for n in ("lk", "lb", "lh")]
    ftr, fva, fwk, fho = [torch.from_numpy(f[n].astype(np.float32)) for n in ("ftr", "fva", "fwk", "fho")]
    ytr = torch.from_numpy(f["ytr"])
    out = {}
    def four(l):
        p = l.softmax(1); return p[:, :3].max(1).values - p[:, 3]
    out["4th class (p_known - p_unknown)"] = [four(x) for x in (lk, lb, lh)]
    out["max softmax, 3 known logits"] = [x[:, :3].softmax(1).max(1).values for x in (lk, lb, lh)]
    out["energy, 3 known logits"] = [torch.logsumexp(x[:, :3], 1) for x in (lk, lb, lh)]
    # prototypes
    ntr = norm(ftr)
    proto = norm(torch.stack([ntr[ytr == c].mean(0) for c in range(3)]))
    out["prototype cosine"] = [(norm(x) @ proto.T).max(1).values for x in (fva, fwk, fho)]
    # Mahalanobis, shared covariance
    mu = torch.stack([ftr[ytr == c].mean(0) for c in range(3)])
    xc = ftr - mu[ytr]
    cov = (xc.T @ xc) / len(xc)
    cov = cov + 0.05 * cov.diag().mean() * torch.eye(cov.shape[0])
    P = torch.linalg.inv(cov.double()).float()
    def maha(x):
        dd = [(((x - m) @ P) * (x - m)).sum(1) for m in mu]
        return -torch.stack(dd, 1).min(1).values
    out["Mahalanobis (shared cov)"] = [maha(x) for x in (fva, fwk, fho)]
    # kNN
    def knn(x, kk=10):
        return (norm(x) @ ntr.T).topk(kk, dim=1).values[:, -1]
    out["kNN cosine (k=10)"] = [knn(x) for x in (fva, fwk, fho)]
    return out


def main():
    tag = sys.argv[1]
    res = {}
    for k in range(5):
        fp, cp = RES_DIR / f"oe_cv_{tag}_k{k}.npz", HERE / "cache" / f"feat_{tag}_k{k}.npz"
        if not (fp.exists() and cp.exists()):
            continue
        d = dict(np.load(fp)); d["k"] = k
        for name, (sv, sw, sh) in scores(d, np.load(cp)).items():
            sv, sw, sh = sv.numpy(), sw.numpy(), sh.numpy()
            kn = np.r_[sv, sw]
            thr = np.quantile(kn, 0.10)
            res.setdefault(name, []).append(dict(
                brand=HOLDOUT[k],
                auroc_task=float(roc_auc_score(np.r_[np.ones(len(sv)), np.zeros(len(sh))], np.r_[sv, sh])),
                auroc_web=float(roc_auc_score(np.r_[np.ones(len(sw)), np.zeros(len(sh))], np.r_[sw, sh])),
                rej_at_10=float((sh < thr).mean())))
    print(f"{'score':34s} {'AUROC task-val':>14s} {'AUROC web-known':>15s} {'rej@10% knowns lost':>20s}   per-brand rej@10%")
    for name, r in res.items():
        print(f"{name:34s} {np.mean([x['auroc_task'] for x in r]):14.3f} {np.mean([x['auroc_web'] for x in r]):15.3f} "
              f"{np.mean([x['rej_at_10'] for x in r]):20.3f}   " + " ".join(f"{x['brand']} {x['rej_at_10']:.2f}" for x in r))
    json.dump(res, open(RES_DIR / f"oe_featspace_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
