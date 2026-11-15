"""Open-set scores of the known-brands-only CV models (ko_cv.py).

python ko_scores.py r256_s10 [more tags] [--acc 0.88]
A photo is labelled `unknown` when its score is below a threshold, otherwise argmax of the 3 logits.
The threshold is the highest one that keeps  known accuracy (correct AND accepted) >= --acc
on the pooled out-of-fold task photos. Web photos are never used to pick anything.
-> results/ko_scores_<tag>.json
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ko_cv import HERE, RES_DIR, TEST_BRANDS

SETS = ["va", "hk", "unk", "oth"]


def norm(f):
    return f / f.norm(dim=1, keepdim=True).clamp(min=1e-6)


def fold_scores(d, f):
    lg = {s: torch.from_numpy(d["l_" + s]).float() for s in SETS}
    ft = {s: torch.from_numpy(f["f_" + s].astype(np.float32)) for s in SETS + ["tr"]}
    ytr = torch.from_numpy(d["ytr"])
    out = {}
    out["max softmax"] = {s: lg[s].softmax(1).max(1).values for s in SETS}
    out["max logit"] = {s: lg[s].max(1).values for s in SETS}
    out["energy (logsumexp)"] = {s: torch.logsumexp(lg[s], 1) for s in SETS}
    ntr = norm(ft["tr"])
    proto = norm(torch.stack([ntr[ytr == c].mean(0) for c in range(3)]))
    out["prototype cosine"] = {s: (norm(ft[s]) @ proto.T).max(1).values for s in SETS}
    mu = torch.stack([ft["tr"][ytr == c].mean(0) for c in range(3)])
    xc = ft["tr"] - mu[ytr]
    cov = (xc.T @ xc) / len(xc)
    cov = cov + 0.1 * cov.diag().mean() * torch.eye(cov.shape[0])
    P = torch.linalg.inv(cov.double()).float()
    def maha(x):
        return -torch.stack([(((x - m) @ P) * (x - m)).sum(1) for m in mu], 1).min(1).values
    out["Mahalanobis"] = {s: maha(ft[s]) for s in SETS}
    out["kNN cosine k=1"] = {s: (norm(ft[s]) @ ntr.T).max(1).values for s in SETS}
    out["kNN cosine k=5"] = {s: (norm(ft[s]) @ ntr.T).topk(5, dim=1).values[:, -1] for s in SETS}
    out["max softmax x prototype cosine"] = {s: out["max softmax"][s] * out["prototype cosine"][s].clamp(min=0) for s in SETS}
    return {n: {s: v.numpy() for s, v in sc.items()} for n, sc in out.items()}


def run(tag, acc_min):
    folds = []
    for k in range(5):
        fp, cp = RES_DIR / f"ko_{tag}_k{k}.npz", HERE / "cache" / f"ko_feat_{tag}_k{k}.npz"
        if fp.exists() and cp.exists():
            d = dict(np.load(fp)); folds.append((k, d, fold_scores(d, np.load(cp))))
    if not folds:
        print(tag, "no folds on disk"); return
    ok_va = np.concatenate([d["l_va"].argmax(1) == d["yva"] for _, d, _ in folds])
    hk_ok = [d["l_hk"].argmax(1) == d["hk_y"] for _, d, _ in folds]
    print(f"== {tag}: {len(folds)} folds, closed-set accuracy task photos {ok_va.mean():.3f}, web photos of the 3 brands {np.mean(hk_ok):.3f}")
    print(f"   threshold = highest one with task known accuracy >= {acc_min}")
    hdr = f"{'score':32s} {'thr':>8s} | task acc  rej | web-known acc  rej | " + " ".join(f"{b:>6s}" for b in TEST_BRANDS) + " |  other | AUROC task/web vs 4 brands"
    print(hdr)
    res = {"closed_task": float(ok_va.mean()), "closed_web": float(np.mean(hk_ok)), "acc_min": acc_min, "scores": {}}
    for name in folds[0][2]:
        sv = np.concatenate([sc[name]["va"] for _, _, sc in folds])
        cand = np.unique(sv)[::-1]                      # try thresholds from strict to loose
        acc_at = np.array([(ok_va & (sv >= t)).mean() for t in cand])
        good = np.where(acc_at >= acc_min)[0]
        if not len(good):
            print(f"{name:32s} no threshold reaches {acc_min}"); continue
        thr = float(cand[good[0]])
        r = dict(thr=thr, task_acc=float((ok_va & (sv >= thr)).mean()), task_rej=float((sv < thr).mean()))
        r["web_acc"] = float(np.mean([(o & (sc[name]["hk"] >= thr)).mean() for o, (_, _, sc) in zip(hk_ok, folds)]))
        r["web_rej"] = float(np.mean([(sc[name]["hk"] < thr).mean() for _, _, sc in folds]))
        r["web_acc_by_brand"] = {b: float(np.mean([(o & (sc[name]["hk"] >= thr))[d["hk_y"] == i].mean() for o, (_, d, sc) in zip(hk_ok, folds)]))
                                 for i, b in enumerate(["adidas", "converse", "nike"])}
        r["rej"] = {b: float(np.mean([(sc[name]["unk"][d["unk_brand"] == b] < thr).mean() for _, d, sc in folds])) for b in TEST_BRANDS}
        r["rej_fold_min_max"] = {b: [float(f(np.array([(sc[name]["unk"][d["unk_brand"] == b] < thr).mean() for _, d, sc in folds]))) for f in (np.min, np.max)]
                                 for b in TEST_BRANDS}
        r["rej_other"] = float(np.mean([(sc[name]["oth"] < thr).mean() for _, _, sc in folds]))
        r["auroc_task"] = float(np.mean([roc_auc_score(np.r_[np.ones(len(sc[name]["va"])), np.zeros(len(sc[name]["unk"]))], np.r_[sc[name]["va"], sc[name]["unk"]]) for _, _, sc in folds]))
        r["auroc_web"] = float(np.mean([roc_auc_score(np.r_[np.ones(len(sc[name]["hk"])), np.zeros(len(sc[name]["unk"]))], np.r_[sc[name]["hk"], sc[name]["unk"]]) for _, _, sc in folds]))
        res["scores"][name] = r
        print(f"{name:32s} {thr:8.3f} |   {r['task_acc']:.3f} {r['task_rej']:.3f} |         {r['web_acc']:.3f} {r['web_rej']:.3f} | " +
              " ".join(f"{r['rej'][b]:6.3f}" for b in TEST_BRANDS) + f" | {r['rej_other']:6.3f} | {r['auroc_task']:.3f} / {r['auroc_web']:.3f}")
    json.dump(res, open(RES_DIR / f"ko_scores_{tag}_acc{int(acc_min*100)}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("tags", nargs="+"); ap.add_argument("--acc", type=float, default=0.88)
    a = ap.parse_args()
    for t in a.tags:
        run(t, a.acc)
