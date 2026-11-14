"""Validation of a training recipe.

A) closed-set: stratified 5-fold, out-of-fold accuracy + open-set scores of known images.
B) pseudo-unknown: hold one brand out entirely, train on the other two (80 %), score the held-out
   20 % of knowns together with every image of the held-out brand. Rotate over the three brands.

Usage: python validate.py --tag NAME [--res 160 --epochs 300 --oe 0.5 --mixup 0.2 --ls 0.0 --seeds 1]
Writes results/<tag>.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import shoelib as L

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "task_vietnam"


def auroc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def gate_table(known_score, known_correct, unk_score):
    """For a range of thresholds, known accuracy (correct and accepted) and rejection of unknowns."""
    out = {}
    for q in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        thr = float(np.quantile(known_score, q)) if q > 0 else -1e9
        acc = float(np.mean(known_correct & (known_score >= thr)))
        rej = float(np.mean(unk_score < thr))
        out[f"q{int(q*100):02d}"] = (round(acc, 3), round(rej, 3))
    return out


def split_idx(y, k, seed):
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(y), int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % k
    return folds


def run_models(xtr, ytr, xte, n_classes, args):
    logits, feats, means = [], [], []
    for s in range(args.seeds):
        m = L.train_model(xtr, torch.as_tensor(ytr), n_classes, res=args.res, epochs=args.epochs, width=args.width,
                          oe_weight=args.oe, mixup=args.mixup, label_smooth=args.ls, lr=args.lr, wd=args.wd, seed=s)
        ltr, ftr = L.infer(m, xtr, args.res)
        lte, fte = L.infer(m, xte, args.res)
        logits.append(lte); feats.append(fte); means.append(L.class_means(ftr, torch.as_tensor(ytr).cuda(), n_classes))
    return L.scores(logits, feats, means)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--res", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--oe", type=float, default=0.0)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--ls", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--skip_closed", action="store_true")
    ap.add_argument("--skip_pseudo", action="store_true")
    args = ap.parse_args()

    import csv
    (HERE / "results").mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(DATA / "train-output.csv")))
    names = [r["filename"] for r in rows]
    y = np.array([L.CLASSES.index(r["label"]) for r in rows])
    x = L.load_images(DATA / "train-input", names)
    res = {"args": vars(args)}
    t0 = time.time()

    # B) pseudo-unknown rotation
    res["pseudo"] = {}
    folds = split_idx(y, 5, 123)
    for u in ([] if args.skip_pseudo else range(3)):
        kn = [c for c in range(3) if c != u]
        remap = {c: i for i, c in enumerate(kn)}
        tr = np.where((y != u) & (folds != 0))[0]
        te_k = np.where((y != u) & (folds == 0))[0]
        te_u = np.where(y == u)[0]
        te = np.concatenate([te_k, te_u])
        ytr = np.array([remap[c] for c in y[tr]])
        sc = run_models(x[tr], ytr, x[te], 2, args)
        nk = len(te_k)
        ytrue = np.array([remap[c] for c in y[te_k]])
        correct = (sc["pred"][:nk].cpu().numpy() == ytrue)
        r = {"closed_acc": float(correct.mean())}
        for key in ["msp", "energy", "cos"]:
            s = sc[key].cpu().numpy()
            r[key] = {"auroc": round(auroc(s[:nk], s[nk:]), 4), "gates": gate_table(s[:nk], correct, s[nk:])}
        res["pseudo"][L.CLASSES[u]] = r
        print(L.CLASSES[u], json.dumps(r), f"{time.time()-t0:.0f}s", flush=True)

    # A) closed-set 5-fold
    if not args.skip_closed:
        oof = {k: np.zeros(len(y)) for k in ["msp", "energy", "cos"]}
        pred = np.zeros(len(y), int)
        for f in range(5):
            tr = np.where(folds != f)[0]
            te = np.where(folds == f)[0]
            sc = run_models(x[tr], y[tr], x[te], 3, args)
            pred[te] = sc["pred"].cpu().numpy()
            for k in oof:
                oof[k][te] = sc[k].cpu().numpy()
        res["closed"] = {"acc": float((pred == y).mean()),
                         "per_class": {L.CLASSES[c]: float((pred[y == c] == c).mean()) for c in range(3)}}
        for k in oof:
            res["closed"][k + "_quantiles"] = [float(v) for v in np.quantile(oof[k], [0.05, 0.1, 0.15, 0.2, 0.25])]
        print("closed", json.dumps(res["closed"]), f"{time.time()-t0:.0f}s", flush=True)
        np.savez(HERE / "results" / f"{args.tag}_oof.npz", pred=pred, y=y, **oof)

    res["seconds"] = time.time() - t0
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(res, open(HERE / "results" / f"{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
