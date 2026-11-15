"""Honest validation: 5-fold stratified CV on the 735 knowns; in fold k one outlier brand is
removed from training entirely and used as the 'unseen look-alike' test set.

python oe_cv.py --tag b0 [--arch efficientnet_b0] [--res 224] [--epochs 15] [--hf_known 0] [--no_oe]
-> results/oe_cv_<tag>.npz (logits), results/oe_cv_<tag>.json (metrics over tau)
Each fold is written to results/oe_cv_<tag>_k<k>.npz when it finishes; a rerun skips finished folds.
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import oe_lib as L

HERE = Path(__file__).resolve().parent
RES_DIR = HERE / "results"
TEST_BRANDS = ["puma", "reebok", "vans", "asics"]
HOLDOUT = ["puma", "reebok", "vans", "asics", "new"]   # "new" = New Balance in the HF labels   # one per fold
TAUS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def load():
    t = np.load(HERE / "cache/train.npz"); h = np.load(HERE / "cache/hf.npz")
    xk = torch.from_numpy(t["x"]).permute(0, 3, 1, 2).contiguous()
    yk = np.array([L.CLASSES.index(s) for s in t["y"]])
    xh = torch.from_numpy(h["x"]).permute(0, 3, 1, 2).contiguous()
    return xk, yk, xh, h["brand"]


def metrics(lk, yk, lu, tau):
    pk = L.decide(lk, tau).numpy(); pu = L.decide(lu, tau).numpy()
    return float((pk == yk).mean()), float((pu == 3).mean())


def score(l):  # higher = more 'known'
    p = l.softmax(1)
    return (p[:, :3].max(1).values - p[:, 3]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--hf_known", type=int, default=0)
    ap.add_argument("--no_oe", action="store_true"); ap.add_argument("--u_ratio", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=4e-4); ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--res", type=int, default=224); ap.add_argument("--feats", type=int, default=0)
    ap.add_argument("--holdout_test", type=int, default=0)
    a = ap.parse_args()
    L.RES = a.res
    RES_DIR.mkdir(exist_ok=True)
    logf = open(RES_DIR / f"oe_cv_{a.tag}.log", "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log(f"== {time.ctime()} {vars(a)}")
    xk, yk, xh, brand = load()
    unk_mask = ~np.isin(brand, list(L.NOT_UNKNOWN))
    rng = np.random.default_rng(0)
    # 15 % of the images of the *training* outlier brands never go into training: 'seen brand, new photo'
    seen_hold = rng.random(len(brand)) < 0.15
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    out = {}
    for k, (tr, va) in enumerate(skf.split(np.zeros(len(yk)), yk)):
        if str(k) not in a.folds.split(","):
            continue
        hb = HOLDOUT[k]
        fold_file = RES_DIR / f"oe_cv_{a.tag}_k{k}.npz"
        if fold_file.exists():                       # finished before a restart: reuse
            out.update(dict(np.load(fold_file)))
            log(f"fold {k}: already done, loaded {fold_file.name}")
            continue
        u_train = np.where(unk_mask & (brand != hb) & ~seen_hold)[0]
        u_seen = np.where(unk_mask & (brand != hb) & seen_hold)[0]
        u_hold = np.where(brand == hb)[0]
        if a.holdout_test:     # the four test brands never in training, in any fold (onitsuka = asics, dropped too)
            hb = "+".join(TEST_BRANDS)
            banned = np.isin(brand, TEST_BRANDS + ["onitsuka"])
            u_train = np.where(unk_mask & ~banned & ~seen_hold)[0]
            u_seen = np.where(unk_mask & ~banned & seen_hold)[0]
            u_hold = np.where(np.isin(brand, TEST_BRANDS))[0]
            out[f"k{k}_hbr"] = brand[u_hold]
        xtr, ytr = xk[tr], yk[tr]
        if a.hf_known:
            kn = {"adidas": 0, "converse": 1, "nike": 2}
            hk = np.where(np.isin(brand, list(kn)) & ~seen_hold)[0]
            hk = hk[rng.permutation(len(hk))[: a.hf_known]]
            xtr = torch.cat([xtr, xh[hk]]); ytr = np.concatenate([ytr, [kn[b] for b in brand[hk]]])
        xu = xh[u_train] if not a.no_oe else xh[:0]
        log(f"fold {k}: hold-out brand {hb} ({len(u_hold)} imgs), train knowns {len(xtr)}, train outliers {len(xu)}")
        t0 = time.time()
        m = L.train_model(xtr, ytr, xu, arch=a.arch, epochs=a.epochs, lr=a.lr, u_ratio=a.u_ratio, seed=k, log=log)
        lk = L.predict_logits(m, xk[va]); lh = L.predict_logits(m, xh[u_hold]); ls = L.predict_logits(m, xh[u_seen])
        out[f"k{k}_va"] = va; out[f"k{k}_lk"] = lk.numpy(); out[f"k{k}_lh"] = lh.numpy(); out[f"k{k}_ls"] = ls.numpy()
        # doctored copies of the (unseen) validation photos: one 24-48 px square pasted/blurred/flattened/shifted
        from oe_bank import pil_tf
        er = np.random.default_rng(100 + k); src = xk.permute(0, 2, 3, 1).numpy()
        ed = np.stack([pil_tf(src[i], ["paste", "blur", "flatten", "shift"][j % 4], er, xh[u_hold[:200]].permute(0, 2, 3, 1).numpy())
                       for j, i in enumerate(va)])
        out[f"k{k}_le"] = L.predict_logits(m, torch.from_numpy(ed).permute(0, 3, 1, 2).contiguous()).numpy()
        # web photos of the three known brands that were never trained on: does the net reject a
        # photo because of its source rather than its brand?
        kn_hold = np.where(np.isin(brand, ["adidas", "converse", "nike"]) & seen_hold)[0]
        out[f"k{k}_lb"] = L.predict_logits(m, xh[kn_hold]).numpy()
        out[f"k{k}_yb"] = np.array([L.CLASSES.index(b) for b in brand[kn_hold]])
        pb = L.decide(torch.from_numpy(out[f"k{k}_lb"]), 0.0).numpy()
        log(f"  fold {k}: web photos of known brands (not trained on): accuracy {(pb == out[f'k{k}_yb']).mean():.3f}, labelled unknown {(pb == 3).mean():.3f}")
        for tau in [0.0, 0.5, 0.7]:
            acc, rej = metrics(lk, yk[va], lh, tau); _, rs = metrics(lk, yk[va], ls, tau)
            log(f"  fold {k} tau {tau}: known acc {acc:.3f}  rejection held-out {hb} {rej:.3f}  seen-brand new photos {rs:.3f}")
        log(f"  closed-set acc {(lk[:, :3].argmax(1).numpy() == yk[va]).mean():.3f}  AUROC held-out {roc_auc_score(np.r_[np.ones(len(lk)), np.zeros(len(lh))], np.r_[score(lk), score(lh)]):.3f}  ({time.time()-t0:.0f}s)")
        if a.feats:      # penultimate features for the feature-space scores (oe_featspace.py)
            np.savez(HERE / "cache" / f"feat_{a.tag}_k{k}.npz",
                     ftr=L.predict_feats(m, xtr).numpy().astype(np.float16), ytr=ytr,
                     fva=L.predict_feats(m, xk[va]).numpy().astype(np.float16), yva=yk[va],
                     fho=L.predict_feats(m, xh[u_hold]).numpy().astype(np.float16),
                     fwk=L.predict_feats(m, xh[kn_hold]).numpy().astype(np.float16), ywk=out[f"k{k}_yb"])
        tmp = RES_DIR / f"oe_cv_{a.tag}_k{k}.tmp.npz"
        np.savez(tmp, **{n: v for n, v in out.items() if n.startswith(f"k{k}_")})
        os.replace(tmp, fold_file)                   # a fold is on disk as soon as it is finished
        del m; torch.cuda.empty_cache()
    np.savez(RES_DIR / f"oe_cv_{a.tag}.npz", **out)
    # pooled summary
    ks = sorted({int(n[1]) for n in out if n.endswith("_lk")})
    LK = torch.from_numpy(np.concatenate([out[f"k{k}_lk"] for k in ks])); YK = np.concatenate([yk[out[f"k{k}_va"]] for k in ks])
    summ = {"args": vars(a), "folds": ks, "by_tau": {}}
    for tau in TAUS:
        acc = float((L.decide(LK, tau).numpy() == YK).mean())
        rej = {f"{HOLDOUT[k]}" if not a.holdout_test else f"fold{k}": float((L.decide(torch.from_numpy(out[f"k{k}_lh"]), tau).numpy() == 3).mean()) for k in ks}
        rs = float(np.mean([(L.decide(torch.from_numpy(out[f"k{k}_ls"]), tau).numpy() == 3).mean() for k in ks]))
        summ["by_tau"][tau] = {"known_acc": acc, "rej_heldout_mean": float(np.mean(list(rej.values()))), "rej_heldout": rej, "rej_seen_brands": rs}
        log(f"POOLED tau {tau}: known acc {acc:.3f} | held-out rejection mean {np.mean(list(rej.values())):.3f} {json.dumps({b: round(v, 3) for b, v in rej.items()})} | seen-brand new photos {rs:.3f}")
    json.dump(summ, open(RES_DIR / f"oe_cv_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
