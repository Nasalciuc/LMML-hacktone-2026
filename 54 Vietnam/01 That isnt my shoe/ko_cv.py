"""Known-brands-only model: 5-fold CV on the 735 task photos, nothing else in training.

Every fold model is then run on photos it never trained on:
  val  : the fold's 147 task photos
  hk   : 1500 web photos (HF ipogorelov/sneakers) of adidas / converse / nike  -> must stay 'known'
  unk  : all web photos of puma, reebok, vans, asics (1200 each)                 -> must be rejected
  oth  : 1500 web photos of other brands (new balance, saucony, ...)             -> must be rejected
Logits go to results/ko_<tag>_k<k>.npz, penultimate features to cache/ko_feat_<tag>_k<k>.npz
as soon as a fold finishes; a rerun skips finished folds.

python ko_cv.py --tag r256 [--res 256] [--epochs 40] [--smooth 0.1] [--loss ce|bce]
"""
import argparse, os, time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

import oe_lib as L

HERE = Path(__file__).resolve().parent
RES_DIR = HERE / "results"
TEST_BRANDS = ["puma", "reebok", "vans", "asics"]
KNOWN = {"adidas": 0, "converse": 1, "nike": 2}
SKIP_OTHER = set(TEST_BRANDS) | set(KNOWN) | {"jordan", "off-white", "onitsuka"}   # onitsuka = asics


def load_all():
    t = np.load(HERE / "cache/train.npz"); h = np.load(HERE / "cache/hf.npz")
    xk = torch.from_numpy(t["x"]).permute(0, 3, 1, 2).contiguous()
    yk = np.array([L.CLASSES.index(s) for s in t["y"]])
    brand = h["brand"]; rng = np.random.default_rng(0)
    hk = np.concatenate([rng.permutation(np.where(brand == b)[0])[:500] for b in KNOWN])
    unk = np.where(np.isin(brand, TEST_BRANDS))[0]
    oth = rng.permutation(np.where(~np.isin(brand, list(SKIP_OTHER)))[0])[:1500]
    idx = np.concatenate([hk, unk, oth])
    xh = torch.from_numpy(h["x"][np.sort(idx)]).permute(0, 3, 1, 2).contiguous()
    pos = np.searchsorted(np.sort(idx), idx)
    sets = dict(hk=pos[:len(hk)], unk=pos[len(hk):len(hk) + len(unk)], oth=pos[len(hk) + len(unk):])
    meta = dict(hk_y=np.array([KNOWN[b] for b in brand[hk]]), unk_brand=brand[unk], oth_brand=brand[oth])
    return xk, yk, xh, sets, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--res", type=int, default=256); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=4e-4); ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--loss", default="ce", choices=["ce", "bce"])
    a = ap.parse_args()
    L.RES = a.res
    logf = open(RES_DIR / f"ko_{a.tag}.log", "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log(f"== {time.ctime()} {vars(a)}")
    xk, yk, xh, sets, meta = load_all()
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for k, (tr, va) in enumerate(skf.split(np.zeros(len(yk)), yk)):
        ff = RES_DIR / f"ko_{a.tag}_k{k}.npz"
        if ff.exists():
            log(f"fold {k}: already done"); continue
        t0 = time.time()
        m = L.train_model(xk[tr], yk[tr], xk[:0], arch=a.arch, epochs=a.epochs, lr=a.lr, smooth=a.smooth,
                          seed=k, n_out=3, log=None, loss_kind=a.loss)
        out, feat = dict(tr=tr, va=va, ytr=yk[tr], yva=yk[va], **meta), {}
        for name, x in [("tr", xk[tr]), ("va", xk[va])] + [(n, xh[i]) for n, i in sets.items()]:
            out["l_" + name] = L.predict_logits(m, x).numpy()
            feat["f_" + name] = L.predict_feats(m, x).numpy().astype(np.float16)
        np.savez(HERE / "cache" / f"ko_feat_{a.tag}_k{k}.npz", **feat)
        tmp = RES_DIR / f"ko_{a.tag}_k{k}.tmp.npz"
        np.savez(tmp, **out); os.replace(tmp, ff)
        acc = (out["l_va"].argmax(1) == yk[va]).mean(); acc_h = (out["l_hk"].argmax(1) == meta["hk_y"]).mean()
        log(f"fold {k}: closed-set acc task val {acc:.3f} | web photos of the 3 brands {acc_h:.3f} | {time.time()-t0:.0f}s")
        del m; torch.cuda.empty_cache()
    log("done")


if __name__ == "__main__":
    main()
