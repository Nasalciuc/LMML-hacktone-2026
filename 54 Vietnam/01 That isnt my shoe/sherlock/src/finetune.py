"""Fine-tune a timm backbone on the 3 known brands and measure closed-set accuracy + open-set proxies.

usage:
  python -I finetune.py --name effb0_ns --split holdout --epochs 15 --out runs/x
  --split holdout   : stratified 80/20 split (seeded), val = known only
  --split loco:K    : class K (0 nike,1 adidas,2 converse) removed from training; it becomes the 'unknown' set,
                      and a 20% split of the remaining classes is the known validation set
  --split full      : train on everything (final model)
"""
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_feats import CANDS, build, load_images  # noqa: E402

CLASSES = ['nike', 'adidas', 'converse']


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--data', default=f'{DATA}')
    p.add_argument('--weights', default=f'{WEIGHTS}')
    p.add_argument('--split', default='holdout')
    p.add_argument('--size', type=int, default=0)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--bs', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--head_lr_mult', type=float, default=10.0)
    p.add_argument('--wd', type=float, default=0.02)
    p.add_argument('--ls', type=float, default=0.0)
    p.add_argument('--scale_min', type=float, default=0.55)
    p.add_argument('--freeze_frac', type=float, default=0.0, help='fraction of leading param tensors frozen')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--out', required=True)
    return p.parse_args()


def make_split(y, mode, seed):
    rng = np.random.RandomState(seed)
    n = len(y)
    if mode == 'full':
        return np.arange(n), np.array([], dtype=int), np.array([], dtype=int)
    if mode.startswith('kfold:'):
        k = int(mode.split(':')[1])
        fold_of = np.zeros(n, dtype=int)
        r = np.random.RandomState(1234)
        for c in range(3):
            idx = np.where(y == c)[0]
            r.shuffle(idx)
            fold_of[idx] = np.arange(len(idx)) % 5
        return np.where(fold_of != k)[0], np.where(fold_of == k)[0], np.array([], dtype=int)
    held = None
    if mode.startswith('loco:'):
        held = int(mode.split(':')[1])
    tr, va = [], []
    for c in range(3):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        if c == held:
            continue
        k = int(round(0.2 * len(idx)))
        va += list(idx[:k])
        tr += list(idx[k:])
    unk = np.where(y == held)[0] if held is not None else np.array([], dtype=int)
    return np.array(sorted(tr)), np.array(sorted(va)), unk


def augment(xb_u8, size, scale_min, gen):
    """xb_u8: uint8 [B,3,256,256] -> float normalised later. Per-sample random resized crop + flip + jitter."""
    B = xb_u8.shape[0]
    out = torch.empty(B, 3, size, size)
    for i in range(B):
        x = xb_u8[i].float() / 255.0
        H, W = x.shape[1:]
        for _ in range(10):
            s = float(torch.empty(1).uniform_(scale_min, 1.0, generator=gen))
            r = math.exp(float(torch.empty(1).uniform_(math.log(3 / 4), math.log(4 / 3), generator=gen)))
            w = int(round(math.sqrt(s * H * W * r)))
            h = int(round(math.sqrt(s * H * W / r)))
            if w <= W and h <= H:
                break
        else:
            w, h = W, H
        top = int(torch.randint(0, H - h + 1, (1,), generator=gen))
        left = int(torch.randint(0, W - w + 1, (1,), generator=gen))
        crop = x[:, top:top + h, left:left + w].unsqueeze(0)
        crop = F.interpolate(crop, size=(size, size), mode='bilinear', align_corners=False, antialias=True)[0]
        if float(torch.rand(1, generator=gen)) < 0.5:
            crop = torch.flip(crop, dims=[2])
        # colour jitter (brightness / contrast / saturation)
        b = float(torch.empty(1).uniform_(0.8, 1.2, generator=gen))
        c = float(torch.empty(1).uniform_(0.8, 1.2, generator=gen))
        sat = float(torch.empty(1).uniform_(0.7, 1.3, generator=gen))
        crop = crop * b
        m = crop.mean()
        crop = (crop - m) * c + m
        g = crop.mean(0, keepdim=True)
        crop = (crop - g) * sat + g
        if float(torch.rand(1, generator=gen)) < 0.05:
            crop = crop.mean(0, keepdim=True).expand(3, -1, -1).clone()
        out[i] = crop.clamp(0, 1)
    return out


class Net(nn.Module):
    def __init__(self, backbone, dim, ncls=3):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(dim, ncls)

    def forward(self, x):
        f = self.backbone(x)
        return self.head(f), f


@torch.no_grad()
def embed(model, x_u8, idx, size, mean, std, bs=64):
    model.eval()
    logits, feats = [], []
    for i in range(0, len(idx), bs):
        xb = x_u8[idx[i:i + bs]].float() / 255.0
        if size != xb.shape[-1]:
            xb = F.interpolate(xb, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
        xb = (xb - mean) / std
        l1, f1 = model(xb)
        l2, f2 = model(torch.flip(xb, dims=[3]))
        logits.append((l1 + l2) / 2)
        feats.append((f1 + f2) / 2)
    return torch.cat(logits).numpy(), torch.cat(feats).numpy()


def ood_scores(logits, feats, bank_feats, bank_y, class_means=None):
    from scipy.special import logsumexp
    s = {}
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    s['msp'] = p.max(1)
    s['mls'] = logits.max(1)
    s['energy'] = logsumexp(logits, axis=1)
    fn = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    bn = bank_feats / np.linalg.norm(bank_feats, axis=1, keepdims=True)
    sim = fn @ bn.T
    srt = -np.sort(-sim, axis=1)
    s['knn1'] = srt[:, 0]
    s['knn5'] = srt[:, 4]
    # cosine to class centroids (normalised features)
    cents = np.stack([bn[bank_y == c].mean(0) for c in np.unique(bank_y)])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    s['cent'] = (fn @ cents.T).max(1)
    return s


def summarize(ks, us):
    from sklearn.metrics import roc_auc_score
    out = {}
    for k in ks:
        kk, uu = ks[k], us[k]
        yy = np.concatenate([np.ones_like(kk), np.zeros_like(uu)])
        au = roc_auc_score(yy, np.concatenate([kk, uu]))
        r95 = float((uu < np.quantile(kk, 0.05)).mean())
        r90 = float((uu < np.quantile(kk, 0.10)).mean())
        out[k] = dict(auroc=round(au, 4), rej_keep95=round(r95, 3), rej_keep90=round(r90, 3))
    return out


def main():
    a = parse()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    x_u8, labels, files = load_images(a.data)
    y = np.array([CLASSES.index(l) for l in labels])
    tr, va, unk = make_split(y, a.split, a.seed)
    held = int(a.split.split(':')[1]) if a.split.startswith('loco:') else None
    backbone, mean, std, size0, cfg = build(a.name, a.weights)
    size = a.size or size0
    with torch.no_grad():
        dim = backbone(torch.zeros(1, 3, size, size)).shape[1]
    # label remap for loco (2 classes)
    cls_present = sorted(set(y[tr].tolist()))
    remap = {c: i for i, c in enumerate(cls_present)}
    ytr = torch.tensor([remap[c] for c in y[tr]])
    model = Net(backbone, dim, len(cls_present))
    params = list(model.backbone.named_parameters())
    nfreeze = int(len(params) * a.freeze_frac)
    for i, (n, p) in enumerate(params):
        p.requires_grad = i >= nfreeze
    bb_params = [p for _, p in params if p.requires_grad]
    opt = torch.optim.AdamW([
        {'params': bb_params, 'lr': a.lr},
        {'params': model.head.parameters(), 'lr': a.lr * a.head_lr_mult},
    ], weight_decay=a.wd)
    steps_per_epoch = math.ceil(len(tr) / a.bs)
    total = steps_per_epoch * a.epochs
    warm = steps_per_epoch
    base_lrs = [g['lr'] for g in opt.param_groups]
    gen = torch.Generator().manual_seed(a.seed)
    step = 0
    t0 = time.time()
    hist = []
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=gen)
        tot_loss, tot_ok, tot_n = 0.0, 0, 0
        for bi in range(steps_per_epoch):
            sel = perm[bi * a.bs:(bi + 1) * a.bs]
            if len(sel) < 2:
                continue
            xb = augment(x_u8[tr[sel.numpy()]], size, a.scale_min, gen)
            xb = (xb - mean) / std
            yb = ytr[sel]
            f = step / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, bl in zip(opt.param_groups, base_lrs):
                g['lr'] = bl * f
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=a.ls)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            tot_loss += loss.item() * len(sel)
            tot_ok += (logits.argmax(1) == yb).sum().item()
            tot_n += len(sel)
        rec = dict(ep=ep, loss=round(tot_loss / tot_n, 4), tr_acc=round(tot_ok / tot_n, 4), t=round(time.time() - t0))
        if len(va) and (ep == a.epochs - 1 or ep % 3 == 2):
            lv, _ = embed(model, x_u8, va, size, mean, std)
            yv = np.array([remap[c] for c in y[va]])
            rec['va_acc'] = round(float((lv.argmax(1) == yv).mean()), 4)
        hist.append(rec)
        print(json.dumps(rec), flush=True)
    # ---------- final evaluation ----------
    res = dict(args=vars(a), hist=hist)
    if len(va):
        lb, fb = embed(model, x_u8, tr, size, mean, std)
        lv, fv = embed(model, x_u8, va, size, mean, std)
        ybank = np.array([remap[c] for c in y[tr]])
        yv = np.array([remap[c] for c in y[va]])
        res['va_acc'] = float((lv.argmax(1) == yv).mean())
        np.savez(os.path.join(a.out, 'emb.npz'), lb=lb, fb=fb, ybank=ybank, lv=lv, fv=fv, yv=yv, va=va, tr=tr)
        if len(unk):
            lu, fu = embed(model, x_u8, unk, size, mean, std)
            ks = ood_scores(lv, fv, fb, ybank)
            us = ood_scores(lu, fu, fb, ybank)
            res['osr'] = summarize(ks, us)
            np.savez(os.path.join(a.out, 'emb.npz'), lb=lb, fb=fb, ybank=ybank, lv=lv, fv=fv, yv=yv,
                     lu=lu, fu=fu, va=va, tr=tr, unk=unk)
    torch.save({'state_dict': model.state_dict(), 'name': a.name, 'size': size, 'dim': dim,
                'classes': [CLASSES[c] for c in cls_present]}, os.path.join(a.out, 'model.pt'))
    json.dump(res, open(os.path.join(a.out, 'result.json'), 'w'), indent=1)
    print('RESULT', json.dumps({k: res[k] for k in res if k in ('va_acc', 'osr')}), flush=True)


if __name__ == '__main__':
    main()
