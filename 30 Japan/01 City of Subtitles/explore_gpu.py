"""GPU leave-one-out benchmark of residual variants on the credited photos."""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import prnu
from prnu import gblur

dev = "cuda"
credited, anonymous = prnu.list_photos()
handles = list(credited)
paths = [p for h in handles for p in credited[h]]
labels = np.repeat(np.arange(6), 18)

t = time.time()
X = torch.stack([torch.from_numpy(prnu.load_rgb(p)).permute(2, 0, 1).to(torch.uint8) for p in paths]).to(dev)
M = torch.stack([torch.from_numpy(prnu.caption_mask(prnu.load_rgb(p))) for p in paths]).to(dev)
print("loaded", time.time() - t, flush=True)


def boxf(x, k):
    return F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="reflect"), k, 1)


def wiener(x, k=5, nv=None):
    mu = boxf(x, k)
    var = boxf(x * x, k) - mu * mu
    if nv is None:
        nv = var.mean(dim=(2, 3), keepdim=True)
    return mu + (var - nv).clamp(min=0) / var.clamp(min=1e-6) * (x - mu)


def medf(x, k=3):
    p = F.pad(x, (k // 2,) * 4, mode="reflect")
    u = F.unfold(p, k)  # N, C*k*k, H*W
    n, c = x.shape[:2]
    u = u.view(n, c, k * k, -1).median(2).values
    return u.view_as(x)


def atrous(x, j):
    h = torch.tensor([1, 4, 6, 4, 1], device=x.device, dtype=torch.float32) / 16
    k = torch.zeros(4 * 2 ** j + 1, device=x.device)
    k[:: 2 ** j] = h
    p = len(k) // 2
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, p, p), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return x


SIG = [0.889, 0.200, 0.086, 0.041]


def star_resid(x, levels=4, wins=(3, 5, 7, 9)):
    c = x
    res = 0
    sigma = None
    for j in range(levels):
        c2 = atrous(c, j)
        w = c - c2
        if sigma is None:
            sigma = (w.abs().flatten(2).median(2).values / 0.6745 / SIG[0]).view(-1, w.shape[1], 1, 1)
        nv = (sigma * SIG[j]) ** 2
        lv = None
        for k in wins:
            v = (boxf(w * w, k) - nv).clamp(min=0)
            lv = v if lv is None else torch.minimum(lv, v)
        wd = w * lv / (lv + nv)
        res = res + (w - wd)
        c = c2
    return res


def residual(x, kind):
    if kind == "g06":
        return x - gblur(x, 0.6)
    if kind == "g10":
        return x - gblur(x, 1.0)
    if kind == "med3":
        return x - medf(x, 3)
    if kind.startswith("wav"):
        import wavelet
        return wavelet.residual(x, float(kind[3:]))
    if kind == "wie3":
        return x - wiener(x, 3)
    if kind == "wie5":
        return x - wiener(x, 5)
    if kind == "wie7":
        return x - wiener(x, 7)
    if kind.startswith("star"):
        return star_resid(x, int(kind[4:]) if len(kind) > 4 else 4)
    raise ValueError(kind)


def zm(a):
    a = a - a.mean(-1, keepdim=True)
    return a - a.mean(-2, keepdim=True)


def feats(kind, chan, whiten, clipz=0):
    W, I = [], []
    for i in range(len(paths)):
        x = X[i:i + 1].float()
        if chan == "luma":
            x = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
        m = M[i].float()
        r = residual(x, kind)[0]  # C,H,W
        lum = x[0]
        r = r * m
        if whiten:
            lv = boxf((r * r)[None], 15)[0]
            r = r / lv.clamp(min=1e-3).sqrt()
        if clipz:
            s = r[:, m > 0].std()
            r = r.clamp(-clipz * s, clipz * s)
        r = zm(r) * m
        W.append(r.half())
        I.append((lum * m).half())
    return W, I


def loo(W, I, whiten, mult=None):
    whiten = (not mult) if mult is not None else whiten
    n = len(W)
    # per-person sums
    num = {}
    den = {}
    for c in range(6):
        idx = np.where(labels == c)[0]
        num[c] = sum(W[i].float() * (I[i].float() if not whiten else 1) for i in idx)
        den[c] = sum((I[i].float() ** 2 if not whiten else torch.ones_like(I[i].float())) for i in idx)
    correct = 0
    margins = []
    for i in range(n):
        sc = []
        ks = []
        for c in range(6):
            nu = num[c] - (W[i].float() * (I[i].float() if not whiten else 1) if labels[i] == c else 0)
            de = den[c] - ((I[i].float() ** 2 if not whiten else 1) if labels[i] == c else 0)
            ks.append(zm(nu / de.clamp(min=1)))
        if DEMEAN:
            km = sum(ks) / 6
            ks = [k - km for k in ks]
        for c in range(6):
            k = ks[c]
            a = W[i].float()
            b = k * (I[i].float() if not whiten else 1)
            a = a - a.mean(); b = b - b.mean()
            sc.append(float((a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum())))
        sc = np.array(sc)
        correct += sc.argmax() == labels[i]
        o = np.sort(np.delete(sc, labels[i]))[-1]
        margins.append(sc[labels[i]] - o)
    return correct, float(np.median(margins))


import os
DEMEAN = int(os.environ.get("DEMEAN", "0"))
configs = [a.split(":") for a in sys.argv[1:]] or [["wie5", "luma", "0"], ["g06", "luma", "0"], ["wie3", "luma", "0"],
                                                   ["med3", "luma", "0"], ["wie5", "rgb", "0"], ["wie5", "luma", "1"],
                                                   ["g06", "luma", "1"], ["wie7", "luma", "0"]]
for cfg in configs:
    kind, chan, wh = cfg[:3]
    mult = int(cfg[3]) if len(cfg) > 3 else None
    clipz = float(cfg[4]) if len(cfg) > 4 else 0
    t = time.time()
    W, I = feats(kind, chan, int(wh), clipz)
    c, mg = loo(W, I, int(wh), mult)
    print(f"{kind} {chan} whiten={wh} mult={mult} clip={clipz}: LOO {c}/108 median margin {mg*1000:.2f}  ({time.time()-t:.0f}s)", flush=True)
    del W, I
    torch.cuda.empty_cache()
