"""Correlate each credited photo with its owner's leave-one-out fingerprint over all shifts; where is the peak?"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

import prnu

dev = "cuda"
credited, anonymous = prnu.list_photos()
handles = list(credited)


def boxf(x, k):
    return F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="reflect"), k, 1)


def wiener(x, k=3):
    mu = boxf(x, k)
    var = boxf(x * x, k) - mu * mu
    nv = var.mean(dim=(2, 3), keepdim=True)
    return mu + (var - nv).clamp(min=0) / var.clamp(min=1e-6) * (x - mu)


def zm(a):
    a = a - a.mean(-1, keepdim=True)
    return a - a.mean(-2, keepdim=True)


def feat(p):
    rgb = prnu.load_rgb(p)
    m = torch.from_numpy(prnu.caption_mask(rgb)).to(dev).float()
    x = torch.from_numpy(rgb).permute(2, 0, 1).to(dev)
    y = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(3, 1, 1)).sum(0)[None, None]
    r = (y - wiener(y, 3))[0, 0] * m
    lv = boxf((r * r)[None, None], 15)[0, 0]
    r = zm(r / lv.clamp(min=1e-3).sqrt()) * m
    return r, y[0, 0] * m


hs = sys.argv[1:] or handles
for h in hs:
    fs = [feat(p) for p in credited[h]]
    for i, (w, lum) in enumerate(fs):
        num = sum(fs[j][0] * fs[j][1] for j in range(len(fs)) if j != i)
        den = sum(fs[j][1] ** 2 for j in range(len(fs)) if j != i)
        k = zm(num / den.clamp(min=1))
        a = w
        b = k * lum  # note: lum of the test image at zero shift (approximation for shifted case)
        b2 = k
        H, W = a.shape
        for name, bb in (("KI", b), ("K", b2 * lum.mean())):
            xc = torch.fft.irfft2(torch.fft.rfft2(a) * torch.conj(torch.fft.rfft2(bb)), s=(H, W))
            # PCE at zero
            e = (xc ** 2).mean()
            idx = int(torch.argmax(xc))
            py, px = divmod(idx, W)
            py = py if py < H // 2 else py - H
            px = px if px < W // 2 else px - W
            print(f"{h[:6]} {i:2d} {credited[h][i].stem} {name}: pce0 {float(xc[0,0]**2/e):7.1f} sign {float(torch.sign(xc[0,0])):+.0f} | peak ({py:4d},{px:4d}) pce {float(xc.flatten()[idx]**2/e):7.1f}")
