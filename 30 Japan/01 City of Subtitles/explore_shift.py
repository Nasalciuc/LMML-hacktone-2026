"""Pairwise cross-correlation of residuals (all shifts) inside one person: is the PRNU at a fixed position?"""
import sys

import numpy as np
import torch

import prnu
from prnu import gblur

dev = "cuda"
credited, anonymous = prnu.list_photos()
h = sys.argv[1] if len(sys.argv) > 1 else "rojiura"
others = sys.argv[2] if len(sys.argv) > 2 else None


def res_of(p):
    rgb = prnu.load_rgb(p)
    m = torch.from_numpy(prnu.caption_mask(rgb)).to(dev).float()
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(dev)
    y = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
    r = (y - gblur(y, 0.6))[0, 0] * m
    r = r - r.mean(1, keepdim=True)
    r = r - r.mean(0, keepdim=True)
    r = r * m
    return r / r.std()


R = [res_of(p) for p in credited[h]]
if others:
    R2 = [res_of(p) for p in credited[others]]
else:
    R2 = R
FF = [torch.fft.rfft2(r) for r in R]
FF2 = [torch.fft.rfft2(r) for r in R2]
H, W = R[0].shape
for i in range(len(R)):
    line = []
    for j in range(len(R2)):
        if others is None and j <= i:
            continue
        xc = torch.fft.irfft2(FF[i] * torch.conj(FF2[j]), s=(H, W))
        k = int(torch.argmax(xc))
        py, px = divmod(k, W)
        py = py if py < H // 2 else py - H
        px = px if px < W // 2 else px - W
        pk = xc.flatten()[k]
        e = (xc ** 2).mean()
        pce = float(pk ** 2 / e)
        z0 = float(xc[0, 0] / xc.std())
        if pce > 40:
            line.append(f"{j}:({py},{px}) pce{pce:.0f} z0={z0:.1f}")
    print(i, credited[h][i].name, " | ".join(line), flush=True)
