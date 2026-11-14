"""For every photo of a photographer: full cross-correlation of its residual with the (oriented, leave-one-out)
sensor fingerprint. Peak position and PCE; a photo cropped/shifted off the sensor grid would peak elsewhere."""
import sys

import numpy as np
import torch

from render_sensor import info, d, names, dev, rot, FE

h = sys.argv[1] if len(sys.argv) > 1 else "toorisugari"
ix = [i for i in range(len(names)) if info(i)[0] == h]
num = 0; den = 0
for i in ix:
    w = d["W"][i].float().to(dev); li = d["I"][i].float().to(dev)
    if info(i)[1]:
        w, li = rot(w), rot(li)
    num = num + w * li; den = den + li * li
for i in ix:
    w = d["W"][i].float().to(dev); li = d["I"][i].float().to(dev)
    r = info(i)[1]
    ws, ls = (rot(w), rot(li)) if r else (w, li)
    k = FE.zm((num - ws * ls) / (den - ls * ls).clamp(min=1))
    a = ws.sum(0); b = (k * ls).sum(0)
    a = a - a.mean(); b = b - b.mean()
    xc = torch.fft.irfft2(torch.fft.rfft2(a) * torch.conj(torch.fft.rfft2(b)), s=a.shape)
    e = (xc ** 2).mean()
    j = int(torch.argmax(xc)); H, W = xc.shape
    py, px = divmod(j, W); py = (py + H // 2) % H - H // 2; px = (px + W // 2) % W - W // 2
    print(f"{names[i]:28s} rot{r} pce0 {float(xc[0,0]**2/e):8.1f} peak ({py:4d},{px:4d}) pce {float(xc.flatten()[j]**2/e):8.1f}", flush=True)
