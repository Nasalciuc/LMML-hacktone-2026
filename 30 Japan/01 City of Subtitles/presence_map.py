"""Where in the frame is a camera's pattern present? For every photo of the group, block-wise correlation between
its residual and the leave-one-out group fingerprint; averaged over the group. A pattern laid only inside the
strokes of letters would draw those letters here.
usage: python presence_map.py block cam [cam ...]"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import feats as FE
from render_fp import members, d, dev, OUT

blk = int(sys.argv[1])
for cam in sys.argv[2:]:
    ix = members(cam)
    num = sum(d["W"][i].float().to(dev) * d["I"][i].float().to(dev) for i in ix)
    den = sum(d["I"][i].float().to(dev) ** 2 for i in ix)
    acc = 0
    for i in ix:
        w = d["W"][i].float().to(dev)
        li = d["I"][i].float().to(dev)
        k = FE.zm((num - w * li) / (den - li * li).clamp(min=1))
        s = k * li
        pool = lambda t: F.avg_pool2d(t.sum(0)[None, None], blk, blk)[0, 0]
        c = pool(w * s) / (pool(w * w) * pool(s * s)).sqrt().clamp(min=1e-9)
        acc = acc + c
    m = acc / len(ix)
    print(cam, len(ix), "mean local corr", float(m.mean()), "sd", float(m.std()))
    a = (m - m.median()) / m.std()
    Image.fromarray(np.clip(a.cpu().numpy() * 45 + 128, 0, 255).astype(np.uint8)).resize((1280, 960), Image.NEAREST).save(OUT / f"presence_{cam}_{blk}.png")
