"""Chroma views of a camera group: (1) fingerprint colour differences R-G and B-G, (2) average of the photos'
own chroma (low-pass), per group. A mark hidden in colour only cancels in luma views.
usage: python chroma_scan.py cam [cam ...]"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import prnu
from render_fp import members, d, dev, OUT, names
import feats as FE


def save(a, path):
    a = (a - a.median()) / (a.std() + 1e-12)
    Image.fromarray(np.clip(a.cpu().numpy() * 45 + 128, 0, 255).astype(np.uint8)).resize((1280, 960), Image.NEAREST).save(path)


for cam in sys.argv[1:]:
    ix = members(cam)
    num = sum(d["W"][i].float().to(dev) * d["I"][i].float().to(dev) for i in ix)
    den = sum(d["I"][i].float().to(dev) ** 2 for i in ix)
    k = FE.zm(num / den.clamp(min=1)) * den.sqrt()
    for nm, c in (("RG", k[0] - k[1]), ("BG", k[2] - k[1])):
        save(F.avg_pool2d(c[None, None], 8, 8)[0, 0], OUT / f"chroma_fp_{cam}_{nm}.png")
        save(F.avg_pool2d((c * c)[None, None], 16, 16)[0, 0], OUT / f"chroma_fpE_{cam}_{nm}.png")
    acc = 0
    cnt = 0
    for i in ix:
        h, f = names[i].split("/")
        rgb = prnu.load_rgb(prnu.DATA / ("anonymous" if h == "anonymous" else f"credited/{h}") / f)
        m = torch.from_numpy(prnu.caption_mask(rgb)).float().to(dev)
        x = torch.from_numpy(rgb).permute(2, 0, 1).to(dev)
        cr = x[0] - x[1]
        cb = x[2] - x[1]
        v = torch.stack([cr, cb])[None]
        v = prnu.gblur(v, 1.5) - prnu.gblur(v, 25.0)
        v = v[0] / (v[0][:, m > 0].std(1).view(2, 1, 1) + 1e-6)
        acc = acc + v * m
        cnt = cnt + m
    mean = acc / cnt.clamp(min=1)
    save(mean[0], OUT / f"chroma_img_{cam}_RG.png")
    save(mean[1], OUT / f"chroma_img_{cam}_BG.png")
    print(cam, len(ix), "done")
