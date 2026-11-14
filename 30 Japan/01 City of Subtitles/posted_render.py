"""Render the posted-frame mean residual of a photographer's anonymous photos with its sensor pattern removed
(each photo minus its fitted, correctly oriented sensor term). Anything shared in the posted frame stays."""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from render_sensor import info, d, names, dev, rot, sensor_fp, OUT

h = sys.argv[1] if len(sys.argv) > 1 else "toorisugari"
K, _, _ = sensor_fp(h)
ix = [i for i in range(len(names)) if info(i)[0] == h and names[i].startswith("anonymous")]
acc = 0
cnt = 0
for i in ix:
    w = d["W"][i].float().to(dev); li = d["I"][i].float().to(dev); m = d["M"][i].float().to(dev)
    k = rot(K) if info(i)[1] else K
    s = k * li
    beta = (w * s).sum() / (s * s).sum()
    acc = acc + (w - beta * s)
    cnt = cnt + m
P = (acc / cnt.clamp(min=1)).mean(0) * cnt.sqrt()
for b in (4, 8, 16):
    a = F.avg_pool2d(P[None, None], b, b)[0, 0]
    a = (a - a.median()) / a.std()
    Image.fromarray(np.clip(a.cpu().numpy() * 45 + 128, 0, 255).astype(np.uint8)).resize((1280, 960), Image.NEAREST).save(OUT / f"posted_{h}_b{b}.png")
e = F.avg_pool2d((P * P)[None, None], 16, 16)[0, 0]
e = (e - e.median()) / e.std()
Image.fromarray(np.clip(e.cpu().numpy() * 45 + 128, 0, 255).astype(np.uint8)).resize((1280, 960), Image.NEAREST).save(OUT / f"posted_{h}_energy16.png")
print("done", len(ix))
