"""Low-frequency gain pattern of a sensor (flat-field style): mean over a photographer's photos, in sensor
orientation, of log(luma) - log(blur(luma)). A shape written into the sensor gain (not into the fine noise)
would survive here while the scenes average out.
usage: python flatfield.py sigma handle [handle ...] -> out/flat_<handle>_s<sigma>.png"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import prnu
from render_sensor import info, names, dev

HERE = Path(__file__).resolve().parent
sig = float(sys.argv[1])
for h in sys.argv[2:]:
    acc = 0
    cnt = 0
    for i, n in enumerate(names):
        hh, r = info(i)
        if hh != h:
            continue
        folder, f = n.split("/")
        rgb = prnu.load_rgb(prnu.DATA / ("anonymous" if folder == "anonymous" else f"credited/{folder}") / f)
        m = torch.from_numpy(prnu.caption_mask(rgb) & (rgb.mean(2) > 20)).to(dev).float()
        y = torch.from_numpy(rgb).to(dev).mean(2)[None, None]
        ly = torch.log(y + 1.0)
        hp = (ly - prnu.gblur(ly, sig))[0, 0]
        hp = hp - (hp * m).sum() / m.sum()
        if r:
            hp, m = torch.flip(hp, (0, 1)), torch.flip(m, (0, 1))
        acc = acc + hp * m
        cnt = cnt + m
    a = acc / cnt.clamp(min=1)
    a = a - a.mean()
    a = prnu.gblur(a[None, None], 2.0)[0, 0]
    v = a / a.std()
    Image.fromarray(np.clip(v.cpu().numpy() * 40 + 128, 0, 255).astype(np.uint8)).save(HERE / "out" / f"flat_{h}_s{int(sig)}.png")
    print(h, "done")
