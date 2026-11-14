"""Per-photo map of where the photographer's sensor pattern is present (block-wise correlation of the photo's
residual with its sensor fingerprint, leave-one-out, in the photo's own orientation).
usage: python photo_presence.py <handle> [block] -> out/presence_sheet_<handle>.png (posting order)"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from render_sensor import info, d, names, dev, rot, FE, OUT

h = sys.argv[1]
blk = int(sys.argv[2]) if len(sys.argv) > 2 else 32
ix = [i for i in range(len(names)) if info(i)[0] == h]
num = 0
den = 0
for i in ix:
    w = d["W"][i].float().to(dev); li = d["I"][i].float().to(dev)
    if info(i)[1]:
        w, li = rot(w), rot(li)
    num = num + w * li
    den = den + li * li
anon = sorted([i for i in ix if names[i].startswith("anonymous")], key=lambda i: names[i])
tiles = []
maps = {}
for i in anon:
    w = d["W"][i].float().to(dev); li = d["I"][i].float().to(dev)
    r = info(i)[1]
    ws, ls = (rot(w), rot(li)) if r else (w, li)
    k = FE.zm((num - ws * ls) / (den - ls * ls).clamp(min=1))
    s = k * ls
    pool = lambda t: F.avg_pool2d(t.sum(0)[None, None], blk, blk)[0, 0]
    c = pool(ws * s) / (pool(ws * ws) * pool(s * s)).sqrt().clamp(min=1e-9)
    c = F.avg_pool2d(F.pad(c[None, None], (1, 1, 1, 1), mode="replicate"), 3, 1)[0, 0]  # light smoothing
    if r:
        c = rot(c)  # back to posted orientation
    maps[names[i]] = c.cpu().numpy()
    a = (c - c.median()) / (c.std() + 1e-9)
    im = Image.fromarray(np.clip(a.cpu().numpy() * 50 + 128, 0, 255).astype(np.uint8)).resize((240, 180), Image.NEAREST)
    ImageDraw.Draw(im).text((3, 3), names[i].split("/")[1][:7] + (" R" if r else ""), fill=255)
    tiles.append(im)
cols = 9
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("L", (cols * 242, rows * 182), 0)
for k, t in enumerate(tiles):
    sheet.paste(t, ((k % cols) * 242, (k // cols) * 182))
sheet.save(OUT / f"presence_sheet_{h}_{blk}.png")
np.savez_compressed(Path(__file__).resolve().parent / "cache" / f"presence_{h}_{blk}.npz", **{k.replace('/', '_'): v for k, v in maps.items()})
print("saved", len(tiles))
