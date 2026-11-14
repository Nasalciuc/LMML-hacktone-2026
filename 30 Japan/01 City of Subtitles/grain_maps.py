"""Local energy of the raw (not whitened) noise residual per photo: a grain laid in a shape would show here.
usage: python grain_maps.py out.png file1 file2 ... (anonymous file names)"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import prnu
import wavelet

dev = "cuda" if torch.cuda.is_available() else "cpu"
tiles = []
for f in sys.argv[2:]:
    rgb = prnu.load_rgb(prnu.DATA / "anonymous" / f)
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(dev)
    r = wavelet.residual(x, 3.0)[0].mean(0)
    e = F.avg_pool2d((r * r)[None, None], 8, 8)[0, 0].sqrt()
    e = torch.log(e + 0.05)
    a = (e - e.median()) / e.std()
    im = Image.fromarray(np.clip(a.cpu().numpy() * 50 + 128, 0, 255).astype(np.uint8)).resize((320, 240))
    ImageDraw.Draw(im).text((3, 3), f, fill=255)
    tiles.append(im)
cols = 4
s = Image.new("L", (cols * 322, ((len(tiles) + cols - 1) // cols) * 242))
for k, t in enumerate(tiles):
    s.paste(t, ((k % cols) * 322, (k // cols) * 242))
s.save(sys.argv[1])
