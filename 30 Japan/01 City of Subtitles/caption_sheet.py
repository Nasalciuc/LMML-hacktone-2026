"""Contact sheet of the caption blocks of a list of anonymous photos (to read them side by side)."""
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import prnu


def blocks(path):
    im = Image.open(path)
    g = np.asarray(im.convert("L"), dtype=np.float32)
    dark = ndimage.uniform_filter((g < 45).astype(np.float32), 41) > 0.55
    text = (g > 140) & dark
    text = ndimage.binary_opening(text, np.ones((2, 2)))
    lab, n = ndimage.label(ndimage.binary_dilation(text, np.ones((40, 60))))
    out = []
    for k in range(1, n + 1):
        ys, xs = np.where((lab == k) & text)
        if len(ys) < 150:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        if y1 - y0 < 12:
            continue
        out.append(im.crop((max(0, x0 - 6), max(0, y0 - 6), min(1280, x1 + 6), min(960, y1 + 6))))
    return out


files = sys.argv[2:]
pieces = []
for f in files:
    bl = blocks(prnu.DATA / "anonymous" / f)
    for b in bl:
        if b.width > 900:
            b = b.resize((900, int(b.height * 900 / b.width)))
        pieces.append((f, b))
W = 1000
H = sum(p[1].height + 8 for p in pieces)
s = Image.new("RGB", (W, H), (40, 40, 40))
y = 0
for f, b in pieces:
    s.paste(b, (95, y))
    ImageDraw.Draw(s).text((2, y + 2), f[:7], fill=(255, 255, 0))
    y += b.height + 8
s.save(sys.argv[1])
print(len(pieces), "blocks", H)
