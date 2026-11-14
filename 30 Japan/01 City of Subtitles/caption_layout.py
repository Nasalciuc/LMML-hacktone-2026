"""Caption layout per photo: where the black band is (top / bottom / none) and how many text lines it holds."""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import prnu

HERE = Path(__file__).resolve().parent


def layout(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    dark = (g < 22).mean(1) > 0.6
    dark = ndimage.binary_closing(dark, np.ones(25))
    lab, n = ndimage.label(dark)
    top = bot = 0
    for k in range(1, n + 1):
        idx = np.where(lab == k)[0]
        if len(idx) < 20:
            continue
        if idx[0] == 0:
            top = len(idx)
        elif idx[-1] == len(dark) - 1:
            bot = len(idx)
    # text lines: rows with bright pixels inside the band(s)
    def lines(r0, r1):
        if r1 <= r0:
            return 0
        txt = (g[r0:r1] > 150).sum(1) > 3
        l, m = ndimage.label(ndimage.binary_closing(txt, np.ones(3)))
        return sum(1 for k in range(1, m + 1) if (l == k).sum() >= 8)
    return top, bot, lines(0, top), lines(len(dark) - bot, len(dark))


if __name__ == "__main__":
    who = sys.argv[1] if len(sys.argv) > 1 else "toorisugari"
    rows = [r for r in csv.DictReader(open(HERE / "attribution.csv")) if r["photographer"] == who]
    for r in sorted(rows, key=lambda r: r["file"]):
        t, b, lt, lb = layout(prnu.DATA / "anonymous" / r["file"])
        pos = "top" if t and not b else ("bottom" if b and not t else ("both" if t and b else "overlay"))
        print(r["file"], r["rotated180"], pos, t, b, lt, lb)
