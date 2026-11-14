"""Count caption text lines per photo (light text on a dark band or box) and report their position."""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import prnu

HERE = Path(__file__).resolve().parent


def caption_lines(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    dark = ndimage.uniform_filter((g < 45).astype(np.float32), 41) > 0.55
    text = (g > 140) & dark
    rows = text.sum(1) > 6
    lab, n = ndimage.label(ndimage.binary_closing(rows, np.ones(5)))
    lines = []
    for k in range(1, n + 1):
        idx = np.where(lab == k)[0]
        if 8 <= len(idx) <= 60:
            cols = np.where(text[idx].any(0))[0]
            lines.append((int(idx[0]), int(idx[-1]), int(cols[0]), int(cols[-1])))
    return lines


if __name__ == "__main__":
    who = sys.argv[1] if len(sys.argv) > 1 else "toorisugari"
    rows = [r for r in csv.DictReader(open(HERE / "attribution.csv")) if who == "all" or r["photographer"] == who]
    for r in sorted(rows, key=lambda r: r["file"]):
        L = caption_lines(prnu.DATA / "anonymous" / r["file"])
        print(r["file"], r["photographer"][:4], len(L), [(a, b - a + 1, c, e) for a, b, c, e in L])
