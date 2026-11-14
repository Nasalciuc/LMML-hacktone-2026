"""Calibrate the 'is this one of our genuine photos?' test used by main.py.

Genuine copies are simulated with what a re-upload does (JPEG re-encode 70-95, resize round trip, mirror);
counterfeits are a genuine photo with one small local change (patch from another shoe, blur, colour shift,
erased detail); strangers are the nearest *other* training photo. Prints the distance statistics so the
thresholds in main.py can sit between genuine and counterfeit."""
import csv
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "task_vietnam"
S = 128
rng = np.random.default_rng(0)


def small(im):
    return np.asarray(im.convert("RGB").resize((S, S), Image.BILINEAR, reducing_gap=None), dtype=np.float32)


def dist(a, b):
    d = np.abs(a - b).mean(2)
    blk = d.reshape(S // 8, 8, S // 8, 8).mean((1, 3))
    return float(d.mean()), float(blk.max())


def best(a, bank):
    out = []
    for b in bank:
        out.append(min(dist(a, b), dist(a, b[:, ::-1]), key=lambda t: t[1]))
    return out


rows = list(csv.DictReader(open(DATA / "train-output.csv")))
files = [DATA / "train-input" / r["filename"] for r in rows]
ims = [Image.open(f).convert("RGB") for f in files]
bank = [small(im) for im in ims]
idx = rng.choice(len(ims), 60, replace=False)


def reencode(im, q):
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=q); return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


gen, fake, other = [], [], []
for k in idx:
    im = ims[k]
    for v in range(3):
        g = reencode(im, int(rng.integers(70, 96)))
        if v == 1:
            g = g.resize((300, 300), Image.BICUBIC).resize((256, 256), Image.BICUBIC)
        if v == 2:
            g = g.transpose(Image.FLIP_LEFT_RIGHT)
        gen.append(min(best(small(g), [bank[k]]), key=lambda t: t[1]))
    for v in range(4):
        f = im.copy()
        s = int(rng.integers(20, 40)); x, y = rng.integers(40, 256 - s - 40, size=2)
        box = (int(x), int(y), int(x) + s, int(y) + s)
        if v == 0:
            src = ims[int(rng.integers(len(ims)))].crop(box); f.paste(src, box)
        elif v == 1:
            f.paste(f.crop(box).filter(ImageFilter.GaussianBlur(3)), box)
        elif v == 2:
            p = np.asarray(f.crop(box), dtype=np.float32); p[..., 0] *= 0.7; p[..., 2] *= 1.3
            f.paste(Image.fromarray(np.clip(p, 0, 255).astype(np.uint8)), box)
        else:
            p = np.asarray(f.crop(box), dtype=np.float32); p[:] = p.mean((0, 1))
            f.paste(Image.fromarray(p.astype(np.uint8)), box)
        f = reencode(f, 90)
        fake.append(min(best(small(f), [bank[k]]), key=lambda t: t[1]))
    others = [b for j, b in enumerate(bank) if j != k]
    other.append(min(best(bank[k], others), key=lambda t: t[1]))
for nm, arr in (("genuine", gen), ("counterfeit", fake), ("nearest other", other)):
    a = np.array(arr)
    print(f"{nm:14s} mean-diff pct [0,50,95,100] {np.percentile(a[:,0],[0,50,95,100]).round(2)}  "
          f"max-block pct [0,5,50,100] {np.percentile(a[:,1],[0,5,50,100]).round(2)}")
