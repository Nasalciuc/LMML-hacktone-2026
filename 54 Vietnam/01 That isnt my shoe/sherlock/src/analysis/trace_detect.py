import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import os, sys, csv
import numpy as np
from PIL import Image

D = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(D, 'train-output.csv'))))
img_dir = os.path.join(D, 'train-input')

def folded(x, P=16):
    d2x = np.abs(x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]).mean(0)   # len 254, position j+1
    d2y = np.abs(x[2:, :] - 2 * x[1:-1, :] + x[:-2, :]).mean(1)
    pos = np.arange(1, 255)
    fx = np.array([d2x[pos % P == k].mean() for k in range(P)])
    fy = np.array([d2y[pos % P == k].mean() for k in range(P)])
    return fx, fy

def z(v):
    return (v - v.mean()) / (v.std() + 1e-9)

imgs = [np.asarray(Image.open(os.path.join(img_dir, r['filename'])).convert('L'), dtype=np.float64) for r in rows]
F = [folded(x) for x in imgs]
T = np.mean([z(fx) + z(fy) for fx, fy in F], 0); T = z(T)
print('template (phase 0..15):', np.round(T, 2))
corr = np.array([(np.dot(z(fx), T) + np.dot(z(fy), T)) / (2 * 16) for fx, fy in F])
print('genuine corr percentiles 1/5/10/25/50:', np.round(np.percentile(corr, [1, 5, 10, 25, 50]), 3))

def reproc(x, mode):
    im = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
    if mode == 'down_up_300':      # pretend a large source downscaled to 256
        im = im.resize((300, 300), Image.BICUBIC).resize((256, 256), Image.BILINEAR)
    elif mode == 'from_200':       # pretend a 200px source upscaled to 256
        im = im.resize((200, 200), Image.BILINEAR).resize((256, 256), Image.BILINEAR)
    elif mode == 'from_224':
        im = im.resize((224, 224), Image.BILINEAR).resize((256, 256), Image.BILINEAR)
    elif mode == 'from_240_bicubic':
        im = im.resize((240, 240), Image.BILINEAR).resize((256, 256), Image.BICUBIC)
    elif mode == 'from_240_bilinear':
        im = im.resize((240, 240), Image.BILINEAR).resize((256, 256), Image.BILINEAR)
    elif mode == 'from_240_lanczos':
        im = im.resize((240, 240), Image.BILINEAR).resize((256, 256), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64)

for mode in ['down_up_300', 'from_200', 'from_224', 'from_240_bilinear', 'from_240_bicubic', 'from_240_lanczos']:
    c = []
    for x in imgs[:200]:
        fx, fy = folded(reproc(x, mode))
        c.append((np.dot(z(fx), T) + np.dot(z(fy), T)) / 32)
    c = np.array(c)
    thr = np.percentile(corr, 5)
    print(f'{mode:18s} corr pct 50/90/99: {np.round(np.percentile(c, [50, 90, 99]), 3)}  frac >= genuine-p5 ({thr:.3f}): {(c >= thr).mean():.3f}')
