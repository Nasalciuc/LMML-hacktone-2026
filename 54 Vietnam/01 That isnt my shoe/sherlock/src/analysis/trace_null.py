import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import io, os, sys, csv
import numpy as np
from PIL import Image
import skimage.data as sd

D = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(D, 'train-output.csv'))))
img_dir = os.path.join(D, 'train-input')
def folded(x, P=16):
    d2x = np.abs(x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]).mean(0)
    d2y = np.abs(x[2:, :] - 2 * x[1:-1, :] + x[:-2, :]).mean(1)
    pos = np.arange(1, 255)
    return (np.array([d2x[pos % P == k].mean() for k in range(P)]), np.array([d2y[pos % P == k].mean() for k in range(P)]))
def z(v): return (v - v.mean()) / (v.std() + 1e-9)
imgs = [np.asarray(Image.open(os.path.join(img_dir, r['filename'])).convert('L'), dtype=np.float64) for r in rows]
T = z(np.mean([z(a) + z(b) for a, b in map(folded, imgs)], 0))
def score(x):
    a, b = folded(x); return (np.dot(z(a), T) + np.dot(z(b), T)) / 32
gen = np.array([score(x) for x in imgs])
srcs = []
for nm in ['astronaut', 'coffee', 'chelsea', 'rocket', 'cat', 'immunohistochemistry', 'hubble_deep_field', 'retina', 'colorwheel', 'logo', 'motorcycle_left', 'motorcycle_right']:
    try:
        a = getattr(sd, nm)()
        if isinstance(a, tuple): a = a[0]
        if a.ndim == 2: a = np.stack([a] * 3, -1)
        if a.shape[-1] == 4: a = a[..., :3]
        srcs.append(Image.fromarray(a.astype(np.uint8)))
    except Exception as e:
        pass
rng = np.random.RandomState(0)
nulls = {'down_from_large': [], 'up_from_200': [], 'up_from_180': [], 'exact_240_up': []}
for im in srcs:
    W, H = im.size
    for t in range(25):
        s = rng.randint(int(min(W, H) * 0.5), min(W, H) + 1)
        l, tp = rng.randint(0, W - s + 1), rng.randint(0, H - s + 1)
        crop = im.crop((l, tp, l + s, tp + s))
        for key, mid in (('down_from_large', None), ('up_from_200', 200), ('up_from_180', 180), ('exact_240_up', 240)):
            c = crop if mid is None else crop.resize((mid, mid), Image.BICUBIC)
            out = c.resize((256, 256), Image.BILINEAR)
            buf = io.BytesIO(); out.save(buf, 'JPEG', quality=95); buf.seek(0)
            x = np.asarray(Image.open(buf).convert('L'), dtype=np.float64)
            nulls[key].append(score(x))
print('n source crops:', len(nulls['down_from_large']))
print('genuine  p5/p10/p25/p50:', np.round(np.percentile(gen, [5, 10, 25, 50]), 3))
for k, v in nulls.items():
    v = np.array(v)
    print(f'{k:16s} p50/p90/p95/p99: {np.round(np.percentile(v, [50, 90, 95, 99]), 3)}  frac >= genuine p10 ({np.percentile(gen, 10):.3f}): {(v >= np.percentile(gen, 10)).mean():.3f}')
