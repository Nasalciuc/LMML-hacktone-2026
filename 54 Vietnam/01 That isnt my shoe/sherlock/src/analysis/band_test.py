"""Band-limit test: genuine images were upsampled 240->256, so they carry ~no energy above 240/256 of Nyquist."""
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
from sklearn.datasets import load_sample_images
from sklearn.metrics import roc_auc_score

D = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(D, 'train-output.csv'))))
def luma(a):
    a = a.astype(np.float64)
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
f = np.fft.fftfreq(256)
FX, FY = np.meshgrid(f, f)
M = np.maximum(np.abs(FX), np.abs(FY))
win = np.outer(np.hanning(256), np.hanning(256))
def feats(a):
    g = luma(a); g = (g - g.mean()) * win
    P = np.abs(np.fft.fft2(g)) ** 2
    hi = P[M > 0.475].mean(); mid = P[(M > 0.40) & (M <= 0.455)].mean(); low = P[(M > 0.25) & (M <= 0.40)].mean()
    return np.log(hi + 1e-6) - np.log(mid + 1e-6), np.log(hi + 1e-6) - np.log(low + 1e-6)
gen = [np.asarray(Image.open(os.path.join(D, 'train-input', r['filename'])).convert('RGB')) for r in rows]
srcs = []
for nm in ['astronaut', 'coffee', 'chelsea', 'rocket', 'cat', 'immunohistochemistry', 'hubble_deep_field', 'retina', 'camera', 'coins', 'horse', 'moon', 'brick', 'grass', 'gravel', 'colorwheel']:
    try:
        x = np.asarray(getattr(sd, nm)())
        if x.dtype != np.uint8: x = (255 * (x - x.min()) / (x.max() - x.min() + 1e-9)).astype(np.uint8)
        if x.ndim == 2: x = np.stack([x] * 3, -1)
        srcs.append(Image.fromarray(x[..., :3]))
    except Exception: pass
for x in load_sample_images().images: srcs.append(Image.fromarray(x))
rng = np.random.RandomState(0); null = []
FILTS = [Image.BICUBIC, Image.BILINEAR, Image.LANCZOS]
for im in srcs:
    W, H = im.size
    for t in range(30):
        s = rng.randint(max(96, int(min(W, H) * 0.3)), min(W, H) + 1)
        l, tp = rng.randint(0, W - s + 1), rng.randint(0, H - s + 1)
        c = im.crop((l, tp, l + s, tp + s))
        if rng.rand() < 0.5:
            b = io.BytesIO(); c.save(b, 'JPEG', quality=int(rng.randint(70, 96))); b.seek(0); c = Image.open(b).convert('RGB')
        c = c.resize((256, 256), FILTS[rng.randint(3)])
        b = io.BytesIO(); c.save(b, 'JPEG', quality=95); b.seek(0)
        null.append(np.asarray(Image.open(b).convert('RGB')))
G = np.array([feats(a) for a in gen]); N = np.array([feats(a) for a in null])
for j, nm in enumerate(['hi/mid', 'hi/low']):
    y = np.r_[np.ones(len(G)), np.zeros(len(N))]; s = -np.r_[G[:, j], N[:, j]]   # genuine = lower hi energy
    g, n = -G[:, j], -N[:, j]
    print(f'{nm}: AUROC={roc_auc_score(y, s):.4f}  ' + '  '.join(f'keep{int(k*100)}: null-rej={(n < np.quantile(g, 1-k)).mean():.3f}' for k in (0.97, 0.95, 0.9)))
np.savez(f'{RUNS}/band_feats.npz', G=G, N=N)
