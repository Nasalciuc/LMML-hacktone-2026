"""Fit the resize-trace detector (logistic regression on folded |d2| profiles).

genuine = training images (all came through a 240->256 resize)
null    = natural images resized to 256 along other paths (+JPEG q95 like the challenge files)
Outputs weights + out-of-fold scores for calibration.
"""
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import io
import os
import sys
import csv

import numpy as np
import torch
from PIL import Image
import skimage.data as sd
from sklearn.datasets import load_sample_images
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_torch import fold_matrix, gray_255, folded_profiles  # noqa: E402

D, OUT = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(os.path.join(D, 'train-output.csv'))))
M = fold_matrix()


def feats_rgb(arrs):
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float() / 255.0
    out = []
    for i in range(0, len(x), 64):
        out.append(folded_profiles(gray_255(x[i:i + 64]), M))
    return torch.cat(out).numpy()


gen = [np.asarray(Image.open(os.path.join(D, 'train-input', r['filename'])).convert('RGB')) for r in rows]

srcs = []
for nm in ['astronaut', 'coffee', 'chelsea', 'rocket', 'cat', 'immunohistochemistry', 'hubble_deep_field', 'retina',
           'motorcycle_left', 'motorcycle_right', 'logo', 'camera', 'coins', 'horse', 'page', 'text', 'moon', 'brick',
           'grass', 'gravel', 'colorwheel', 'lily', 'kidney', 'skin', 'eagle']:
    try:
        a = getattr(sd, nm)()
        if isinstance(a, tuple):
            a = a[0]
        a = np.asarray(a)
        if a.dtype == bool:
            a = a.astype(np.uint8) * 255
        if a.dtype != np.uint8:
            a = (255 * (a - a.min()) / (a.max() - a.min() + 1e-9)).astype(np.uint8)
        if a.ndim == 3 and a.shape[-1] not in (3, 4):
            a = a[..., :3] if a.shape[-1] > 4 else a[..., 0]
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        if a.shape[-1] == 4:
            a = a[..., :3]
        if min(a.shape[:2]) >= 200:
            srcs.append(Image.fromarray(a))
    except Exception:
        pass
for a in load_sample_images().images:
    srcs.append(Image.fromarray(a))
print('null sources:', len(srcs))

rng = np.random.RandomState(0)
FILTS = [Image.BICUBIC, Image.BILINEAR, Image.LANCZOS, Image.HAMMING, Image.BOX]
null, grp = [], []
PER = 40
for si, im in enumerate(srcs):
    W, H = im.size
    for t in range(PER):
        s = rng.randint(max(96, int(min(W, H) * 0.3)), min(W, H) + 1)
        l, tp = rng.randint(0, W - s + 1), rng.randint(0, H - s + 1)
        crop = im.crop((l, tp, l + s, tp + s))
        if rng.rand() < 0.5:  # source-side JPEG (web images)
            b = io.BytesIO()
            crop.save(b, 'JPEG', quality=int(rng.randint(70, 96)))
            b.seek(0)
            crop = Image.open(b).convert('RGB')
        if rng.rand() < 0.4:
            mid = int(rng.choice([150, 180, 200, 210, 224, 230, 250, 300, 320, 400, 480, 512, 640]))
            crop = crop.resize((mid, mid), FILTS[rng.randint(len(FILTS))])
        out = crop.resize((256, 256), FILTS[rng.randint(3)])
        b = io.BytesIO()
        out.save(b, 'JPEG', quality=95)
        b.seek(0)
        null.append(np.asarray(Image.open(b).convert('RGB')))
        grp.append(si)
grp = np.array(grp)
print('null samples:', len(null))

Xg, Xn = feats_rgb(gen), feats_rgb(null)
X = np.r_[Xg, Xn]
y = np.r_[np.ones(len(Xg)), np.zeros(len(Xn))]
r2 = np.random.RandomState(1)
fg = r2.randint(0, 5, len(Xg))
src_fold = r2.permutation(len(srcs)) % 5
folds = np.r_[fg, src_fold[grp]]
C = 0.1
oof = np.zeros(len(y))
for f in range(5):
    tr, te = folds != f, folds == f
    clf = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr])
    oof[te] = clf.decision_function(X[te])
g, n = oof[y == 1], oof[y == 0]
print(f'OOF AUROC={roc_auc_score(y, oof):.4f}  ' + '  '.join(
    f'keep{int(k * 100)}: null-rej={(n < np.quantile(g, 1 - k)).mean():.3f}' for k in (0.97, 0.95, 0.9, 0.85)))
clf = LogisticRegression(C=C, max_iter=5000).fit(X, y)
np.savez(OUT, w=clf.coef_[0].astype(np.float32), b=np.float32(clf.intercept_[0]), oof_gen=g, oof_null=n,
         files=np.array([r['filename'] for r in rows]))
print('saved', OUT)
