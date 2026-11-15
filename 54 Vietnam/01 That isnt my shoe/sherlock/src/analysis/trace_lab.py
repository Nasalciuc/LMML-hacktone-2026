"""Compare per-image statistics for detecting the 240->256 resize trace (genuine train images vs natural images resized otherwise)."""
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

D = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(D, 'train-output.csv'))))
img_dir = os.path.join(D, 'train-input')
P = 16
POS = np.arange(1, 255)
MASKS = [POS % P == k for k in range(P)]

def profiles(x, kind):
    d2x = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    d2y = (x[2:, :] - 2 * x[1:-1, :] + x[:-2, :]).T
    out = []
    for d in (d2x, d2y):
        a = np.abs(d)
        if kind == 'abs_mean':
            col = a.mean(0)
        elif kind == 'abs_clip':
            c = np.percentile(a, 95) + 1e-6
            col = np.minimum(a, c).mean(0)
        elif kind == 'sq_clip':
            c = np.percentile(a, 95) + 1e-6
            col = (np.minimum(a, c) ** 2).mean(0)
        elif kind == 'abs_median':
            col = np.median(a, 0)
        elif kind == 'rownorm':
            # normalise each row's |d2| by its own mean -> reduces dominance of textured rows
            col = (a / (a.mean(1, keepdims=True) + 1e-3)).mean(0)
        out.append(np.array([col[m].mean() for m in MASKS]))
    return out

def z(v): return (v - v.mean()) / (v.std() + 1e-9)

def load_gray(p): return np.asarray(Image.open(p).convert('L'), dtype=np.float64)

gen_imgs = [load_gray(os.path.join(img_dir, r['filename'])) for r in rows]

srcs = []
for nm in ['astronaut', 'coffee', 'chelsea', 'rocket', 'cat', 'immunohistochemistry', 'hubble_deep_field', 'retina', 'motorcycle_left', 'logo', 'camera', 'coins', 'horse', 'page', 'text', 'moon', 'brick', 'grass', 'gravel', 'colorwheel']:
    try:
        a = getattr(sd, nm)()
        if isinstance(a, tuple): a = a[0]
        a = np.asarray(a)
        if a.dtype == bool: a = a.astype(np.uint8) * 255
        if a.ndim == 2: a = np.stack([a] * 3, -1)
        if a.shape[-1] == 4: a = a[..., :3]
        srcs.append(Image.fromarray(a.astype(np.uint8)))
    except Exception as e:
        pass
for a in load_sample_images().images:
    srcs.append(Image.fromarray(a))
print('null sources:', len(srcs))
rng = np.random.RandomState(0)
null_imgs = []
for im in srcs:
    W, H = im.size
    for t in range(30):
        s = rng.randint(max(64, int(min(W, H) * 0.35)), min(W, H) + 1)
        l, tp = rng.randint(0, W - s + 1), rng.randint(0, H - s + 1)
        crop = im.crop((l, tp, l + s, tp + s))
        mode = rng.choice(['direct', 'via_mid'])
        filt = [Image.BICUBIC, Image.BILINEAR, Image.LANCZOS][rng.randint(3)]
        if mode == 'via_mid':
            mid = int(rng.choice([160, 180, 200, 224, 300, 320, 400, 500]))
            crop = crop.resize((mid, mid), filt)
        out = crop.resize((256, 256), [Image.BICUBIC, Image.BILINEAR, Image.LANCZOS][rng.randint(3)])
        buf = io.BytesIO(); out.save(buf, 'JPEG', quality=95); buf.seek(0)
        null_imgs.append(np.asarray(Image.open(buf).convert('L'), dtype=np.float64))
print('null samples:', len(null_imgs))

for kind in ['abs_mean', 'abs_clip', 'sq_clip', 'abs_median', 'rownorm']:
    G = [profiles(x, kind) for x in gen_imgs]
    T = z(np.mean([z(a) + z(b) for a, b in G], 0))
    sc = lambda pr: (np.dot(z(pr[0]), T) + np.dot(z(pr[1]), T)) / (2 * P)
    g = np.array([sc(p) for p in G])
    n = np.array([sc(profiles(x, kind)) for x in null_imgs])
    from sklearn.metrics import roc_auc_score
    au = roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(n))], np.r_[g, n])
    out = []
    for keep in (0.95, 0.90, 0.85, 0.80):
        thr = np.quantile(g, 1 - keep)
        out.append(f'keep{int(keep*100)}: null-rej={(n < thr).mean():.3f}')
    print(f'{kind:10s} AUROC={au:.4f} ' + ' '.join(out))

# ---- learned linear detector on folded profiles (group CV over null sources) ----
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
def feat(x):
    a, b = profiles(x, 'abs_mean')
    return np.r_[z(a), z(b)]
Xg = np.array([feat(x) for x in gen_imgs]); Xn = np.array([feat(x) for x in null_imgs])
grp_n = np.repeat(np.arange(len(srcs)), 30)
X = np.r_[Xg, Xn]; y = np.r_[np.ones(len(Xg)), np.zeros(len(Xn))]
# folds: genuine split randomly into 5, null split by source groups into 5
rng = np.random.RandomState(1)
fg = rng.randint(0, 5, len(Xg)); src_fold = rng.permutation(len(srcs)) % 5; fn = src_fold[grp_n]
folds = np.r_[fg, fn]
for C in (0.01, 0.1, 1.0):
    pred = np.zeros(len(y))
    for f in range(5):
        tr, te = folds != f, folds == f
        clf = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
        pred[te] = clf.decision_function(X[te])
    g, n = pred[y == 1], pred[y == 0]
    au = roc_auc_score(y, pred)
    print(f'LR C={C}: AUROC={au:.4f} ' + ' '.join(f'keep{int(k*100)}: null-rej={(n < np.quantile(g, 1-k)).mean():.3f}' for k in (0.95, 0.9, 0.85)))
