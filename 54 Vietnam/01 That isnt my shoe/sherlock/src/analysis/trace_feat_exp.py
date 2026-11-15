"""Try richer folded-profile features for the provenance detector (grouped CV vs natural-image null)."""
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import io, os, sys, csv
import numpy as np
import torch
from PIL import Image
import skimage.data as sd
from sklearn.datasets import load_sample_images
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, f'{SRC}')
from trace_torch import fold_matrix
torch.set_num_threads(1)
D = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(D, 'train-output.csv'))))
M16 = fold_matrix(256, 16)
def zn(f): return (f - f.mean(1, keepdim=True)) / (f.std(1, unbiased=False, keepdim=True) + 1e-9)
def F_all(x01):
    g = (x01[:, 0] * 0.299 + x01[:, 1] * 0.587 + x01[:, 2] * 0.114) * 255.0
    out = {}
    d2x = (g[:, :, 2:] - 2 * g[:, :, 1:-1] + g[:, :, :-2]); d2y = (g[:, 2:, :] - 2 * g[:, 1:-1, :] + g[:, :-2, :])
    out['d2abs'] = torch.cat([zn(d2x.abs().mean(1) @ M16), zn(d2y.abs().mean(2) @ M16)], 1)
    out['d2sq'] = torch.cat([zn((d2x ** 2).mean(1) @ M16), zn((d2y ** 2).mean(2) @ M16)], 1)
    d1x = (g[:, :, 1:] - g[:, :, :-1]).abs().mean(1)[:, :254]; d1y = (g[:, 1:, :] - g[:, :-1, :]).abs().mean(2)[:, :254]
    out['d1abs'] = torch.cat([zn(d1x @ M16), zn(d1y @ M16)], 1)
    # per-row normalised |d2| (each row/col weighted equally)
    rx = d2x.abs() / (d2x.abs().mean(2, keepdim=True) + 1.0); ry = d2y.abs() / (d2y.abs().mean(1, keepdim=True) + 1.0)
    out['d2rown'] = torch.cat([zn(rx.mean(1) @ M16), zn(ry.mean(2) @ M16)], 1)
    return out
def feats(arrs):
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float() / 255.0
    res = {}
    for i in range(0, len(x), 64):
        o = F_all(x[i:i + 64])
        for k, v in o.items(): res.setdefault(k, []).append(v)
    return {k: torch.cat(v).numpy() for k, v in res.items()}
gen = [np.asarray(Image.open(os.path.join(D, 'train-input', r['filename'])).convert('RGB')) for r in rows]
srcs = []
for nm in ['astronaut', 'coffee', 'chelsea', 'rocket', 'cat', 'immunohistochemistry', 'hubble_deep_field', 'retina', 'camera', 'coins', 'horse', 'page', 'text', 'moon', 'brick', 'grass', 'gravel', 'colorwheel', 'lily', 'kidney', 'skin', 'eagle', 'logo']:
    try:
        x = np.asarray(getattr(sd, nm)())
        if isinstance(x, tuple): x = x[0]
        if x.dtype == bool: x = x.astype(np.uint8) * 255
        if x.dtype != np.uint8: x = (255 * (x - x.min()) / (x.max() - x.min() + 1e-9)).astype(np.uint8)
        if x.ndim == 3 and x.shape[-1] not in (3, 4): x = x[..., 0]
        if x.ndim == 2: x = np.stack([x] * 3, -1)
        if min(x.shape[:2]) >= 200: srcs.append(Image.fromarray(x[..., :3]))
    except Exception: pass
for x in load_sample_images().images: srcs.append(Image.fromarray(x))
rng = np.random.RandomState(0); null, grp = [], []
FILTS = [Image.BICUBIC, Image.BILINEAR, Image.LANCZOS, Image.HAMMING, Image.BOX]
for si, im in enumerate(srcs):
    W, H = im.size
    for t in range(40):
        s = rng.randint(max(96, int(min(W, H) * 0.3)), min(W, H) + 1)
        l, tp = rng.randint(0, W - s + 1), rng.randint(0, H - s + 1)
        c = im.crop((l, tp, l + s, tp + s))
        if rng.rand() < 0.5:
            b = io.BytesIO(); c.save(b, 'JPEG', quality=int(rng.randint(70, 96))); b.seek(0); c = Image.open(b).convert('RGB')
        if rng.rand() < 0.4:
            mid = int(rng.choice([150, 180, 200, 210, 224, 230, 250, 300, 320, 400, 480, 512, 640])); c = c.resize((mid, mid), FILTS[rng.randint(5)])
        c = c.resize((256, 256), FILTS[rng.randint(3)])
        b = io.BytesIO(); c.save(b, 'JPEG', quality=95); b.seek(0)
        null.append(np.asarray(Image.open(b).convert('RGB'))); grp.append(si)
grp = np.array(grp)
FG, FN = feats(gen), feats(null)
r2 = np.random.RandomState(1); fg = r2.randint(0, 5, len(gen)); sf = r2.permutation(len(srcs)) % 5; folds = np.r_[fg, sf[grp]]
y = np.r_[np.ones(len(gen)), np.zeros(len(null))]
for combo in (['d2abs'], ['d2abs', 'd2sq'], ['d2abs', 'd1abs'], ['d2abs', 'd2rown'], ['d2abs', 'd2sq', 'd1abs', 'd2rown']):
    X = np.r_[np.concatenate([FG[k] for k in combo], 1), np.concatenate([FN[k] for k in combo], 1)]
    for C in (0.03, 0.1, 0.3):
        oof = np.zeros(len(y))
        for f in range(5):
            tr, te = folds != f, folds == f
            oof[te] = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr]).decision_function(X[te])
        g, n = oof[y == 1], oof[y == 0]
        print(f'{"+".join(combo):28s} C={C:<5} AUROC={roc_auc_score(y, oof):.4f} ' + ' '.join(f'keep{int(k*100)}:{(n < np.quantile(g, 1-k)).mean():.3f}' for k in (0.98, 0.97, 0.95, 0.9)), flush=True)
