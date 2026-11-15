"""Provenance detector v2: 128-d phase-folded features, natural-image nulls + content-matched nulls
(genuine shoes re-sampled by small rotations / zooms / sub-pixel shifts, which break the aligned 240->256 trace)."""
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_torch import fold_matrix, trace_features
torch.set_num_threads(1)
D, OUT = sys.argv[1], sys.argv[2]
C = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
rows = list(csv.DictReader(open(os.path.join(D, 'train-output.csv'))))
M = fold_matrix()
def feats(arrs):
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float() / 255.0
    return torch.cat([trace_features(x[i:i + 64], M) for i in range(0, len(x), 64)]).numpy()
def jpeg(im, q=95):
    b = io.BytesIO(); im.save(b, 'JPEG', quality=q); b.seek(0); return Image.open(b).convert('RGB')
gen_im = [Image.open(os.path.join(D, 'train-input', r['filename'])).convert('RGB') for r in rows]
gen = [np.asarray(im) for im in gen_im]
# --- natural nulls
srcs = []
for nm in ['astronaut', 'coffee', 'chelsea', 'rocket', 'cat', 'immunohistochemistry', 'hubble_deep_field', 'retina', 'camera', 'coins', 'horse', 'page', 'text', 'moon', 'brick', 'grass', 'gravel', 'colorwheel', 'lily', 'kidney', 'skin', 'eagle', 'logo']:
    try:
        x = np.asarray(getattr(sd, nm)())
        if x.dtype == bool: x = x.astype(np.uint8) * 255
        if x.dtype != np.uint8: x = (255 * (x - x.min()) / (x.max() - x.min() + 1e-9)).astype(np.uint8)
        if x.ndim == 3 and x.shape[-1] not in (3, 4): x = x[..., 0]
        if x.ndim == 2: x = np.stack([x] * 3, -1)
        if min(x.shape[:2]) >= 200: srcs.append(Image.fromarray(x[..., :3]))
    except Exception: pass
for x in load_sample_images().images: srcs.append(Image.fromarray(x))
rng = np.random.RandomState(0)
FILTS = [Image.BICUBIC, Image.BILINEAR, Image.LANCZOS, Image.HAMMING, Image.BOX]
nat, nat_grp = [], []
for si, im in enumerate(srcs):
    W, H = im.size
    for t in range(40):
        s = rng.randint(max(96, int(min(W, H) * 0.3)), min(W, H) + 1)
        l, tp = rng.randint(0, W - s + 1), rng.randint(0, H - s + 1)
        c = im.crop((l, tp, l + s, tp + s))
        if rng.rand() < 0.5: c = jpeg(c, int(rng.randint(70, 96)))
        if rng.rand() < 0.4:
            mid = int(rng.choice([150, 180, 200, 210, 224, 230, 250, 300, 320, 400, 480, 512, 640])); c = c.resize((mid, mid), FILTS[rng.randint(5)])
        nat.append(np.asarray(jpeg(c.resize((256, 256), FILTS[rng.randint(3)])))); nat_grp.append(si)
nat_grp = np.array(nat_grp)
# --- content-matched nulls (one per genuine image)
cm = []
for i, im in enumerate(gen_im):
    k = rng.randint(3)
    if k == 0:   # small rotation
        ang = float(rng.uniform(1.0, 4.0)) * (1 if rng.rand() < 0.5 else -1)
        o = im.rotate(ang, resample=Image.BICUBIC, fillcolor=tuple(int(v) for v in np.asarray(im)[0, 0]))
    elif k == 1:  # zoom-in + crop
        z = float(rng.uniform(1.04, 1.3)); n = int(round(256 * z))
        o = im.resize((n, n), FILTS[rng.randint(3)]); l, tp = rng.randint(0, n - 255), rng.randint(0, n - 255)
        o = o.crop((l, tp, l + 256, tp + 256))
    else:         # zoom-out (downscale then pad/upsample back) -> different resample history
        n = int(rng.choice([180, 200, 220, 232, 300, 360, 420]))
        o = im.resize((n, n), FILTS[rng.randint(3)]).resize((256, 256), FILTS[rng.randint(3)])
    cm.append(np.asarray(jpeg(o)))
FG, FN, FC = feats(gen), feats(nat), feats(cm)
# phase-shifted genuine images: identical content, pixel grid shifted by s px (breaks the aligned trace)
def shift(a, s):
    b = np.empty_like(a); b[:, :256 - s] = a[:, s:]; b[:, 256 - s:] = a[:, -1:]
    c = np.empty_like(b); c[:256 - s] = b[s:]; c[256 - s:] = b[-1:]
    return c
FS = {s: feats([shift(a, s) for a in gen]) for s in (3, 8)}
np.savez(f'{RUNS}/trace_feats_cache.npz', FG=FG, FN=FN, FC=FC, nat_grp=nat_grp, FS3=FS[3], FS8=FS[8])
r2 = np.random.RandomState(1)
fg = r2.randint(0, 5, len(gen)); sf = r2.permutation(len(srcs)) % 5
X = np.r_[FG, FN, FC]
y = np.r_[np.ones(len(FG)), np.zeros(len(FN)), np.zeros(len(FC))]
folds = np.r_[fg, sf[nat_grp], fg]
kind = np.r_[np.zeros(len(FG)), np.ones(len(FN)), 2 * np.ones(len(FC))]
for train_kinds, label in (((0, 1), 'natural nulls only'), ((0, 1, 2), 'natural + content nulls')):
    oof = np.zeros(len(y))
    for f in range(5):
        tr = (folds != f) & np.isin(kind, train_kinds)
        te = folds == f
        oof[te] = LogisticRegression(C=C, max_iter=8000).fit(X[tr], y[tr]).decision_function(X[te])
    g = oof[kind == 0]
    msg = f'[{label}] C={C}'
    # shifted genuine images scored by the fold model that did not see their base image
    for sh in (3, 8):
        sc = np.zeros(len(FG))
        for f in range(5):
            tr = (folds != f) & np.isin(kind, train_kinds)
            clf = LogisticRegression(C=C, max_iter=8000).fit(X[tr], y[tr])
            sc[fg == f] = clf.decision_function(FS[sh][fg == f])
        msg += f' | shift{sh}: rej@k97={(sc < np.quantile(g, 0.03)).mean():.3f}'
    for kk, nm in ((1, 'natural'), (2, 'content')):
        n = oof[kind == kk]
        au = roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(n))], np.r_[g, n])
        msg += f' | {nm}: AUROC={au:.4f} ' + ' '.join(f'k{int(q*100)}:{(n < np.quantile(g, 1-q)).mean():.3f}' for q in (0.98, 0.97, 0.95))
    print(msg, flush=True)
    if len(train_kinds) == 3:
        clf = LogisticRegression(C=C, max_iter=8000).fit(X[np.isin(kind, train_kinds)], y[np.isin(kind, train_kinds)])
        np.savez(OUT, w=clf.coef_[0].astype(np.float32), b=np.float32(clf.intercept_[0]), oof_gen=g,
                 oof_null=oof[kind == 1], oof_content=oof[kind == 2], C=C)
        print('saved', OUT)
