"""Compare A (224-trained, 256 inference), B (256-trained) and the A+B ensemble on shared CV folds; known-acc vs keep."""
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import sys
import numpy as np

folds = [int(v) for v in sys.argv[1].split(',')]
tr = np.load(f'{RUNS}/trace_lr3.npz')
oof_tr = tr['oof_gen']


def sm(l):
    z = np.exp(l - l.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def knn(d):
    fb = d['fb'] / np.linalg.norm(d['fb'], axis=1, keepdims=True)
    fv = d['fv'] / np.linalg.norm(d['fv'], axis=1, keepdims=True)
    return (fv @ fb.T).max(1)


rows = {k: [] for k in ('A', 'B', 'AB')}
sems = {k: [] for k in ('A', 'B', 'AB')}
ys, trs = [], []
for k in folds:
    a = np.load(f'{RUNS}/mnv3_cv{k}/emb256.npz')
    b = np.load(f'{RUNS}/m256_cv{k}/emb.npz')
    assert (a['va'] == b['va']).all(), 'fold mismatch'
    pa, pb = sm(a['lv']), sm(b['lv'])
    rows['A'].append(pa)
    rows['B'].append(pb)
    rows['AB'].append((pa + pb) / 2)
    sems['A'].append(knn(a))
    sems['B'].append(knn(b))
    sems['AB'].append((knn(a) + knn(b)) / 2)
    ys.append(a['yv'])
    trs.append(oof_tr[a['va']])
y = np.concatenate(ys)
t = np.concatenate(trs)


def pv(cal, s):
    c = np.sort(cal)
    return (np.searchsorted(c, s, side='right') + 1.0) / (len(c) + 2.0)


for k in rows:
    p = np.concatenate(rows[k])
    ok = p.argmax(1) == y
    s = np.concatenate(sems[k])
    stat = np.log(pv(s, s)) + 2.0 * np.log(pv(oof_tr, t))
    out = []
    for keep in (0.97, 0.98, 0.985):
        thr = np.quantile(stat, 1 - keep)
        out.append(f'keep{keep}: known_acc={((stat >= thr) & ok).mean():.4f}')
    print(f'{k:3s} closed-set acc={ok.mean():.4f} (n={len(y)})  ' + '  '.join(out))
