import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import numpy as np, sys
runs = [f'{RUNS}/mnv3_cv{k}' for k in range(5)]
emb = sys.argv[1] if len(sys.argv) > 1 else 'emb256.npz'
tr = np.load(f'{RUNS}/trace_lr3.npz'); oof_tr = tr['oof_gen']; null_tr = tr['oof_null']
sem, trc, ok = [], [], []
for r in runs:
    d = np.load(f'{r}/{emb}')
    fb = d['fb'] / np.linalg.norm(d['fb'], axis=1, keepdims=True); fv = d['fv'] / np.linalg.norm(d['fv'], axis=1, keepdims=True)
    sem.append((fv @ fb.T).max(1)); trc.append(oof_tr[d['va']]); ok.append(d['lv'].argmax(1) == d['yv'])
sem, trc, ok = map(np.concatenate, (sem, trc, ok))
def pv(cal, s):
    c = np.sort(cal); return (np.searchsorted(c, s, side='right') + 1.0) / (len(c) + 2.0)
ps, pt = pv(sem, sem), pv(oof_tr, trc)
rng = np.random.RandomState(0)
U = rng.uniform(size=20000); Tn = rng.choice(null_tr, 20000)
# semantic p-values of LOCO unknowns (224-based runs) for a rough "semantic helps" estimate
lo = []
for k in range(3):
    d = np.load(f'{RUNS}/mnv3_loco{k}/emb.npz')
    fb = d['fb'] / np.linalg.norm(d['fb'], axis=1, keepdims=True)
    fv = d['fv'] / np.linalg.norm(d['fv'], axis=1, keepdims=True); fu = d['fu'] / np.linalg.norm(d['fu'], axis=1, keepdims=True)
    sv, su = (fv @ fb.T).max(1), (fu @ fb.T).max(1)
    lo.append(pv(sv, su))
lo = np.concatenate(lo)
print(f'OOF closed-set acc = {ok.mean():.4f}  (n={len(ok)})')
for w in (1.0, 1.5):
    stat = np.log(ps) + w * np.log(pt)
    for keep in (0.95, 0.97, 0.98, 0.985, 0.99):
        thr = np.quantile(stat, 1 - keep)
        acc = ((stat >= thr) & ok).mean()
        rej_cons = (np.log(U) + w * np.log(pv(oof_tr, Tn)) < thr).mean()
        rej_loco = (np.log(rng.choice(lo, 20000)) + w * np.log(pv(oof_tr, Tn)) < thr).mean()
        print(f'w={w} keep={keep}: known_acc={acc:.4f}  null-rej (semantic uninformative)={rej_cons:.3f}  null-rej (LOCO-like semantic)={rej_loco:.3f}  thr={thr:.3f}')
print('--- more weights / rules')
ptn = pv(oof_tr, Tn); lo_s = rng.choice(lo, 20000)
for w in (2.0, 3.0, 5.0):
    stat = np.log(ps) + w * np.log(pt)
    for keep in (0.97, 0.98, 0.985):
        thr = np.quantile(stat, 1 - keep)
        print(f'fisher w={w} keep={keep}: known_acc={((stat >= thr) & ok).mean():.4f}  rej(sem-uninf)={(np.log(U) + w*np.log(ptn) < thr).mean():.3f}  rej(LOCO-sem)={(np.log(lo_s) + w*np.log(ptn) < thr).mean():.3f}')
for at in (0.01, 0.015, 0.02, 0.03):
    for as_ in (0.0, 0.005, 0.01):
        acc_mask = (pt >= at) & (ps >= as_)
        r1 = ((ptn < at) | (U < as_)).mean(); r2 = ((ptn < at) | (lo_s < as_)).mean()
        print(f'gate p_tr>={at} & p_sem>={as_}: accepted={acc_mask.mean():.3f} known_acc={(acc_mask & ok).mean():.4f}  rej(sem-uninf)={r1:.3f}  rej(LOCO-sem)={r2:.3f}')
