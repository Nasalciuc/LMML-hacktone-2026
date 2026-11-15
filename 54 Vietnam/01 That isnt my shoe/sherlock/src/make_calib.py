"""Out-of-fold calibration for the accept/reject rule.

usage: python -I make_calib.py --runs RUN1 [RUN2 ...] --keep 0.95 --w_trace 1.0 --out calib.npz
Each run dir has emb.npz with fb (train-bank embeddings), fv (val embeddings), va (val indices).
"""
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--runs', nargs='+', required=True)
ap.add_argument('--trace', default=f'{RUNS}/trace_lr3.npz')
ap.add_argument('--keep', type=float, default=0.95)
ap.add_argument('--w_trace', type=float, default=1.0)
ap.add_argument('--out', required=True)
ap.add_argument('--emb', default='emb.npz')
a = ap.parse_args()

tr = np.load(a.trace)
oof_tr_all = tr['oof_gen']
sem, trc, ok = [], [], []
for run in a.runs:
    d = np.load(f'{run}/{a.emb}')
    fb = d['fb'] / np.linalg.norm(d['fb'], axis=1, keepdims=True)
    fv = d['fv'] / np.linalg.norm(d['fv'], axis=1, keepdims=True)
    sem.append((fv @ fb.T).max(1))
    trc.append(oof_tr_all[d['va']])
    ok.append(d['lv'].argmax(1) == d['yv'])
sem, trc, ok = np.concatenate(sem), np.concatenate(trc), np.concatenate(ok)
n = len(sem)


def pv(cal, s):
    c = np.sort(cal)
    return (np.searchsorted(c, s, side='right') + 1.0) / (len(c) + 2.0)


p_sem = pv(sem, sem)
p_tr = pv(oof_tr_all, trc)          # trace ECDF uses all genuine OOF scores (same as exported tr_cal)
stat = np.log(p_sem) + a.w_trace * np.log(p_tr)
thr_joint = np.quantile(stat, 1 - a.keep)
thr_sem = np.quantile(np.log(p_sem), 1 - a.keep)
acc_mask = stat >= thr_joint
print(f'calibration n={n}  closed-set acc={ok.mean():.4f}')
print(f'keep target {a.keep}: accepted={acc_mask.mean():.3f}  known-acc (accepted & correct)={(acc_mask & ok).mean():.4f}')
print(f'thr_joint={thr_joint:.4f}  thr_sem={thr_sem:.4f}  sem-only known-acc={((np.log(p_sem) >= thr_sem) & ok).mean():.4f}')
np.savez(a.out, sem_cal=sem.astype(np.float32), tr_cal=oof_tr_all.astype(np.float32), thr_joint=np.float32(thr_joint),
         thr_sem=np.float32(thr_sem), w_trace=np.float32(a.w_trace), keep=np.float32(a.keep))
print('saved', a.out)
