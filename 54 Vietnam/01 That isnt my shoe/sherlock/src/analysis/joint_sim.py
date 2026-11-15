"""Simulate the joint (semantic + resize-trace) accept/reject rule on the leave-one-brand-out proxy.

Scenario A: unknowns come from another source -> their trace scores are drawn from the null distribution.
Scenario B: unknowns went through the same 240->256 pipeline -> trace scores look genuine.
"""
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import sys
import numpy as np

RUNS = sys.argv[1:4]
TR = np.load(f'{RUNS}/trace_lr3.npz')
gen_tr, null_tr = TR['oof_gen'], TR['oof_null']
rng = np.random.RandomState(0)


def ecdf(cal):
    s = np.sort(cal)
    return lambda v: (np.searchsorted(s, v, side='right') + 1) / (len(s) + 2)


def sem_scores(d, which):
    lb, fb = d['lb'], d['fb']
    def sc(l, f):
        if which == 'mls':
            return l.max(1)
        fn = f / np.linalg.norm(f, axis=1, keepdims=True)
        bn = fb / np.linalg.norm(fb, axis=1, keepdims=True)
        return (fn @ bn.T).max(1)
    return sc(d['lv'], d['fv']), sc(d['lu'], d['fu'])


def combine(us, ut, rule, w=1.0):
    if rule == 'sem_only':
        return np.log(us)
    if rule == 'trace_only':
        return np.log(ut)
    if rule == 'fisher':
        return np.log(us) + w * np.log(ut)
    if rule == 'min':
        return np.minimum(us, ut)


trace_cdf = ecdf(gen_tr)
for which in ('knn',):
    print(f'=== semantic score: {which}')
    for rule, w in (('trace_only', 1), ('fisher', 1.0), ('fisher', 2.0), ('fisher', 3.0)):
        for keep in (0.95, 0.97, 0.98):
            accA, rejA, rejB = [], [], []
            for run in RUNS:
                d = np.load(f'{run}/emb.npz')
                sv, su = sem_scores(d, which)
                yv, pv = d['yv'], d['lv'].argmax(1)
                tv = gen_tr[d['va']]
                tuA = rng.choice(null_tr, size=len(su), replace=True)
                tuB = gen_tr[d['unk']]
                n = len(sv)
                perm = rng.permutation(n)
                halves = [perm[: n // 2], perm[n // 2:]]
                for h in range(2):
                    cal, ev = halves[h], halves[1 - h]
                    sem_cdf = ecdf(sv[cal])
                    c_cal = combine(sem_cdf(sv[cal]), trace_cdf(tv[cal]), rule, w)
                    thr = np.quantile(c_cal, 1 - keep)
                    c_ev = combine(sem_cdf(sv[ev]), trace_cdf(tv[ev]), rule, w)
                    acc_k = ((c_ev >= thr) & (pv[ev] == yv[ev])).mean()
                    cA = combine(sem_cdf(su), trace_cdf(tuA), rule, w)
                    cB = combine(sem_cdf(su), trace_cdf(tuB), rule, w)
                    accA.append(acc_k)
                    rejA.append((cA < thr).mean())
                    rejB.append((cB < thr).mean())
            print(f'  {rule:10s} w={w:<3} keep={keep}: known_acc={np.mean(accA):.3f} (min {np.min(accA):.3f})  '
                  f'rej_A={np.mean(rejA):.3f} (min {np.min(rejA):.3f})  rej_B={np.mean(rejB):.3f}')
