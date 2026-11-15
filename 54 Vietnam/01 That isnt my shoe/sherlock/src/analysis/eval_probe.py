"""Linear-probe evaluation on frozen features.

Closed-set: 5-fold stratified CV accuracy (logistic regression on L2-normalised features,
train on orig+flip, test on mean of orig/flip).
Open-set proxy (leave-one-class-out): train on 2 brands, held-out brand = unknown.
Reports AUROC and the rejection rate of unknowns at thresholds that keep 95% / 90% of knowns.
"""
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

CLASSES = ['nike', 'adidas', 'converse']


def l2n(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def fit_lr(Xtr, ytr, C):
    clf = LogisticRegression(C=C, max_iter=3000)
    clf.fit(Xtr, ytr)
    return clf


def scores_all(clf, Xtr_feats, Xte):
    logits = clf.decision_function(Xte)
    if logits.ndim == 1:  # binary -> make 2 columns
        logits = np.stack([-logits / 2, logits / 2], 1)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    sim = Xte @ Xtr_feats.T
    s = {}
    s['msp'] = p.max(1)
    s['maxlogit'] = logits.max(1)
    srt = -np.sort(-sim, axis=1)
    s['knn1'] = srt[:, 0]
    s['knn5'] = srt[:, 4]
    s['knn10'] = srt[:, 9]
    return s, logits


def reject_rate_at(known_s, unk_s, keep):
    thr = np.quantile(known_s, 1 - keep)
    return float((unk_s < thr).mean())


def main():
    feat_dir = sys.argv[1]
    names = sys.argv[2:]
    labels = np.load(os.path.join(feat_dir, 'labels.npy'))
    y = np.array([CLASSES.index(l) for l in labels])
    for name in names:
        d = np.load(os.path.join(feat_dir, f'{name}.npz'))
        Xo, Xf = l2n(d['orig']), l2n(d['flip'])
        Xm = l2n(Xo + Xf)
        res = {}
        # ---------- closed set ----------
        for C in (0.5, 2.0, 8.0):
            skf = StratifiedKFold(5, shuffle=True, random_state=0)
            correct = 0
            for tr, te in skf.split(Xm, y):
                clf = fit_lr(np.concatenate([Xo[tr], Xf[tr]]), np.concatenate([y[tr], y[tr]]), C)
                correct += (clf.predict(Xm[te]) == y[te]).sum()
            res[f'acc_C{C}'] = correct / len(y)
        bestC = max((0.5, 2.0, 8.0), key=lambda c: res[f'acc_C{c}'])
        # ---------- open set proxy (leave one class out) ----------
        osr = {k: {'auroc': [], 'rej95': [], 'rej90': []} for k in ('msp', 'maxlogit', 'knn1', 'knn5', 'knn10')}
        for held in range(3):
            kn = np.where(y != held)[0]
            un = np.where(y == held)[0]
            skf = StratifiedKFold(5, shuffle=True, random_state=1)
            ks = {k: [] for k in osr}
            us = {k: [] for k in osr}
            for tr, te in skf.split(kn, y[kn]):
                tri, tei = kn[tr], kn[te]
                clf = fit_lr(np.concatenate([Xo[tri], Xf[tri]]), np.concatenate([y[tri], y[tri]]), bestC)
                bank = np.concatenate([Xo[tri], Xf[tri]])
                s_k, _ = scores_all(clf, bank, Xm[tei])
                s_u, _ = scores_all(clf, bank, Xm[un])
                for k in osr:
                    ks[k].append(s_k[k])
                    us[k].append(s_u[k])
            for k in osr:
                kk = np.concatenate(ks[k])
                uu = np.concatenate(us[k])  # each unknown scored 5x (once per fold model)
                yy = np.concatenate([np.ones_like(kk), np.zeros_like(uu)])
                osr[k]['auroc'].append(roc_auc_score(yy, np.concatenate([kk, uu])))
                osr[k]['rej95'].append(reject_rate_at(kk, uu, 0.95))
                osr[k]['rej90'].append(reject_rate_at(kk, uu, 0.90))
        line = f'{name:15s} acc ' + ' '.join(f'C{c}={res[f"acc_C{c}"]:.3f}' for c in (0.5, 2.0, 8.0))
        print(line)
        for k in osr:
            a = np.array(osr[k]['auroc'])
            print(f'    {k:9s} AUROC per-held={np.round(a, 3)} mean={a.mean():.3f} '
                  f'rej@keep95={np.mean(osr[k]["rej95"]):.3f} rej@keep90={np.mean(osr[k]["rej90"]):.3f}')


if __name__ == '__main__':
    main()
