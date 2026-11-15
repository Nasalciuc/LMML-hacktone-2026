"""Final provenance LR: genuine vs natural nulls on cached 128-d features; saves weights + out-of-fold genuine scores."""
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
C = float(sys.argv[1]); OUT = sys.argv[2]
d = np.load(f'{RUNS}/trace_feats_cache.npz')
FG, FN, grp = d['FG'], d['FN'], d['nat_grp']
r2 = np.random.RandomState(1)
fg = r2.randint(0, 5, len(FG)); sf = r2.permutation(grp.max() + 1) % 5
X = np.r_[FG, FN]; y = np.r_[np.ones(len(FG)), np.zeros(len(FN))]; folds = np.r_[fg, sf[grp]]
oof = np.zeros(len(y))
for f in range(5):
    tr, te = folds != f, folds == f
    oof[te] = LogisticRegression(C=C, max_iter=8000).fit(X[tr], y[tr]).decision_function(X[te])
g, n = oof[y == 1], oof[y == 0]
print(f'C={C} OOF AUROC={roc_auc_score(y, oof):.4f} ' + ' '.join(f'keep{int(k*100)}: null-rej={(n < np.quantile(g, 1-k)).mean():.3f}' for k in (0.98, 0.97, 0.95, 0.9)))
clf = LogisticRegression(C=C, max_iter=8000).fit(X, y)
np.savez(OUT, w=clf.coef_[0].astype(np.float32), b=np.float32(clf.intercept_[0]), oof_gen=g, oof_null=n)
print('saved', OUT)
