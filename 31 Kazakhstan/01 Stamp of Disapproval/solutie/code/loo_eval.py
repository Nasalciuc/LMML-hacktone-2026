"""Leave-one-exhibit-out validation: train on 29 exhibits, localise the altered region of
the held-out one (argmax of the smoothed probability inside the stamp).  A hit means the
peak falls inside the (slightly dilated) ground-truth mask."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from sklearn.ensemble import HistGradientBoostingClassifier

ap = argparse.ArgumentParser()
ap.add_argument('--cache', required=True)
ap.add_argument('--drop', default='', help='comma separated substrings of feature names to drop')
ap.add_argument('--iters', type=int, default=100)
args = ap.parse_args()

D = []
for i in range(1, 31):
    d = np.load(f'{args.cache}/ex{i:02d}.npz')
    nm = list(d['names'])
    keep = [k for k, n in enumerate(nm) if not (args.drop and any(s in n for s in args.drop.split(',')))]
    D.append(dict(X=d['X'][:, :, keep].astype(np.float32), mask=d['mask'], stamp=d['stamp'], dist=d['dist'].astype(np.float32)))
print('features:', len(keep), flush=True)
rng = np.random.default_rng(0)
S = []
for d in D:
    m, st, dist, X = d['mask'], d['stamp'], d['dist'], d['X']
    pos = np.argwhere(m & (dist > 4))
    neg = np.argwhere(st & (dist > 6) & ~(cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0))
    pos = pos[rng.choice(len(pos), min(len(pos), 600), replace=False)]
    neg = neg[rng.choice(len(neg), min(len(neg), 3000), replace=False)]
    S.append((X[pos[:, 0], pos[:, 1]], X[neg[:, 0], neg[:, 1]]))
hits = 0
for i, d in enumerate(D):
    Xtr = np.concatenate([np.concatenate(s) for j, s in enumerate(S) if j != i])
    ytr = np.concatenate([np.r_[np.ones(len(s[0])), np.zeros(len(s[1]))] for j, s in enumerate(S) if j != i])
    w = np.where(ytr == 1, (ytr == 0).sum() / (ytr == 1).sum(), 1.0)
    clf = HistGradientBoostingClassifier(max_iter=args.iters, learning_rate=0.1, max_leaf_nodes=31,
                                         min_samples_leaf=50, l2_regularization=1.0, random_state=0).fit(Xtr, ytr, sample_weight=w)
    X, st, dist, m = d['X'], d['stamp'], d['dist'], d['mask']
    cand = st & (dist > 6)
    sub = np.zeros_like(cand); sub[::2, ::2] = True
    idx = np.argwhere(cand & sub)
    p = np.zeros(m.shape, np.float32)
    p[idx[:, 0], idx[:, 1]] = clf.predict_proba(X[idx[:, 0], idx[:, 1]])[:, 1]
    p = cv2.dilate(p, np.ones((2, 2), np.uint8))
    ps = cv2.GaussianBlur(p, (0, 0), 6); ps[~cand] = -1
    cy, cx = np.unravel_index(np.argmax(ps), ps.shape)
    hit = bool((cv2.dilate(m.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0)[cy, cx])
    hits += hit
    print('exhibit %02d %s  peak %.3f  best-in-mask %.3f' % (i + 1, 'HIT ' if hit else 'miss', ps[cy, cx], ps[m].max()), flush=True)
print('LOO hits: %d / 30' % hits)
