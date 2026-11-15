"""Train the tamper detector and localise one region per album page.

round 0 : gradient boosting on pixels of the 30 exhibits (mask = altered).
round k : + page negatives (stamp pixels far from the current best candidates)
          + page pseudo-positives (pixels at the current best candidate of the most
            confident half of the pages).  Every page holds exactly one altered region,
            so this adapts the detector to the album pages without any page labels.
Output: boxes.csv (album, page, x, y, w, h), per-page probability maps, per-album sheets.
"""
import sys, os, json, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from sklearn.ensemble import HistGradientBoostingClassifier

H, W = 1200, 1600


def load_exhibits(cache, rng, npos=1000, nneg=4000):
    Xs, ys, names = [], [], None
    for i in range(1, 31):
        d = np.load(f'{cache}/ex{i:02d}.npz')
        X = d['X']; names = list(d['names'])
        m = d['mask']; st = d['stamp']; dist = d['dist'].astype(np.float32)
        pos = np.argwhere(m & (dist > 4))
        neg = np.argwhere(st & (dist > 6) & ~(cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0))
        pos = pos[rng.choice(len(pos), min(len(pos), npos), replace=False)]
        neg = neg[rng.choice(len(neg), min(len(neg), nneg), replace=False)]
        Xs += [X[pos[:, 0], pos[:, 1]], X[neg[:, 0], neg[:, 1]]]
        ys += [np.ones(len(pos)), np.zeros(len(neg))]
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys), names


def load_pages(cache, names):
    PG = {}
    for f in sorted(os.listdir(cache)):
        if f.startswith('ex') or not f.endswith('.npz'):
            continue
        d = np.load(f'{cache}/{f}')
        assert list(d['names']) == names
        PG[f[:-4]] = (d['X'], d['idx'].astype(np.int32))
    return PG


def fit(X, y, seed=0):
    w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                         min_samples_leaf=50, l2_regularization=1.0, random_state=seed)
    clf.fit(X, y, sample_weight=w)
    return clf


def prob_maps(clf, PG):
    out = {}
    for key, (X, idx) in PG.items():
        pr = np.zeros(len(idx), np.float32)
        for s in range(0, len(idx), 50000):
            pr[s:s + 50000] = clf.predict_proba(X[s:s + 50000].astype(np.float32))[:, 1]
        P = np.zeros((H // 2, W // 2), np.float32); P[idx[:, 0] // 2, idx[:, 1] // 2] = pr
        M = np.zeros((H // 2, W // 2), np.uint8); M[idx[:, 0] // 2, idx[:, 1] // 2] = 1
        out[key] = (P, M)
    return out


def smooth(P, M, sigma):
    S = cv2.GaussianBlur(P, (0, 0), sigma) / (cv2.GaussianBlur(M.astype(np.float32), (0, 0), sigma) + 1e-6)
    S[M == 0] = -1
    return S


def peaks(P, M, sigma=3.0, k=5, sep=20):
    S = smooth(P, M, sigma)
    res = []; Q = S.copy()
    for _ in range(k):
        cy, cx = np.unravel_index(np.argmax(Q), Q.shape)
        res.append((int(cx * 2), int(cy * 2), float(Q[cy, cx])))
        cv2.circle(Q, (cx, cy), sep, -1, -1)
    return res


def region_box(P, M, peak, sigma=3.0):
    """bounding box of the blob around the peak (half of peak height), clamped to plausible sizes"""
    S = smooth(P, M, sigma)
    x, y, s = peak
    cx, cy = x // 2, y // 2
    blob = (S >= 0.5 * s).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(blob)
    k = lab[cy, cx]
    if k == 0:
        return x - 30, y - 22, 60, 44
    bx, by, bw, bh, _ = st[k]
    bx, by, bw, bh = 2 * bx, 2 * by, 2 * bw, 2 * bh
    bw = int(np.clip(bw, 30, 120)); bh = int(np.clip(bh, 24, 100))
    return int(x - bw // 2), int(y - bh // 2), bw, bh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--pos_frac', type=float, default=0.5)
    ap.add_argument('--neg_per_page', type=int, default=1500)
    ap.add_argument('--dq_strong', type=float, default=-50.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    EX, EY, names = load_exhibits(args.cache, rng)
    PG = load_pages(args.cache, names)
    keys = sorted(PG)
    clf = fit(EX, EY)
    log = []
    for r in range(args.rounds + 1):
        maps = prob_maps(clf, PG)
        pk = {k: peaks(*maps[k]) for k in keys}
        json.dump(pk, open(f'{args.out}/peaks_r{r}.json', 'w'))
        msg = 'round %d mean top-1 score %.3f' % (r, np.mean([pk[k][0][2] for k in keys]))
        print(msg, flush=True); log.append(msg)
        if r == args.rounds:
            break
        conf = np.array([pk[k][0][2] - pk[k][1][2] for k in keys])
        thr = np.quantile(conf, 1 - args.pos_frac)
        PX, Py = [], []
        for j, k in enumerate(keys):
            X, idx = PG[k]
            pts = np.array([[y, x] for x, y, s in pk[k][:3]])
            dd = np.sqrt(((idx[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
            negc = np.where(dd.min(1) > 50)[0]
            sel = rng.choice(negc, min(args.neg_per_page, len(negc)), replace=False)
            PX.append(X[sel].astype(np.float32)); Py.append(np.zeros(len(sel)))
            if conf[j] >= thr:
                posc = np.where(dd[:, 0] < 16)[0]
                sel = rng.choice(posc, min(200, len(posc)), replace=False)
                PX.append(X[sel].astype(np.float32)); Py.append(np.ones(len(sel)))
        clf = fit(np.concatenate([EX] + PX), np.concatenate([EY] + Py))
    pickle.dump((clf, names), open(f'{args.out}/model.pkl', 'wb'))
    np.savez_compressed(f'{args.out}/prob_maps.npz', **{k: v[0].astype(np.float16) for k, v in maps.items()})
    # final decision per page: a strong double-quantisation contradiction (summed log-likelihood
    # ratio < DQ_STRONG over a 33x33 window) is taken as is -- on the exhibits this rule never
    # fired outside the altered region; otherwise the classifier's best region.
    jL = names.index('dqL33')
    rows = []
    for k in keys:
        a, num = k.rsplit('_page_', 1)
        page = 'page_' + num
        P, M = maps[k]
        X, idx = PG[k]
        dqv = X[:, jL].astype(np.float32)
        j = int(np.argmin(dqv))
        if dqv[j] < args.dq_strong:
            yy, xx = idx[j]
            x, y, w, h = region_box(P, M, (int(xx), int(yy), float(P[yy // 2, xx // 2])))
            if not (x <= xx <= x + w and y <= yy <= y + h):
                x, y, w, h = int(xx) - 30, int(yy) - 22, 60, 44
            src, sc = 'dq', float(dqv[j])
        else:
            x, y, w, h = region_box(P, M, pk[k][0])
            src, sc = 'classifier', pk[k][0][2]
        rows.append((a, page, x, y, w, h, src, sc, pk[k][0][2], float(dqv[j])))
    with open(f'{args.out}/boxes.csv', 'w') as fh:
        fh.write('album,page,x,y,w,h,source,score,classifier_top,dq_min\n')
        for r_ in rows:
            fh.write('%s,%s,%d,%d,%d,%d,%s,%.4f,%.4f,%.1f\n' % r_)
    open(f'{args.out}/train_log.txt', 'w').write('\n'.join(log) + '\n')


if __name__ == '__main__':
    main()
