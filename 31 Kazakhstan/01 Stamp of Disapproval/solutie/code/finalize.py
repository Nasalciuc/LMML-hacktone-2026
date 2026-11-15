"""Write boxes.csv from a finished train_detect run (prob_maps.npz + peaks) and the feature cache,
applying the same per-page decision rule as train_detect.py (strong DQ contradiction first,
otherwise the classifier's best region)."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from train_detect import region_box

ap = argparse.ArgumentParser()
ap.add_argument('--cache', required=True)
ap.add_argument('--run', required=True)
ap.add_argument('--rounds', type=int, default=3)
ap.add_argument('--dq_strong', type=float, default=-50.0)
args = ap.parse_args()

maps = np.load(f'{args.run}/prob_maps.npz')
pk = json.load(open(f'{args.run}/peaks_r{args.rounds}.json'))
rows = []
for k in sorted(pk):
    d = np.load(f'{args.cache}/{k}.npz')
    names = list(d['names']); X = d['X']; idx = d['idx'].astype(np.int32)
    P = maps[k].astype(np.float32)
    M = np.zeros(P.shape, np.uint8); M[idx[:, 0] // 2, idx[:, 1] // 2] = 1
    dqv = X[:, names.index('dqL33')].astype(np.float32)
    j = int(np.argmin(dqv))
    a, num = k.rsplit('_page_', 1)
    if dqv[j] < args.dq_strong:
        yy, xx = idx[j]
        x, y, w, h = region_box(P, M, (int(xx), int(yy), float(P[yy // 2, xx // 2])))
        if not (x <= xx <= x + w and y <= yy <= y + h):
            x, y, w, h = int(xx) - 30, int(yy) - 22, 60, 44
        src, sc = 'dq', float(dqv[j])
    else:
        x, y, w, h = region_box(P, M, pk[k][0])
        src, sc = 'classifier', pk[k][0][2]
    rows.append((a, 'page_' + num, x, y, w, h, src, sc, pk[k][0][2], float(dqv[j])))
with open(f'{args.run}/boxes.csv', 'w') as fh:
    fh.write('album,page,x,y,w,h,source,score,classifier_top,dq_min\n')
    for r in rows:
        fh.write('%s,%s,%d,%d,%d,%d,%s,%.4f,%.4f,%.1f\n' % r)
print('wrote', len(rows), 'boxes;', sum(r[6] == 'dq' for r in rows), 'from strong DQ evidence')
