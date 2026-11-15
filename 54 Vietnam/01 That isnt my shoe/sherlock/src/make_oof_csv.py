"""Write per-image out-of-fold predictions/scores (5-fold CV, 256-px inference) and the v1 accept/reject decision."""
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import csv
import os
import sys
import numpy as np

RUNS = sys.argv[1]
DATA = sys.argv[2]
OUT = sys.argv[3]
CLASSES = ['nike', 'adidas', 'converse']
rows = list(csv.DictReader(open(os.path.join(DATA, 'train-output.csv'))))
cal = np.load(os.path.join(RUNS, 'calib_v1.npz'))
tr = np.load(os.path.join(RUNS, 'trace_lr3.npz'))
oof_tr = tr['oof_gen']
n = len(rows)
fold = np.full(n, -1)
pred = np.full(n, -1)
sem = np.full(n, np.nan)
for k in range(5):
    d = np.load(os.path.join(RUNS, f'mnv3_cv{k}', 'emb256.npz'))
    fb = d['fb'] / np.linalg.norm(d['fb'], axis=1, keepdims=True)
    fv = d['fv'] / np.linalg.norm(d['fv'], axis=1, keepdims=True)
    va = d['va']
    fold[va] = k
    pred[va] = d['lv'].argmax(1)
    sem[va] = (fv @ fb.T).max(1)
assert (fold >= 0).all()


def pv(c, s):
    c = np.sort(c)
    return (np.searchsorted(c, s, side='right') + 1.0) / (len(c) + 2.0)


p_sem = pv(cal['sem_cal'], sem)
p_tr = pv(cal['tr_cal'], oof_tr)
stat = np.log(p_sem) + float(cal['w_trace']) * np.log(p_tr)
acc = stat >= float(cal['thr_joint'])
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['filename', 'label', 'cv_fold', 'oof_pred', 'oof_correct', 'oof_knn_cosine', 'oof_trace_score',
                'p_semantic', 'p_trace', 'fisher_stat', 'accepted', 'final_label'])
    for i, r in enumerate(rows):
        fl = CLASSES[pred[i]] if acc[i] else 'unknown'
        w.writerow([r['filename'], r['label'], fold[i], CLASSES[pred[i]], int(CLASSES[pred[i]] == r['label']),
                    f'{sem[i]:.5f}', f'{oof_tr[i]:.4f}', f'{p_sem[i]:.5f}', f'{p_tr[i]:.5f}', f'{stat[i]:.4f}',
                    int(acc[i]), fl])
lab = np.array([r['label'] for r in rows])
final = np.array([CLASSES[pred[i]] if acc[i] else 'unknown' for i in range(n)])
print(f'OOF closed-set acc={np.mean(np.array(CLASSES)[pred] == lab):.4f}  accepted={acc.mean():.4f}  '
      f'known_acc (final label == truth)={np.mean(final == lab):.4f}')
