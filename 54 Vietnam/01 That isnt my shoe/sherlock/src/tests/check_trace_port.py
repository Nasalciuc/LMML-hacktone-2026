import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import sys, csv, os, numpy as np, torch
from PIL import Image
sys.path.insert(0, f'{SRC}')
from trace_torch import fold_matrix, trace_features
D = f'{DATA}'
rows = list(csv.DictReader(open(os.path.join(D, 'train-output.csv'))))[:40]
x = torch.from_numpy(np.stack([np.asarray(Image.open(os.path.join(D, 'train-input', r['filename'])).convert('RGB')) for r in rows])).permute(0, 3, 1, 2).float() / 255.0
m = torch.jit.load(sys.argv[1])
a = trace_features(x, fold_matrix())
b = m.trace_features(x)
tr = np.load(f'{RUNS}/trace_lr3.npz')
print('max |feat diff| =', float((a - b).abs().max()))
print('score diff =', float(((a @ torch.from_numpy(tr['w']) + float(tr['b'])) - m.trace_score(x)).abs().max()))
