import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import os, sys, csv
import numpy as np
from PIL import Image
D = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(D, 'train-output.csv'))))
img_dir = os.path.join(D, 'train-input')
specs = []
labs = []
for r in rows:
    x = np.asarray(Image.open(os.path.join(img_dir, r['filename'])).convert('L'), dtype=np.float64)
    # second derivative along x and along y
    dx = np.abs(x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]).mean(axis=0)
    dy = np.abs(x[2:, :] - 2 * x[1:-1, :] + x[:-2, :]).mean(axis=1)
    s = np.abs(np.fft.rfft(dx - dx.mean())) + np.abs(np.fft.rfft(dy - dy.mean()))
    specs.append(s / (np.median(s) + 1e-9))
    labs.append(r['label'])
S = np.array(specs)
n = 254
freqs = np.fft.rfftfreq(n)
m = S.mean(0)
top = np.argsort(-m)[:12]
print('top mean-spectrum peaks (freq cycles/px, period px, strength):')
for i in top:
    print('  %.4f  %.2f  %.2f' % (freqs[i], 1 / freqs[i] if freqs[i] > 0 else 0, m[i]))
# per image: is the strongest peak at period ~16?
k16 = np.argmin(np.abs(freqs - 1/16.0))
k8 = np.argmin(np.abs(freqs - 1/8.0))
print('per-image normalized strength at 1/16:', np.percentile(S[:, k16], [5, 25, 50, 75, 95]).round(2))
print('per-image normalized strength at 1/8 :', np.percentile(S[:, k8], [5, 25, 50, 75, 95]).round(2))
labs = np.array(labs)
for c in ['nike', 'adidas', 'converse']:
    print(c, 'median@1/16 %.2f  frac>5: %.2f' % (np.median(S[labs == c, k16]), (S[labs == c, k16] > 5).mean()))
