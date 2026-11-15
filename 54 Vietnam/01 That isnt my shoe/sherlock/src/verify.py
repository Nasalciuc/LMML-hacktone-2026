"""Verify an exported .pt the way an evaluator might run it.

usage: python -I verify.py MODEL.pt [--val_run RUN_DIR]
"""
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import argparse
import csv
import io
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser()
ap.add_argument('model')
ap.add_argument('--data', default=f'{DATA}')
ap.add_argument('--val_run', default='')
a = ap.parse_args()
torch.set_num_threads(2)

print('torch', torch.__version__, '| file size', os.path.getsize(a.model), 'bytes')
m = torch.jit.load(a.model, map_location='cpu')
m.eval()
try:
    m2 = torch.load(a.model, map_location='cpu', weights_only=False)
    print('torch.load(weights_only=False) ->', type(m2).__name__)
except Exception as e:
    print('torch.load(weights_only=False) failed:', str(e)[:120])
try:
    torch.load(a.model, map_location='cpu')
    print('torch.load(default) ok')
except Exception as e:
    print('torch.load(default weights_only=True) fails as expected for TorchScript:', str(e).split('\n')[0][:100])

rows = list(csv.DictReader(open(os.path.join(a.data, 'train-output.csv'))))
imgs = [Image.open(os.path.join(a.data, 'train-input', r['filename'])).convert('RGB') for r in rows]
labels = np.array([r['label'] for r in rows])
u8 = torch.from_numpy(np.stack([np.asarray(im) for im in imgs]))  # N,H,W,3

def to_labels(out):
    if isinstance(out, list):
        return np.array(out)
    if out.shape[1] == 4:
        names = list(m.classes4)
        return np.array([names[i] for i in out.argmax(1).tolist()])
    p = torch.softmax(out, 1)
    names = list(m.classes)
    return np.array([names[i] if p[j].max() > 0.5 else 'unknown' for j, i in enumerate(out.argmax(1).tolist())])

sel = np.arange(0, len(rows), 7)[:64]
xb = u8[sel]
fmts = {
    'float01_NCHW': xb.permute(0, 3, 1, 2).float() / 255.0,
    'uint8_NCHW': xb.permute(0, 3, 1, 2).contiguous(),
    'uint8_NHWC': xb.contiguous(),
    'float255_NCHW': xb.permute(0, 3, 1, 2).float(),
    'imagenet_norm': (xb.permute(0, 3, 1, 2).float() / 255.0 - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
}
ref = None
with torch.no_grad():
    for k, x in fmts.items():
        t0 = time.time()
        lab = to_labels(m(x))
        dt = (time.time() - t0) / len(sel)
        if ref is None:
            ref = lab
        print(f'{k:15s} agree_with_ref={np.mean(lab == ref):.3f}  acc_vs_truth={np.mean(lab == labels[sel]):.3f}  '
              f'unknown_frac={np.mean(lab == "unknown"):.3f}  {dt * 1000:.0f} ms/img')
    single = to_labels(m(fmts['float01_NCHW'][0]))
    print('single CHW image ->', single, '(truth', labels[sel][0], ')')
    r224 = F.interpolate(fmts['float01_NCHW'], size=(224, 224), mode='bilinear', antialias=True, align_corners=False)
    lab = to_labels(m(r224))
    print(f'resized-224 input (trace disabled): acc_vs_truth={np.mean(lab == labels[sel]):.3f} unknown_frac={np.mean(lab == "unknown"):.3f}')

if a.val_run:
    d = np.load(os.path.join(a.val_run, 'emb.npz'))
    va = d['va']
    full_bank = m.bank.clone()
    m.bank = full_bank[torch.from_numpy(d['tr'])]   # honest check: held-out images must not sit in the kNN bank
    with torch.no_grad():
        outs = []
        for i in range(0, len(va), 64):
            outs.append(to_labels(m(u8[va[i:i + 64]].permute(0, 3, 1, 2).float() / 255.0)))
    lab = np.concatenate(outs)
    print(f'held-out knowns (n={len(va)}): known_acc={np.mean(lab == labels[va]):.3f}  '
          f'rejected={np.mean(lab == "unknown"):.3f}  wrong_brand={np.mean((lab != labels[va]) & (lab != "unknown")):.3f}')
    m.bank = full_bank

# non-shoe natural images resized to 256 via other paths (should be rejected)
import skimage.data as sd
nat = []
rng = np.random.RandomState(5)
for nm in ['astronaut', 'coffee', 'chelsea', 'rocket', 'cat', 'retina', 'immunohistochemistry', 'hubble_deep_field']:
    im = Image.fromarray(np.asarray(getattr(sd, nm)())[..., :3])
    for t in range(6):
        W, H = im.size
        s = rng.randint(int(min(W, H) * 0.5), min(W, H) + 1)
        l, tp = rng.randint(0, W - s + 1), rng.randint(0, H - s + 1)
        c = im.crop((l, tp, l + s, tp + s)).resize((256, 256), Image.BICUBIC)
        b = io.BytesIO(); c.save(b, 'JPEG', quality=95); b.seek(0)
        nat.append(np.asarray(Image.open(b).convert('RGB')))
x = torch.from_numpy(np.stack(nat)).permute(0, 3, 1, 2).float() / 255.0
with torch.no_grad():
    lab = to_labels(m(x))
print(f'natural non-shoe images (n={len(nat)}): rejected={np.mean(lab == "unknown"):.3f}')
