"""Recompute out-of-fold logits/embeddings for each CV fold model with a given inference size (default 256 = native).

Writes RUN/emb{size}.npz with the same keys as finetune.py's emb.npz (lb, fb, ybank, lv, fv, yv, va, tr).
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
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_feats import build, load_images  # noqa: E402
from finetune import Net, CLASSES  # noqa: E402

torch.set_num_threads(2)
size = int(sys.argv[1])
runs = sys.argv[2:]
x_u8, labels, files = load_images(f'{DATA}')
y = np.array([CLASSES.index(l) for l in labels])


@torch.no_grad()
def embed(net, mean, std, idx):
    L, Fs = [], []
    for i in range(0, len(idx), 48):
        x = x_u8[idx[i:i + 48]].float() / 255.0
        if x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
        x = (x - mean) / std
        l1, f1 = net(x)
        l2, f2 = net(torch.flip(x, dims=[3]))
        L.append(((l1 + l2) / 2).numpy())
        Fs.append(((f1 + f2) / 2).numpy())
    return np.concatenate(L), np.concatenate(Fs)


for run in runs:
    ck = torch.load(f'{run}/model.pt', map_location='cpu', weights_only=False)
    bb, mean, std, _, _ = build(ck['name'], f'{WEIGHTS}', pretrained=False)
    net = Net(bb, ck['dim'], len(ck['classes']))
    net.load_state_dict(ck['state_dict'])
    net.eval()
    e = np.load(f'{run}/emb.npz')
    tr, va = e['tr'], e['va']
    lb, fb = embed(net, mean, std, tr)
    lv, fv = embed(net, mean, std, va)
    np.savez(f'{run}/emb{size}.npz', lb=lb, fb=fb, ybank=y[tr], lv=lv, fv=fv, yv=y[va], va=va, tr=tr)
    print(run, 'val acc @', size, '=', round(float((lv.argmax(1) == y[va]).mean()), 4), flush=True)
