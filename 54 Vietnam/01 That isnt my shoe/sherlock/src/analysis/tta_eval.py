"""Out-of-fold accuracy of test-time-augmentation schemes using the 5 CV fold models."""
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
x_u8, labels, files = load_images(f'{DATA}')
y = np.array([CLASSES.index(l) for l in labels])


def views(x01, scheme, size=224):
    out = []
    def add(name, t):
        out.append((name, t))
        out.append((name + '_flip', torch.flip(t, dims=[3])))
    if 'resize' in scheme:
        add('resize', F.interpolate(x01, size=(size, size), mode='bilinear', align_corners=False, antialias=True))
    if 'full' in scheme:
        add('full', x01)
    if 'center' in scheme:
        o = (x01.shape[-1] - size) // 2
        add('center', x01[:, :, o:o + size, o:o + size])
    if 'up288' in scheme:
        add('up288', F.interpolate(x01, size=(288, 288), mode='bilinear', align_corners=False))
    if 'up320' in scheme:
        add('up320', F.interpolate(x01, size=(320, 320), mode='bilinear', align_corners=False))
    return out


schemes = ['full', 'up288', 'up320', 'full+up288', 'up288+up320', 'full+up288+up320']
res = {s: np.zeros((len(y), 3)) for s in schemes}
FOLDS = [int(v) for v in sys.argv[1].split(',')] if len(sys.argv) > 1 else list(range(5))
for k in FOLDS:
    ck = torch.load(f'{RUNS}/mnv3_cv{k}/model.pt', map_location='cpu', weights_only=False)
    bb, mean, std, size, cfg = build(ck['name'], f'{WEIGHTS}', pretrained=False)
    net = Net(bb, ck['dim'], 3)
    net.load_state_dict(ck['state_dict'])
    net.eval()
    va = np.load(f'{RUNS}/mnv3_cv{k}/emb.npz')['va']
    with torch.no_grad():
        for i in range(0, len(va), 48):
            idx = va[i:i + 48]
            x01 = x_u8[idx].float() / 255.0
            cache = {}
            for s in schemes:
                lg = []
                for key, v in views(x01, s):
                    if key not in cache:
                        cache[key] = torch.softmax(net((v - mean) / std)[0], 1).numpy()
                    lg.append(cache[key])
                res[s][idx] = np.mean(lg, 0)
    print('fold', k, 'done', flush=True)
for s in schemes:
    m = np.abs(res[s]).sum(1) > 0
    print(f'{s:20s} OOF acc={np.mean(res[s][m].argmax(1) == y[m]):.4f} (n={m.sum()})')
