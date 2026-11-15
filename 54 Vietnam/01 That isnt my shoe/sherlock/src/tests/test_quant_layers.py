import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import sys, copy, warnings, torch, numpy as np
import torch.nn as nn
warnings.filterwarnings('ignore')
sys.path.insert(0, f'{SRC}')
from export_model import build_backbone_from_ckpt, QConv2d, fold_bn
from extract_feats import load_images
torch.set_num_threads(2)
x_u8, labels, files = load_images(f'{DATA}')
x = x_u8[::12][:60].float() / 255.0
net, ck, mean, std = build_backbone_from_ckpt(f'{RUNS}/mnv3_full/model.pt', f'{WEIGHTS}')
xn = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
def quant_sel(module, pred, mode):
    for name, child in list(module.named_children()):
        if type(child) is nn.Conv2d and pred(name, child):
            setattr(module, name, QConv2d(child, mode))
        else:
            quant_sel(child, pred, mode)
with torch.no_grad():
    bb = net.backbone.eval(); fold_bn(bb); ref = bb(xn)
    tests = {
        'all int8': lambda n, c: True,
        'depthwise only': lambda n, c: c.groups > 1,
        'pointwise only (1x1, groups=1)': lambda n, c: c.groups == 1 and c.kernel_size == (1, 1) and n not in ('conv_head',) and 'conv_reduce' not in n and 'conv_expand' not in n,
        'SE convs only': lambda n, c: n in ('conv_reduce', 'conv_expand'),
        'conv_head only': lambda n, c: n == 'conv_head',
        'stem only': lambda n, c: n == 'conv_stem',
        'all fp16': lambda n, c: True,
    }
    for t, pred in tests.items():
        b2 = copy.deepcopy(bb); quant_sel(b2, pred, 'fp16' if 'fp16' in t else 'int8'); out = b2(xn)
        cos = torch.nn.functional.cosine_similarity(out, ref)
        print(f'{t:32s} cos min={float(cos.min()):.4f} mean={float(cos.mean()):.4f}')
