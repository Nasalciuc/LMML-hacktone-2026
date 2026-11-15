import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import sys, copy, warnings, torch, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, f'{SRC}')
from export_model import build_backbone_from_ckpt, quantize_convs, fold_bn
from extract_feats import load_images
torch.set_num_threads(2)
x_u8, labels, files = load_images(f'{DATA}')
x = x_u8[::12][:60].float() / 255.0
net, ck, mean, std = build_backbone_from_ckpt(f'{RUNS}/mnv3_full/model.pt', f'{WEIGHTS}')
m_ = torch.tensor(mean).view(1, 3, 1, 1); s_ = torch.tensor(std).view(1, 3, 1, 1)
xn = (x - m_) / s_
with torch.no_grad():
    bb = net.backbone.eval(); ref = bb(xn); lref = net.head(ref).argmax(1)
    for name, fold in (('int8 no-fold', False), ('int8 fold', True)):
        b2 = copy.deepcopy(bb)
        if fold: fold_bn(b2)
        q = quantize_convs(b2, 'int8').eval(); out = q(xn)
        cos = torch.nn.functional.cosine_similarity(out, ref)
        print(f'{name}: cosine min={float(cos.min()):.4f} mean={float(cos.mean()):.4f}  label agree={float((net.head(out).argmax(1) == lref).float().mean()):.3f}')
