import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import io, sys, copy, warnings, torch, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, f'{SRC}')
from export_model import build_backbone_from_ckpt, quantize_convs, fold_bn
torch.set_num_threads(2)
net, ck, mean, std = build_backbone_from_ckpt(f'{RUNS}/mnv3_full/model.pt', f'{WEIGHTS}')
bb = net.backbone.eval()
x = torch.rand(4, 3, 256, 256)
with torch.no_grad():
    ref = bb(x)
    fb = copy.deepcopy(bb); n = fold_bn(fb); fb.eval()
    print('folds:', n, ' remaining BN modules:', sum(1 for m in fb.modules() if isinstance(m, torch.nn.BatchNorm2d)))
    print('fold max diff:', float((fb(x) - ref).abs().max()), ' ref scale', float(ref.abs().max()))
    qb = quantize_convs(fb, 'int8').eval()
    out = qb(x)
    cos = torch.nn.functional.cosine_similarity(out, ref).min()
    print('fold+int8 min cosine vs float:', float(cos))
    s = torch.jit.script(qb)
    b = io.BytesIO(); torch.jit.save(s, b); print('scripted fold+int8 size', len(b.getvalue()))
    print(fb.blocks[1][0])
