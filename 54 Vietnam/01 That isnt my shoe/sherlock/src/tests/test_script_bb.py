import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import os, sys, io, torch, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, f'{SRC}')
from export_model import build_backbone_from_ckpt, quantize_convs
net, ck, mean, std = build_backbone_from_ckpt(f'{RUNS}/mnv3_full/model.pt', f'{WEIGHTS}')
bb = quantize_convs(net.backbone, 'int8').eval()
x = torch.rand(2, 3, 256, 256)
with torch.no_grad():
    ref = bb(x)
try:
    sbb = torch.jit.script(bb)
    b = io.BytesIO(); torch.jit.save(sbb, b); print('scripted size', len(b.getvalue()))
    print('max diff', float((sbb(x) - ref).abs().max()))
except Exception as e:
    print('script failed:', str(e)[:500])
tb = torch.jit.trace(bb, x, check_trace=False)
b = io.BytesIO(); torch.jit.save(tb, b); print('traced size', len(b.getvalue()))
