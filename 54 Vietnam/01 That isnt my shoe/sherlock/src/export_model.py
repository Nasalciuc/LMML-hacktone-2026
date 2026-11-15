"""Build the self-contained TorchScript open-set model.

Pipeline inside the .pt (no external code needed):
  input  : Tensor, any of uint8/float, NCHW/NHWC/CHW, [0,1] or [0,255] or mean/std-normalised, any H,W
  brand  : fine-tuned MobileNetV3 (int8 per-channel weights, dequantised on the fly) -> 3 logits + 1280-d embedding
  scores : semantic novelty = cosine to nearest training embedding (kNN bank)
           provenance      = logistic score of the 240->256 resize trace (only used when the input is 256x256)
  decide : Fisher combination of the two calibrated p-values vs a threshold; reject -> 'unknown'
Output variants (selected at export):
  probs4  : forward -> Tensor [N,4] probabilities over CLASSES4 (argmax = decision)
  labels  : forward -> List[str]
  logits3 : forward -> Tensor [N,3]; rejected rows are flat (softmax = 1/3 each), accepted rows near one-hot
"""
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_feats import build  # noqa: E402
from finetune import Net  # noqa: E402
from trace_torch import fold_matrix  # noqa: E402


# --------------------------------------------------------------------------- weight-only quantisation
class QConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, mode: str):
        super().__init__()
        w = conv.weight.detach().float()
        if mode == 'int8':
            scale = w.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8) / 127.0
            self.register_buffer('wq', torch.round(w / scale).clamp(-127, 127).to(torch.int8))
            self.register_buffer('scale', scale.float())
        else:
            self.register_buffer('wq', w.half())
            self.register_buffer('scale', torch.ones(w.shape[0], 1, 1, 1))
        b = conv.bias.detach().float() if conv.bias is not None else torch.zeros(w.shape[0])
        self.register_buffer('bias', b)
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

    def forward(self, x):
        w = self.wq.float() * self.scale
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def quantize_convs(module: nn.Module, mode: str, min_numel: int = 0):
    """mode: int8 | fp16 | mixed (fp16 for depthwise convs and the stem, int8 for everything else)"""
    for name, child in list(module.named_children()):
        if type(child) is nn.Conv2d and child.weight.numel() >= min_numel:
            m = mode
            if mode == 'mixed':
                m = 'fp16' if (child.groups > 1 or name == 'conv_stem') else 'int8'
            setattr(module, name, QConv2d(child, m))
        else:
            quantize_convs(child, mode, min_numel)
    return module


# --------------------------------------------------------------------------- scripted wrapper
class SneakerSherlock(nn.Module):
    classes: List[str]
    classes4: List[str]
    output_mode: str

    def __init__(self, backbone, head_w, head_b, bank, sem_cal, tr_cal, tr_w, tr_b, fold_m,
                 thr_joint: float, thr_sem: float, w_trace: float, size: int, mean, std, output_mode: str,
                 classes: List[str]):
        super().__init__()
        self.backbone = backbone
        self.register_buffer('head_w', head_w.float())
        self.register_buffer('head_b', head_b.float())
        self.register_buffer('bank', bank.half())               # [K, D] L2-normalised training embeddings
        self.register_buffer('bank_scale', torch.ones(1, 1))   # per-row scale when the bank is stored as int8
        self.register_buffer('sem_cal', sem_cal.float())        # sorted calibration semantic scores (knowns, OOF)
        self.register_buffer('tr_cal', tr_cal.float())          # sorted calibration trace scores (knowns, OOF)
        self.register_buffer('tr_w', tr_w.float())
        self.register_buffer('tr_b', tr_b.float().reshape(1))
        self.register_buffer('fold_m', fold_m.float())          # [254, 16]
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1).float())
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1).float())
        self.register_buffer('in_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('in_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.thr_joint = float(thr_joint)
        self.thr_sem = float(thr_sem)
        self.w_trace = float(w_trace)
        self.size = int(size)
        self.output_mode = output_mode
        self.classes = classes
        self.classes4 = classes + ['unknown']

    # ---------------- input handling
    @torch.jit.export
    def to_unit_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Any reasonable image tensor -> float [N,3,H,W] in [0,1]."""
        is_u8 = x.dtype == torch.uint8
        x = x.float()
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        if x.dim() == 3:
            if x.shape[0] in (1, 3, 4):
                x = x.unsqueeze(0)
            elif x.shape[-1] in (1, 3, 4):
                x = x.permute(2, 0, 1).unsqueeze(0)
            else:
                x = x.unsqueeze(1)
        if x.dim() == 5:
            x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        if x.shape[1] not in (1, 3, 4) and x.shape[-1] in (1, 3, 4):
            x = x.permute(0, 3, 1, 2)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[1] == 4:
            x = x[:, :3]
        if is_u8:
            return x / 255.0
        mn = float(x.min())
        mx = float(x.max())
        if mn < -0.02:
            if mn < -1.05 or mx > 1.05:
                x = x * self.in_std + self.in_mean      # ImageNet-normalised input
            else:
                x = (x + 1.0) * 0.5                     # [-1,1]-normalised input
        elif mx > 1.5:
            x = x / 255.0
        return x.clamp(0.0, 1.0)

    # ---------------- components
    def _zn(self, f: torch.Tensor) -> torch.Tensor:
        return (f - f.mean(1, keepdim=True)) / (f.std(1, unbiased=False, keepdim=True) + 1e-9)

    @torch.jit.export
    def trace_features(self, x01: torch.Tensor) -> torch.Tensor:
        M = self.fold_m
        g = (x01[:, 0] * 0.299 + x01[:, 1] * 0.587 + x01[:, 2] * 0.114) * 255.0
        d2x = g[:, :, 2:] - 2.0 * g[:, :, 1:-1] + g[:, :, :-2]
        d2y = g[:, 2:, :] - 2.0 * g[:, 1:-1, :] + g[:, :-2, :]
        ax = d2x.abs()
        ay = d2y.abs()
        f1 = torch.cat([self._zn(ax.mean(1) @ M), self._zn(ay.mean(2) @ M)], 1)
        f2 = torch.cat([self._zn((d2x * d2x).mean(1) @ M), self._zn((d2y * d2y).mean(2) @ M)], 1)
        d1x = (g[:, :, 1:] - g[:, :, :-1]).abs().mean(1)[:, :254]
        d1y = (g[:, 1:, :] - g[:, :-1, :]).abs().mean(2)[:, :254]
        f3 = torch.cat([self._zn(d1x @ M), self._zn(d1y @ M)], 1)
        rx = ax / (ax.mean(2, keepdim=True) + 1.0)
        ry = ay / (ay.mean(1, keepdim=True) + 1.0)
        f4 = torch.cat([self._zn(rx.mean(1) @ M), self._zn(ry.mean(2) @ M)], 1)
        return torch.cat([f1, f2, f3, f4], 1)

    @torch.jit.export
    def trace_score(self, x01: torch.Tensor) -> torch.Tensor:
        return self.trace_features(x01) @ self.tr_w + self.tr_b

    @torch.jit.export
    def brand_forward(self, x01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x01
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=[self.size, self.size], mode='bilinear', align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        f1 = self.backbone(x)
        f2 = self.backbone(torch.flip(x, dims=[3]))
        f = (f1 + f2) * 0.5
        logits = f @ self.head_w.t() + self.head_b
        return logits, f

    def pvalue(self, cal: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(cal, s.contiguous(), right=True).float()
        return (idx + 1.0) / (float(cal.shape[0]) + 2.0)

    @torch.jit.export
    def analyse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """returns (brand logits [N,3], accept mask [N], semantic score [N], trace score [N] (nan if unused))"""
        x01 = self.to_unit_rgb(x)
        use_trace = x01.shape[-1] == 256 and x01.shape[-2] == 256
        bank = self.bank.float() * self.bank_scale
        lg: List[torch.Tensor] = []
        ss: List[torch.Tensor] = []
        st: List[torch.Tensor] = []
        n = int(x01.shape[0])
        for i in range(0, n, 32):             # chunking keeps memory flat for large batches
            xc = x01[i:i + 32]
            l_c, f_c = self.brand_forward(xc)
            fn = f_c / ((f_c * f_c).sum(1, keepdim=True).sqrt() + 1e-12)
            lg.append(l_c)
            ss.append((fn @ bank.t()).max(1).values)
            if use_trace:
                st.append(self.trace_score(xc))
        logits = torch.cat(lg, 0)
        s_sem = torch.cat(ss, 0)
        p_sem = self.pvalue(self.sem_cal, s_sem)
        if use_trace:
            s_tr = torch.cat(st, 0)
            p_tr = self.pvalue(self.tr_cal, s_tr)
            stat = torch.log(p_sem) + self.w_trace * torch.log(p_tr)
            accept = stat >= self.thr_joint
        else:
            s_tr = torch.full_like(s_sem, float('nan'))
            accept = torch.log(p_sem) >= self.thr_sem
        return logits, accept, s_sem, s_tr

    @torch.jit.export
    def predict_labels(self, x: torch.Tensor) -> List[str]:
        logits, accept, _, _ = self.analyse(x)
        pred = logits.argmax(1)
        out: List[str] = []
        for i in range(int(pred.shape[0])):
            if bool(accept[i]):
                out.append(self.classes[int(pred[i])])
            else:
                out.append('unknown')
        return out

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> List[str]:
        return self.predict_labels(x)

    @torch.jit.export
    def predict_proba4(self, x: torch.Tensor) -> torch.Tensor:
        logits, accept, _, _ = self.analyse(x)
        p3 = torch.softmax(logits, 1)
        acc = accept.float().unsqueeze(1)
        known = torch.cat([p3 * 0.9 + 0.0, torch.full_like(p3[:, :1], 0.1)], 1)
        unk = torch.cat([p3 * 0.1, torch.full_like(p3[:, :1], 0.9)], 1)
        # accepted rows: brand prob >= 0.3 > 0.1 = unknown ; rejected rows: unknown 0.9 > any brand
        out = acc * known + (1.0 - acc) * unk
        # make sure argmax of accepted rows is the brand even when p3 is flat
        bonus = torch.zeros_like(out)
        bonus.scatter_(1, logits.argmax(1, keepdim=True), 1e-3)
        return out + bonus * acc

    @torch.jit.export
    def predict_logits3(self, x: torch.Tensor) -> torch.Tensor:
        logits, accept, _, _ = self.analyse(x)
        onehot = torch.zeros_like(logits).scatter_(1, logits.argmax(1, keepdim=True), 1.0)
        sharp = onehot * 20.0                                  # softmax max ~ 1.0
        flat = torch.zeros_like(logits)                        # softmax = 1/3 each
        acc = accept.float().unsqueeze(1)
        return acc * sharp + (1.0 - acc) * flat



class SherlockProbs4(SneakerSherlock):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_proba4(x)


class SherlockLabels(SneakerSherlock):
    def forward(self, x: torch.Tensor) -> List[str]:
        return self.predict_labels(x)


class SherlockLogits3(SneakerSherlock):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_logits3(x)


VARIANTS = {'probs4': SherlockProbs4, 'labels': SherlockLabels, 'logits3': SherlockLogits3}


def build_backbone_from_ckpt(ckpt_path, weights_dir):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    backbone, mean, std, size0, cfg = build(ck['name'], weights_dir, pretrained=False)
    net = Net(backbone, ck['dim'], len(ck['classes']))
    net.load_state_dict(ck['state_dict'])
    net.eval()
    return net, ck, tuple(float(v) for v in mean.flatten()), tuple(float(v) for v in std.flatten())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--calib', required=True, help='npz with sem_cal, tr_cal, thresholds')
    ap.add_argument('--trace', default=f'{RUNS}/trace_lr3.npz')
    ap.add_argument('--weights', default=f'{WEIGHTS}')
    ap.add_argument('--data', default=f'{DATA}')
    ap.add_argument('--quant', default='int8')
    ap.add_argument('--variants', default='probs4:nike,adidas,converse',
                    help='semicolon-separated list of mode:class_order (mode in probs4|labels|logits3)')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--prefix', default='sherlock')
    ap.add_argument('--infer_size', type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(2)
    os.makedirs(a.outdir, exist_ok=True)
    net, ck, mean, std = build_backbone_from_ckpt(a.ckpt, a.weights)
    size = a.infer_size or ck['size']
    trained_classes = ck['classes']
    bb = quantize_convs(net.backbone, a.quant)
    bb.eval()
    with torch.no_grad():
        traced = torch.jit.trace(bb, torch.rand(2, 3, size, size), check_trace=False)
    cal = np.load(a.calib)
    tr = np.load(a.trace)
    from extract_feats import load_images
    x_u8, labels, files = load_images(a.data)
    bank = None
    for spec in a.variants.split(';'):
        mode, order = spec.split(':')
        classes = order.split(',')
        perm = [trained_classes.index(c) for c in classes]   # reorder head rows to the requested class order
        head_w = net.head.weight.detach()[perm]
        head_b = net.head.bias.detach()[perm]
        model = VARIANTS[mode](
            traced, head_w, head_b, torch.zeros(1, ck['dim']), torch.from_numpy(np.sort(cal['sem_cal'])),
            torch.from_numpy(np.sort(cal['tr_cal'])), torch.from_numpy(tr['w']), torch.tensor(float(tr['b'])),
            fold_matrix(), float(cal['thr_joint']), float(cal['thr_sem']), float(cal['w_trace']), size, mean, std,
            mode, classes)
        model.eval()
        if bank is None:
            # kNN bank = embeddings of all training images computed with the exported (quantised) backbone
            feats = []
            with torch.no_grad():
                for i in range(0, len(x_u8), 64):
                    _, f = model.brand_forward(x_u8[i:i + 64].float() / 255.0)
                    feats.append(f / (f * f).sum(1, keepdim=True).sqrt())
            bank = torch.cat(feats).half()
        model.bank = bank
        sm = torch.jit.script(model)
        tag = mode if order == 'nike,adidas,converse' else mode + '_' + '-'.join(c[:1] for c in classes)
        out = os.path.join(a.outdir, f'{a.prefix}_{tag}.pt')
        sm.save(out)
        print('saved', out, os.path.getsize(out), 'bytes')
        meta = dict(ckpt=a.ckpt, calib=a.calib, quant=a.quant, output_mode=mode, classes=classes, infer_size=size,
                    thr_joint=float(cal['thr_joint']), thr_sem=float(cal['thr_sem']), w_trace=float(cal['w_trace']),
                    size_bytes=os.path.getsize(out))
        json.dump(meta, open(out + '.json', 'w'), indent=1)


if __name__ == '__main__':
    main()


# --------------------------------------------------------------------------- BatchNorm folding (timm MobileNet/EfficientNet blocks)
_BN_PAIRS = {
    'DepthwiseSeparableConv': [('conv_dw', 'bn1'), ('conv_pw', 'bn2')],
    'InvertedResidual': [('conv_pw', 'bn1'), ('conv_dw', 'bn2'), ('conv_pwl', 'bn3')],
    'EdgeResidual': [('conv_exp', 'bn1'), ('conv_pwl', 'bn2')],
    'ConvBnAct': [('conv', 'bn1')],
    'MobileNetV3': [('conv_stem', 'bn1')],
    'EfficientNet': [('conv_stem', 'bn1')],
}


def _fold_one(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    w = conv.weight.detach().float()
    b = conv.bias.detach().float() if conv.bias is not None else torch.zeros(w.shape[0])
    g = bn.weight.detach().float() if bn.weight is not None else torch.ones(w.shape[0])
    beta = bn.bias.detach().float() if bn.bias is not None else torch.zeros(w.shape[0])
    s = g / torch.sqrt(bn.running_var.float() + bn.eps)
    new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode)
    new.weight.data = w * s.view(-1, 1, 1, 1)
    new.bias.data = (b - bn.running_mean.float()) * s + beta
    return new


def fold_bn(module: nn.Module) -> int:
    """Fold BatchNorm(-Act) layers into the preceding conv for known timm block layouts. Returns #folds."""
    n = 0
    for m in list(module.modules()):
        for cname, bname in _BN_PAIRS.get(type(m).__name__, []):
            conv, bn = getattr(m, cname, None), getattr(m, bname, None)
            if type(conv) is nn.Conv2d and isinstance(bn, nn.BatchNorm2d) and bn.num_features == conv.out_channels:
                setattr(m, cname, _fold_one(conv, bn))
                act = getattr(bn, 'act', None)
                drop = getattr(bn, 'drop', None)
                rest = [x for x in (drop, act) if x is not None and not isinstance(x, nn.Identity)]
                setattr(m, bname, nn.Sequential(*rest) if rest else nn.Identity())
                n += 1
    return n
