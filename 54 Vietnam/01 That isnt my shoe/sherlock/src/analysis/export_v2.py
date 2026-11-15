"""v2 export: ensemble of fine-tuned MobileNetV3 models (int8 weights) + kNN novelty (from one member) + provenance trace.

usage: python -I export_v2.py --ckpts A.pt B.pt --knn_idx 1 --calib calib_v2.npz --outdir DIR --variants "..."
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
from export_model import SneakerSherlock, quantize_convs, build_backbone_from_ckpt  # noqa: E402
from trace_torch import fold_matrix  # noqa: E402
from extract_feats import load_images  # noqa: E402


class EnsSherlock(SneakerSherlock):
    knn_idx: int

    def __init__(self, backbones: List[nn.Module], heads_w, heads_b, knn_idx: int, bank_dim: int, **kw):
        super().__init__(backbones[0], heads_w[0], heads_b[0], torch.zeros(1, bank_dim), **kw)
        self.backbone = nn.Identity()   # members live in self.backbones
        self.backbones = nn.ModuleList(backbones)
        self.register_buffer('hw', torch.stack([h.float() for h in heads_w]))   # [M,3,D]
        self.register_buffer('hb', torch.stack([h.float() for h in heads_b]))   # [M,3]
        self.knn_idx = int(knn_idx)

    @torch.jit.export
    def brand_forward(self, x01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x01
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=[self.size, self.size], mode='bilinear', align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        xf = torch.flip(x, dims=[3])
        prob = torch.zeros(x.shape[0], self.hw.shape[1], dtype=x.dtype, device=x.device)
        emb = torch.zeros(x.shape[0], self.hw.shape[2], dtype=x.dtype, device=x.device)
        i = 0
        for bb in self.backbones:
            f = (bb(x) + bb(xf)) * 0.5
            lg = f @ self.hw[i].t() + self.hb[i]
            prob = prob + torch.softmax(lg, 1)
            if i == self.knn_idx:
                emb = f
            i += 1
        prob = prob / float(i)
        return torch.log(prob.clamp_min(1e-8)), emb


class EnsProbs4(EnsSherlock):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_proba4(x)


class EnsLabels(EnsSherlock):
    def forward(self, x: torch.Tensor) -> List[str]:
        return self.predict_labels(x)


class EnsLogits3(EnsSherlock):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_logits3(x)


VARIANTS = {'probs4': EnsProbs4, 'labels': EnsLabels, 'logits3': EnsLogits3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--knn_idx', type=int, default=0)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--trace', default=f'{RUNS}/trace_lr3.npz')
    ap.add_argument('--weights', default=f'{WEIGHTS}')
    ap.add_argument('--data', default=f'{DATA}')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--bank_dtype', default='half')
    ap.add_argument('--variants', default='labels:nike,adidas,converse')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--prefix', default='sherlock_v2')
    a = ap.parse_args()
    torch.set_num_threads(2)
    os.makedirs(a.outdir, exist_ok=True)
    traced, heads, mean, std, dim = [], [], None, None, None
    trained_classes = None
    for ck_path in a.ckpts:
        net, ck, mean_i, std_i = build_backbone_from_ckpt(ck_path, a.weights)
        assert mean is None or (mean == mean_i and std == std_i), 'members must share normalisation'
        mean, std, dim = mean_i, std_i, ck['dim']
        trained_classes = ck['classes']
        bb = quantize_convs(net.backbone, 'int8').eval()
        with torch.no_grad():
            traced.append(torch.jit.trace(bb, torch.rand(2, 3, a.size, a.size), check_trace=False))
        heads.append((net.head.weight.detach(), net.head.bias.detach()))
    cal = np.load(a.calib)
    tr = np.load(a.trace)
    x_u8, labels, files = load_images(a.data)
    bank = None
    for spec in a.variants.split(';'):
        mode, order = spec.split(':')
        classes = order.split(',')
        perm = [trained_classes.index(c) for c in classes]
        model = VARIANTS[mode](
            traced, [h[0][perm] for h in heads], [h[1][perm] for h in heads], a.knn_idx, dim,
            sem_cal=torch.from_numpy(np.sort(cal['sem_cal'])), tr_cal=torch.from_numpy(np.sort(cal['tr_cal'])),
            tr_w=torch.from_numpy(tr['w']), tr_b=torch.tensor(float(tr['b'])), fold_m=fold_matrix(),
            thr_joint=float(cal['thr_joint']), thr_sem=float(cal['thr_sem']), w_trace=float(cal['w_trace']),
            size=a.size, mean=mean, std=std, output_mode=mode, classes=classes)
        model.eval()
        if bank is None:
            feats = []
            with torch.no_grad():
                for i in range(0, len(x_u8), 48):
                    _, f = model.brand_forward(x_u8[i:i + 48].float() / 255.0)
                    feats.append(f / (f * f).sum(1, keepdim=True).sqrt())
            bank = torch.cat(feats)
            if a.bank_dtype == 'int8':
                bank_scale = bank.abs().amax(1, keepdim=True) / 127.0
                bank = torch.round(bank / bank_scale).clamp(-127, 127).to(torch.int8)
            else:
                bank_scale = torch.ones(1, 1)
                bank = bank.half()
        model.bank = bank
        model.bank_scale = bank_scale
        sm = torch.jit.script(model)
        tag = mode if order == 'nike,adidas,converse' else mode + '_' + '-'.join(c[:1] for c in classes)
        out = os.path.join(a.outdir, f'{a.prefix}_{tag}.pt')
        sm.save(out)
        print('saved', out, os.path.getsize(out), 'bytes', flush=True)
        json.dump(dict(ckpts=a.ckpts, knn_idx=a.knn_idx, calib=a.calib, mode=mode, classes=classes, size=a.size,
                       thr_joint=float(cal['thr_joint']), w_trace=float(cal['w_trace']),
                       size_bytes=os.path.getsize(out)), open(out + '.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
