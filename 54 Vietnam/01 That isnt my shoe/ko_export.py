"""Pack the known-brands-only network into one self-contained TorchScript file.

python ko_export.py --ckpt results/ko_final_<tag>.pth --score maxlogit --thr <t> --out vietnam_model.pt [--res 256]

- weights stored as fp16 with BatchNorm folded into the convs, cast to fp32 inside forward (CPU safe)
- preprocessing inside forward: layout / dtype / value range guessed, resize to the training size
- forward(x) -> (B,4) scores, columns = classes = [adidas, converse, nike, unknown]; argmax = decision
- open-set rule: score(x) < thr -> unknown, else argmax of the three brand logits
"""
import argparse, os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import oe_lib as L
from oe_export import to_small_storage

SCORES = {"msp": 0, "maxlogit": 1, "energy": 2, "proto": 3, "msp_proto": 4}


class ShoeModel(nn.Module):
    classes: List[str]
    class_to_idx: Dict[str, int]
    input_size: int
    score_kind: int
    score_name: str
    thr: float
    sharp: float

    def __init__(self, net: nn.Module, score: str, thr: float, sharp: float, proto: torch.Tensor):
        super().__init__()
        self.net = net
        self.classes = list(L.CLASSES)                       # adidas, converse, nike, unknown
        self.class_to_idx = {c: i for i, c in enumerate(L.CLASSES)}
        self.input_size = L.RES
        self.score_kind = SCORES[score]
        self.score_name = score
        self.thr = float(thr)
        self.sharp = float(sharp)
        self.register_buffer("mean", L.MEAN.clone())
        self.register_buffer("std", L.STD.clone())
        self.register_buffer("proto", proto)                 # 3 x D unit-length class means of training features

    def prep(self, x: torch.Tensor) -> torch.Tensor:
        """any reasonable image tensor -> normalised float32 (B,3,input_size,input_size)"""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError("expected an image tensor with 3 or 4 dims")
        if x.shape[1] not in (1, 3, 4) and x.shape[-1] in (1, 3, 4):
            x = x.permute(0, 3, 1, 2)                    # channels-last -> channels-first
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        elif x.shape[1] == 4:
            x = x[:, :3]                                 # drop alpha
        is_int = not x.is_floating_point()
        x = x.to(device=self.mean.device, dtype=torch.float32)
        if is_int:
            x = x / 255.0
        else:                                            # one value range per call, guessed from min/max
            mn = float(x.min())
            mx = float(x.max())
            if mx > 2.7:                                 # 0-255 stored as float
                x = x / 255.0
            elif mn < -1.05 or mx > 1.05:                # ImageNet mean/std already applied
                x = x * self.std + self.mean
            elif mn < -0.02:                             # (x - 0.5) / 0.5
                x = x * 0.5 + 0.5
        x = x.clamp(0.0, 1.0)
        if x.shape[2] != self.input_size or x.shape[3] != self.input_size:
            x = F.interpolate(x, size=[self.input_size, self.input_size], mode="bilinear",
                              antialias=True, align_corners=False)
        return (x - self.mean) / self.std

    def feats(self, xin: torch.Tensor) -> torch.Tensor:
        return self.net.avgpool(self.net.features(xin)).flatten(1)

    @torch.jit.export
    def logits_and_score(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B,3) brand logits (flip-averaged) and the open-set score (higher = more known)"""
        with torch.no_grad():
            xin = self.prep(x)
            f1 = self.feats(xin)
            f2 = self.feats(xin.flip(3))
            lg = (self.net.classifier(f1) + self.net.classifier(f2)) / 2
            if self.score_kind == 0:
                s = lg.softmax(1).max(1).values
            elif self.score_kind == 1:
                s = lg.max(1).values
            elif self.score_kind == 2:
                s = torch.logsumexp(lg, 1)
            else:
                f = (f1 + f2) / 2
                f = F.normalize(f, dim=1)
                s = (f @ self.proto.float().t()).max(1).values
                if self.score_kind == 4:
                    s = s.clamp(min=0.0) * lg.softmax(1).max(1).values
            return lg, s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lg, s = self.logits_and_score(x)
        p = lg.softmax(1)
        u = torch.sigmoid((self.thr - s) * self.sharp)           # > 0.5 exactly when the photo is rejected
        q = torch.cat([p * (1.0 - u).unsqueeze(1), u.unsqueeze(1)], 1)
        mk = q[:, :3].max(1).values                              # make argmax follow the rule in every case
        qu = torch.where(s < self.thr, torch.maximum(q[:, 3], mk + 0.01), torch.minimum(q[:, 3], mk * 0.99))
        q = torch.cat([q[:, :3], qu.unsqueeze(1)], 1)
        return q / q.sum(1, keepdim=True)

    @torch.jit.export
    def predict_index(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(1)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> List[str]:
        idx: List[int] = self.predict_index(x).tolist()
        return [self.classes[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--res", type=int, default=256); ap.add_argument("--score", default="maxlogit", choices=list(SCORES))
    ap.add_argument("--thr", type=float, required=True); ap.add_argument("--sharp", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L.RES = a.res
    net = L.build(a.arch, n_out=3)
    net.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    net.eval()
    # class prototypes of the training features: only the proto scores use them (3 x 1280 numbers)
    proto = torch.zeros(3, net.classifier[1].in_features, dtype=torch.float16)
    if a.score in ("proto", "msp_proto"):
        t = np.load(Path(__file__).resolve().parent / "cache/train.npz")
        xk = torch.from_numpy(t["x"]).permute(0, 3, 1, 2).contiguous()
        yk = torch.tensor([L.CLASSES.index(s) for s in t["y"]])
        fs = []
        with torch.no_grad():
            for i in range(0, len(xk), 64):
                xb = L.to_input(xk[i:i + 64])
                fs.append((net.avgpool(net.features(xb)).flatten(1) + net.avgpool(net.features(xb.flip(3))).flatten(1)) / 2)
        f = torch.cat(fs); f = f / f.norm(dim=1, keepdim=True)
        proto = torch.stack([f[yk == c].mean(0) for c in range(3)])
        proto = (proto / proto.norm(dim=1, keepdim=True)).half()
    sm = torch.jit.script(ShoeModel(to_small_storage(net, 16).eval(), a.score, a.thr, a.sharp, proto).eval())
    torch.jit.save(sm, a.out)
    print(a.out, f"{os.path.getsize(a.out)} bytes = {os.path.getsize(a.out)/1e6:.2f} MB | score {a.score} thr {a.thr}")


if __name__ == "__main__":
    main()
