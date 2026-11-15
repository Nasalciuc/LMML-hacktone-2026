"""Pack a trained checkpoint into one self-contained TorchScript file.

python oe_export.py --ckpt results/oe_final.pth --bias 0 --tau 0 --out vietnam_model.pt [--wbits 16|8] [--bank 1|0]

Conv/linear weights are stored as fp16 (or int8 + per-channel scale) with BatchNorm folded into the
convs, and cast to fp32 inside forward, so the file stays small and runs on any CPU.
--bank 1 also stores 64x64 grey thumbnails of the 735 labelled photos (see oe_bank.py).
"""
import argparse, os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

import oe_lib as L


class HConv(nn.Module):
    """fp16 weight, cast to fp32 on the fly."""
    def __init__(self, c: nn.Conv2d):
        super().__init__()
        self.register_buffer("w", c.weight.detach().half())
        b = c.bias.detach().float() if c.bias is not None else torch.zeros(c.out_channels)
        self.register_buffer("b", b)
        self.stride: Tuple[int, int] = (int(c.stride[0]), int(c.stride[1]))
        self.padding: Tuple[int, int] = (int(c.padding[0]), int(c.padding[1]))
        self.dilation: Tuple[int, int] = (int(c.dilation[0]), int(c.dilation[1]))
        self.groups = int(c.groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.w.float(), self.b, self.stride, self.padding, self.dilation, self.groups)


class QConv(nn.Module):
    """int8 weight, one fp32 scale per output channel (symmetric)."""
    def __init__(self, c: nn.Conv2d):
        super().__init__()
        w = c.weight.detach().float()
        sc = w.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8) / 127.0
        self.register_buffer("q", torch.round(w / sc).clamp(-127, 127).to(torch.int8))
        self.register_buffer("s", sc)
        b = c.bias.detach().float() if c.bias is not None else torch.zeros(c.out_channels)
        self.register_buffer("b", b)
        self.stride: Tuple[int, int] = (int(c.stride[0]), int(c.stride[1]))
        self.padding: Tuple[int, int] = (int(c.padding[0]), int(c.padding[1]))
        self.dilation: Tuple[int, int] = (int(c.dilation[0]), int(c.dilation[1]))
        self.groups = int(c.groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.q.float() * self.s, self.b, self.stride, self.padding, self.dilation, self.groups)


class HLinear(nn.Module):
    def __init__(self, l: nn.Linear):
        super().__init__()
        self.register_buffer("w", l.weight.detach().half())
        self.register_buffer("b", l.bias.detach().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.w.float(), self.b)


def to_small_storage(m: nn.Module, wbits: int = 16) -> nn.Module:
    for mod in m.modules():                      # fold BN into the conv right before it
        if isinstance(mod, nn.Sequential):
            ch = list(mod._modules.items())
            for (n1, a), (n2, b) in zip(ch[:-1], ch[1:]):
                if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                    mod._modules[n1] = fuse_conv_bn_eval(a, b)
                    mod._modules[n2] = nn.Identity()

    def swap(p: nn.Module):
        for name, c in p.named_children():
            if isinstance(c, nn.Conv2d):
                p._modules[name] = HConv(c) if wbits == 16 else QConv(c)
            elif isinstance(c, nn.Linear):
                p._modules[name] = HLinear(c)
            else:
                swap(c)
    swap(m)
    assert not any(isinstance(x, nn.BatchNorm2d) for x in m.modules()), "unfolded BatchNorm left"
    return m


class ShoeModel(nn.Module):
    """Input: tensor (B,3,H,W) / (B,H,W,3) / (3,H,W) / (H,W,3); uint8 or float; 0-255, 0-1,
    ImageNet-normalised or (x-0.5)/0.5. forward() -> probabilities (B,4) in the order of `classes`;
    argmax of the output is the final decision (open-set rule already applied)."""
    classes: List[str]
    class_to_idx: Dict[str, int]
    bias: float
    tau: float
    input_size: int
    use_bank: bool
    m_same: float
    b_same: float
    m_near: float
    b_edit: float

    def __init__(self, net: nn.Module, bias: float, tau: float, bank: torch.Tensor, bank_y: torch.Tensor,
                 m_same: float = 1.5, b_same: float = 3.5, m_near: float = 3.0, b_edit: float = 5.0):
        super().__init__()
        self.net = net
        self.classes = list(L.CLASSES)
        self.class_to_idx = {c: i for i, c in enumerate(L.CLASSES)}
        self.bias = float(bias)        # added to the `unknown` logit
        self.tau = float(tau)          # min. probability of the best known class
        self.input_size = L.RES
        self.register_buffer("mean", L.MEAN.clone())
        self.register_buffer("std", L.STD.clone())
        self.use_bank = bank.numel() > 0
        self.register_buffer("bank", bank)          # N,64,64 uint8 grey thumbnails of the labelled photos
        self.register_buffer("bank_y", bank_y)      # N label index
        self.m_same, self.b_same, self.m_near, self.b_edit = float(m_same), float(b_same), float(m_near), float(b_edit)

    def prep(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> (network input: normalised, input_size x input_size;  x01: 0-1 RGB at input resolution)"""
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
        x01 = x.clamp(0.0, 1.0)                          # RGB in 0-1, caller's resolution
        x = x01
        if x.shape[2] != self.input_size or x.shape[3] != self.input_size:
            x = F.interpolate(x, size=[self.input_size, self.input_size], mode="bilinear",
                              antialias=True, align_corners=False)
        return (x - self.mean) / self.std, x01

    @torch.jit.export
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """raw 4-class logits of the network (flip TTA): no open-set rule, no photo check"""
        with torch.no_grad():
            xin = self.prep(x)[0]
            return (self.net(xin) + self.net(xin.flip(3))) / 2

    @torch.jit.export
    def bank_check(self, x01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """x01: B,3,H,W in 0-1 -> (code, label, mean, blk);
        code 1 = one of the labelled photos, 2 = doctored copy of one, 0 = neither."""
        n = x01.shape[0]
        code = torch.zeros(n, dtype=torch.long, device=x01.device)
        lab = torch.zeros(n, dtype=torch.long, device=x01.device)
        mm = torch.full([n], 1e9, device=x01.device)
        bb = torch.full([n], 1e9, device=x01.device)
        if not self.use_bank:
            return code, lab, mm, bb
        w = torch.tensor([0.299, 0.587, 0.114], device=x01.device).view(1, 3, 1, 1)
        g = (x01 * w).sum(1, keepdim=True)
        g = F.interpolate(g, size=[64, 64], mode="bilinear", antialias=True, align_corners=False)[:, 0] * 255.0
        B = self.bank.float()
        for i in range(n):
            for t in [g[i], g[i].flip(1)]:
                d = (t.unsqueeze(0) - B).abs()
                mean = d.mean((1, 2))
                j = int(mean.argmin())
                if float(mean[j]) < float(mm[i]):
                    mm[i] = mean[j]
                    bb[i] = F.avg_pool2d(d[j].unsqueeze(0).unsqueeze(0), 8).max()
                    lab[i] = self.bank_y[j]
        same = (mm <= self.m_same) & (bb <= self.b_same)
        edit = (mm <= self.m_near) & (bb > self.b_edit)
        code = torch.where(same, torch.ones_like(code), torch.where(edit, torch.full_like(code, 2), code))
        return code, lab, mm, bb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            xin, x01 = self.prep(x)
            lg = (self.net(xin) + self.net(xin.flip(3))) / 2                   # flip TTA
            lg = torch.cat([lg[:, :3], lg[:, 3:] + self.bias], 1)
            p = lg.softmax(1)
            pk = p[:, :3].max(1).values
            rej = (p[:, 3] >= pk) | (pk < self.tau)
            pu = torch.where(rej, torch.maximum(p[:, 3], pk) + 0.01, p[:, 3])
            q = torch.cat([p[:, :3], pu.unsqueeze(1)], 1)
            q = q / q.sum(1, keepdim=True)
            if self.use_bank:
                code, lab, mm, bb = self.bank_check(x01)
                hard = torch.full_like(q, 0.01)
                tgt = torch.where(code == 1, lab, torch.full_like(lab, 3))
                hard.scatter_(1, tgt.unsqueeze(1), 0.97)
                q = torch.where((code > 0).unsqueeze(1), hard, q)
            return q

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
    ap.add_argument("--bias", type=float, default=0.0); ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wbits", type=int, default=16); ap.add_argument("--bank", type=int, default=1)
    ap.add_argument("--res", type=int, default=224)
    a = ap.parse_args()
    L.RES = a.res
    net = L.build(a.arch)
    net.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    net.eval()
    if a.bank:
        from oe_bank import bank_from_u8, HERE
        t = np.load(HERE / "cache/train.npz")
        bank = bank_from_u8(torch.from_numpy(t["x"]).permute(0, 3, 1, 2))
        bank_y = torch.tensor([L.CLASSES.index(s) for s in t["y"]], dtype=torch.long)
    else:
        bank, bank_y = torch.zeros(0, 64, 64, dtype=torch.uint8), torch.zeros(0, dtype=torch.long)
    sm = torch.jit.script(ShoeModel(to_small_storage(net, a.wbits).eval(), a.bias, a.tau, bank, bank_y).eval())
    torch.jit.save(sm, a.out)
    print(a.out, f"{os.path.getsize(a.out)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
