"""Torch implementation of the resize-trace features (must match the numpy version used for fitting)."""
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import torch

P = 16


def fold_matrix(n=256, period=P):
    # second-derivative output index j corresponds to pixel position j+1, j in [0, n-3]
    pos = torch.arange(1, n - 1)
    M = torch.zeros(n - 2, period, dtype=torch.float64)
    M[torch.arange(n - 2), pos % period] = 1.0
    M = M / M.sum(0, keepdim=True)
    return M.float()


def gray_255(x01):
    """x01: float [N,3,H,W] in [0,1] -> PIL-like luma in [0,255] (float, not rounded)."""
    r, g, b = x01[:, 0] * 255.0, x01[:, 1] * 255.0, x01[:, 2] * 255.0
    return r * 0.299 + g * 0.587 + b * 0.114


def folded_profiles(gray, M):
    """gray: [N,H,W] float -> [N, 2*P] z-normalised folded |d2| profiles (x then y)."""
    d2x = (gray[:, :, 2:] - 2.0 * gray[:, :, 1:-1] + gray[:, :, :-2]).abs().mean(1)  # [N, W-2]
    d2y = (gray[:, 2:, :] - 2.0 * gray[:, 1:-1, :] + gray[:, :-2, :]).abs().mean(2)  # [N, H-2]
    fx = d2x @ M
    fy = d2y @ M
    fx = (fx - fx.mean(1, keepdim=True)) / (fx.std(1, unbiased=False, keepdim=True) + 1e-9)
    fy = (fy - fy.mean(1, keepdim=True)) / (fy.std(1, unbiased=False, keepdim=True) + 1e-9)
    return torch.cat([fx, fy], 1)


def _zn(f):
    return (f - f.mean(1, keepdim=True)) / (f.std(1, unbiased=False, keepdim=True) + 1e-9)


def trace_features(x01, M):
    """x01: float [N,3,256,256] in [0,1] -> [N,128] phase-folded statistics (period 16)."""
    g = (x01[:, 0] * 0.299 + x01[:, 1] * 0.587 + x01[:, 2] * 0.114) * 255.0
    d2x = g[:, :, 2:] - 2.0 * g[:, :, 1:-1] + g[:, :, :-2]
    d2y = g[:, 2:, :] - 2.0 * g[:, 1:-1, :] + g[:, :-2, :]
    ax, ay = d2x.abs(), d2y.abs()
    f1 = torch.cat([_zn(ax.mean(1) @ M), _zn(ay.mean(2) @ M)], 1)
    f2 = torch.cat([_zn((d2x * d2x).mean(1) @ M), _zn((d2y * d2y).mean(2) @ M)], 1)
    d1x = (g[:, :, 1:] - g[:, :, :-1]).abs().mean(1)[:, :254]
    d1y = (g[:, 1:, :] - g[:, :-1, :]).abs().mean(2)[:, :254]
    f3 = torch.cat([_zn(d1x @ M), _zn(d1y @ M)], 1)
    rx = ax / (ax.mean(2, keepdim=True) + 1.0)
    ry = ay / (ay.mean(1, keepdim=True) + 1.0)
    f4 = torch.cat([_zn(rx.mean(1) @ M), _zn(ry.mean(2) @ M)], 1)
    return torch.cat([f1, f2, f3, f4], 1)
