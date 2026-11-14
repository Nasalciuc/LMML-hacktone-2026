"""Wavelet (db4, periodic) denoiser in the style of Mihcak et al., as used for PRNU extraction. Torch, GPU."""
import math

import torch
import torch.nn.functional as F

DB4 = [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114,
       -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523]


def _filters(dev):
    lo = torch.tensor(DB4[::-1], device=dev, dtype=torch.float64)  # analysis lowpass (correlation form)
    n = len(lo)
    hi = torch.tensor([(-1) ** k * DB4[k] for k in range(n)], device=dev, dtype=torch.float64)
    return lo, hi


def _analysis_1d(x, f, dim):
    """periodic correlation with f then downsample by 2 along dim (x: N,1,H,W)."""
    n = f.numel()
    if dim == 3:
        xp = torch.cat([x, x[..., : n - 1]], 3)
        y = F.conv2d(xp, f.view(1, 1, 1, n).to(x.dtype))
        return y[..., ::2]
    xp = torch.cat([x, x[..., : n - 1, :]], 2)
    y = F.conv2d(xp, f.view(1, 1, n, 1).to(x.dtype))
    return y[..., ::2, :]


def _synthesis_1d(y, f, dim):
    """upsample by 2 (zeros) then periodic convolution: adjoint of _analysis_1d."""
    n = f.numel()
    if dim == 3:
        N, C, H, W = y.shape
        u = torch.zeros(N, C, H, W * 2, dtype=y.dtype, device=y.device)
        u[..., ::2] = y
        up = torch.cat([u[..., -(n - 1):], u], 3)
        return F.conv2d(up, f.flip(0).view(1, 1, 1, n).to(y.dtype))
    N, C, H, W = y.shape
    u = torch.zeros(N, C, H * 2, W, dtype=y.dtype, device=y.device)
    u[..., ::2, :] = y
    up = torch.cat([u[..., -(n - 1):, :], u], 2)
    return F.conv2d(up, f.flip(0).view(1, 1, n, 1).to(y.dtype))


def dwt2(x, lo, hi):
    L = _analysis_1d(x, lo, 3)
    H = _analysis_1d(x, hi, 3)
    LL = _analysis_1d(L, lo, 2)
    LH = _analysis_1d(L, hi, 2)
    HL = _analysis_1d(H, lo, 2)
    HH = _analysis_1d(H, hi, 2)
    return LL, (LH, HL, HH)


def idwt2(LL, det, lo, hi):
    LH, HL, HH = det
    L = _synthesis_1d(LL, lo, 2) + _synthesis_1d(LH, hi, 2)
    H = _synthesis_1d(HL, lo, 2) + _synthesis_1d(HH, hi, 2)
    return _synthesis_1d(L, lo, 3) + _synthesis_1d(H, hi, 3)


def _wiener_coef(c, sigma2, windows=(3, 5, 7, 9)):
    v = None
    for w in windows:
        m = F.avg_pool2d(F.pad(c * c, (w // 2,) * 4, mode="circular"), w, 1)
        e = (m - sigma2).clamp(min=0)
        v = e if v is None else torch.minimum(v, e)
    return c * v / (v + sigma2)


def denoise(x, sigma=3.0, levels=4):
    """x: N,1,H,W (H, W divisible by 2**levels). Returns denoised image."""
    dev = x.device
    lo, hi = _filters(dev)
    xd = x.double()
    coeffs = []
    cur = xd
    for _ in range(levels):
        cur, det = dwt2(cur, lo, hi)
        coeffs.append(det)
    s2 = sigma ** 2
    out = cur
    for det in reversed(coeffs):
        det = tuple(_wiener_coef(c, s2) for c in det)
        out = idwt2(out, det, lo, hi)
    return out.to(x.dtype)


def residual(x, sigma=3.0, levels=4):
    n, c, h, w = x.shape
    y = x.reshape(n * c, 1, h, w)
    return (y - denoise(y, sigma, levels)).reshape(n, c, h, w)


if __name__ == "__main__":
    x = torch.randn(1, 1, 960, 1280, dtype=torch.float64)
    lo, hi = _filters("cpu")
    LL, det = dwt2(x, lo, hi)
    r = idwt2(LL, det, lo, hi)
    print("reconstruction error", float((r - x).abs().max()))
