"""PRNU tools for City of Subtitles: residual extraction, caption masking, fingerprints, PCE/NCC."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, signal

DATA = Path(__file__).resolve().parents[1] / "task_japan" / "dataset"
CACHE = Path(__file__).resolve().parent / "cache"


def list_photos(data: Path = DATA):
    credited = {p.name: sorted(p.glob("*.jpg")) for p in sorted((data / "credited").iterdir()) if p.is_dir()}
    anonymous = sorted((data / "anonymous").glob("*.jpg"))
    return credited, anonymous


def load_rgb(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def caption_mask(rgb: np.ndarray, dark_thr: float = 22.0, pad: int = 12) -> np.ndarray:
    """True where the pixel is usable. Masks the black caption band(s) and any text inside them,
    plus clipped pixels. A row counts as band if most of it is near black."""
    g = rgb.mean(2)
    h, w = g.shape
    dark_frac = (g < dark_thr).mean(1)
    band = dark_frac > 0.6
    # close small gaps (text lines make rows partly bright)
    band = ndimage.binary_closing(band, structure=np.ones(25))
    band = ndimage.binary_dilation(band, structure=np.ones(2 * pad + 1))
    # bands only count if they touch the top or bottom border region and are long enough
    lab, n = ndimage.label(band)
    keep = np.zeros(h, bool)
    for i in range(1, n + 1):
        idx = np.where(lab == i)[0]
        if len(idx) >= 20:
            keep[idx] = True
    m = np.ones((h, w), bool)
    m[keep, :] = False
    # clipped / saturated pixels carry no PRNU
    m &= ~(rgb.max(2) > 250)
    m &= ~(g < 3)
    return m


def zero_mean(x: np.ndarray) -> np.ndarray:
    """Remove row and column means (kills demosaicing / JPEG periodic artefacts shared by models)."""
    x = x - x.mean(1, keepdims=True)
    x = x - x.mean(0, keepdims=True)
    return x


def wiener_dft(x: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Flatten the spectrum of a noise pattern (removes periodic peaks such as the JPEG 8x8 grid)."""
    F = np.fft.fft2(x)
    if sigma is None:
        sigma = np.std(x)
    # attenuate spectral components whose local power exceeds the white-noise level
    pw = ndimage.uniform_filter(np.abs(F) ** 2 / x.size, 3)
    gain = sigma ** 2 / np.maximum(pw, sigma ** 2)
    return np.real(np.fft.ifft2(F * gain))


def denoise(ch: np.ndarray, method: str = "wiener") -> np.ndarray:
    if method == "gauss":
        return ndimage.gaussian_filter(ch, 1.0)
    if method == "median":
        return ndimage.median_filter(ch, 3)
    if method == "wiener":
        return signal.wiener(ch, (5, 5))
    if method == "wiener3":
        return signal.wiener(ch, (3, 3))
    raise ValueError(method)


def residual(rgb: np.ndarray, method: str = "wiener", channels: str = "rgb") -> tuple[np.ndarray, np.ndarray]:
    """Returns (noise residual W, intensity I) as single 2-D arrays (per-channel residuals combined to luma)."""
    ws = []
    for c in range(3):
        ch = rgb[..., c]
        ws.append(ch - denoise(ch, method))
    w = 0.299 * ws[0] + 0.587 * ws[1] + 0.114 * ws[2]
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return w, lum


def extract(path, method="wiener"):
    rgb = load_rgb(path)
    m = caption_mask(rgb)
    w, lum = residual(rgb, method)
    # suppress strong edges (scene content leaking into the residual)
    edge = np.abs(w) > 4 * np.median(np.abs(w)) * 1.4826 * 3
    m = m & ~ndimage.binary_dilation(edge, iterations=1)
    w = np.where(m, w, 0.0)
    w = zero_mean(w) * m
    return w.astype(np.float32), lum.astype(np.float32), m


def fingerprint(items):
    """MLE fingerprint K = sum(W*I) / sum(I^2) over the usable pixels."""
    num = None
    den = None
    for w, lum, m in items:
        lm = lum * m
        num = w * lm if num is None else num + w * lm
        den = lm * lm if den is None else den + lm * lm
    k = num / np.maximum(den, 1e-6)
    k = zero_mean(k)
    k = wiener_dft(k)
    return k.astype(np.float32), den


def ncc(a, b, m=None):
    if m is not None:
        a = a[m]
        b = b[m]
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def pce(a, b, peak_excl: int = 5):
    """Peak-to-correlation-energy of circular cross-correlation, peak taken at zero shift."""
    a = a - a.mean()
    b = b - b.mean()
    xc = np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))))
    peak = xc[0, 0]
    h, w = xc.shape
    sl = np.ones_like(xc, bool)
    for dy in range(-peak_excl, peak_excl + 1):
        for dx in range(-peak_excl, peak_excl + 1):
            sl[dy % h, dx % w] = False
    energy = np.mean(xc[sl] ** 2)
    return float(np.sign(peak) * peak ** 2 / energy), xc


# ----------------------------------------------------------------------------- GPU helpers
def gblur(x, s):
    """Separable gaussian blur of an N,C,H,W torch tensor."""
    import torch
    import torch.nn.functional as F
    k = int(3 * s) * 2 + 1
    t = torch.arange(k, device=x.device, dtype=torch.float32) - k // 2
    g = torch.exp(-t ** 2 / (2 * s * s))
    g = g / g.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)
    return x
