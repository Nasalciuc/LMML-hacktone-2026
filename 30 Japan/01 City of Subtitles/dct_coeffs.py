"""Recover the quantised luma DCT coefficients of a JPEG from its decoded Y plane (decoder kept in YCbCr),
to look at the coefficient LSBs (JSteg-style hidden data survives only there)."""
import sys

import numpy as np
from PIL import Image
from scipy.fft import dctn

ZIGZAG = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,
                   6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45,
                   38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])


def coeffs(path):
    im = Image.open(path)
    qn = np.array(im.quantization[0], dtype=np.float64).reshape(8, 8)  # Pillow gives natural (row-major) order
    im.draft("YCbCr", im.size)
    y = np.asarray(im, dtype=np.float64)[..., 0] - 128.0
    H, W = y.shape
    blocks = y.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3)
    c = dctn(blocks, type=2, axes=(2, 3), norm="ortho")
    cq = c / qn
    return cq, qn


if __name__ == "__main__":
    for p in sys.argv[1:]:
        cq, qn = coeffs(p)
        fr = cq - np.round(cq)
        print(p, "quant(natural) first row", qn[0], "rounding error |frac| mean", np.abs(fr).mean().round(3),
              "share |frac|<0.15", (np.abs(fr) < 0.15).mean().round(3))
