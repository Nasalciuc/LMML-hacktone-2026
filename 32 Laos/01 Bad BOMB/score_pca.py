"""Anomaly score 1: low-rank (PCA) model of the aligned shell, residual peaks.

Lighting and small pose changes are smooth and shared across images, so a
low-rank model absorbs them; a local defect is not explained by it and
shows up as a compact peak in the residual map.
"""
import sys

import cv2
import numpy as np


def interior_mask(masks):
    m = (np.mean(masks > 0, axis=0) > 0.98).astype(np.uint8)
    m = cv2.erode(m, np.ones((5, 5), np.uint8))
    m[: int(0.06 * m.shape[0])] = 0                  # nose tip: crop jitter, not defects
    return m > 0


def normalise(crops, mask):
    x = crops.astype(np.float32) / 255.0
    lab = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2Lab) for c in x])   # L in 0..100
    v = lab[:, mask, :]                                                # N,P,3
    mu = v.mean(1, keepdims=True)
    sd = v.std(1, keepdims=True) + 1e-3
    return (v - mu) / sd


def pca_residual(X, k):
    mu = X.mean(0)
    Xc = X - mu
    G = Xc @ Xc.T                                   # N x N, cheap
    w, U = np.linalg.eigh(G)
    U = U[:, ::-1][:, :k]
    R = Xc - U @ (U.T @ Xc)                          # remove top-k components
    return R


def peak_scores(R, mask, shape, sigma=2.0):
    H, W = shape
    out = []
    maps = []
    for r in R:
        e = np.zeros((H, W), np.float32)
        e[mask] = np.sqrt((r.reshape(-1, 3) ** 2).sum(1))
        e = cv2.GaussianBlur(e, (0, 0), sigma)
        maps.append(e)
        out.append(float(np.max(e)))
    return np.array(out), np.array(maps)


def run(prep_path, k=20):
    d = np.load(prep_path)
    crops, masks = d["crops"], d["masks"]
    mask = interior_mask(masks)
    V = normalise(crops, mask)
    X = V.reshape(len(V), -1)
    R = pca_residual(X, k)
    s, maps = peak_scores(R, mask, crops.shape[1:3])
    return d["names"], s, maps


if __name__ == "__main__":
    names, s, maps = run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    np.savez(sys.argv[3], names=names, score=s, maps=maps.astype(np.float16))
    o = np.argsort(-s)
    z = (s - np.median(s)) / (1.4826 * np.median(np.abs(s - np.median(s))))
    for r, i in enumerate(o[:45]):
        print(r + 1, names[i], round(float(s[i]), 3), round(float(z[i]), 1))
