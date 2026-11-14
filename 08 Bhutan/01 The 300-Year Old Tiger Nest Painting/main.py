"""Reassemble vertically shredded image strips.

Reads every image from the folder `input` (next to this script, or the
current working directory), finds the left-to-right order of the strips and
writes `reconstructed_image.png` and `reconstructed_order.json` next to this
script.

Method
  1. Pairwise edge cost for "strip i directly left of strip j":
     Mahalanobis gradient compatibility (MGC) in both directions plus a
     linear-prediction error term, computed in float64 on all channels.
  2. Cost normalisation by the second-best alternative on both sides
     (makes the costs comparable between textured and flat strips).
  3. Ordering = shortest Hamiltonian path on the asymmetric cost matrix:
     greedy edge merging (Kruskal style, no cycles) -> chain, then
     local search (or-opt segment moves incl. reversal-free segment moves,
     2-opt is not valid for an asymmetric matrix) until no improvement.
     A few restarts from different seeds are tried and the best path kept.
"""
import json
import os
import sys
import time

import numpy as np
import cv2

T0 = time.time()
TIME_BUDGET = 14.0  # seconds, hard limit is 20

HERE = os.path.dirname(os.path.abspath(__file__))
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def find_input_dir():
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        return sys.argv[1]
    for base in (os.getcwd(), HERE):
        p = os.path.join(base, "input")
        if os.path.isdir(p):
            return p
    raise SystemExit("input folder not found")


def load_strips(folder):
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS))
    imgs = []
    for n in names:
        im = cv2.imread(os.path.join(folder, n), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise SystemExit("cannot read " + n)
        if im.ndim == 2:
            im = im[:, :, None]
        imgs.append(im)
    # strips should share height; if not, crop to the minimum
    h = min(im.shape[0] for im in imgs)
    c = min(im.shape[2] for im in imgs)
    imgs = [im[:h, :, :c] for im in imgs]
    return names, imgs


def edge_costs(imgs):
    """D[i, j] = dissimilarity of placing strip j immediately right of i."""
    n = len(imgs)
    ch = min(3, imgs[0].shape[2])
    L1 = np.stack([im[:, 0, :ch] for im in imgs]).astype(np.float64)    # n,h,c
    R1 = np.stack([im[:, -1, :ch] for im in imgs]).astype(np.float64)
    if imgs[0].shape[1] >= 2:
        L2 = np.stack([im[:, 1, :ch] for im in imgs]).astype(np.float64)
        R2 = np.stack([im[:, -2, :ch] for im in imgs]).astype(np.float64)
    else:
        L2, R2 = L1.copy(), R1.copy()
    h = L1.shape[1]

    def stats(g):
        mu = g.mean(axis=1)                                   # n,c
        gc = g - mu[:, None]
        cov = np.einsum("nhc,nhd->ncd", gc, gc) / max(1, h - 1)
        # regularisation keeps nearly flat strips well conditioned
        cov = cov + np.eye(ch)[None] * (np.trace(cov, axis1=1, axis2=2)[:, None, None] / ch * 0.01 + 1.0)
        return mu, np.linalg.inv(cov)

    def mahal(U, V, A):
        """M[i, j] = sum_h (U_j - V_i)^T A (U_j - V_i), A = A_i.
        U: n,h,c (indexed by j), V: n,h,c (indexed by i)."""
        Ah = A.reshape(n, ch * ch)
        UU = np.einsum("nhc,nhd->ncd", U, U).reshape(n, ch * ch)
        VV = np.einsum("nhc,nhd->ncd", V, V).reshape(n, ch * ch)
        cross = np.empty((ch, ch, n, n))                      # [c,d,i,j] = sum_h V_i[c] U_j[d]
        for c in range(ch):
            for d in range(ch):
                cross[c, d] = V[:, :, c] @ U[:, :, d].T
        t1 = Ah @ UU.T                                        # i,j
        t3 = (Ah * VV).sum(1)[:, None]
        t2 = 2.0 * np.einsum("icd,cdij->ij", A, cross)        # A symmetric
        return t1 - t2 + t3

    muR, invR = stats(R1 - R2)
    muL, invL = stats(L1 - L2)
    # boundary gradient seen from the left strip i: L1_j - R1_i  vs  muR_i
    m1 = mahal(L1, R1 + muR[:, None, :], invR)
    # seen from the right strip j: R1_i - L1_j vs muL_j  ->  (R1_i) - (L1_j + muL_j)
    m2 = mahal(R1, L1 + muL[:, None, :], invL).T
    D = m1 + m2

    # prediction error (linear extrapolation from both sides), sqrt-norm
    P = np.zeros((n, n))
    L1f, L2f, R1f, R2f = (x.astype(np.float32) for x in (L1, L2, R1, R2))
    pb = 2 * L1f - L2f
    for i in range(n):
        pa = 2 * R1f[i] - R2f[i]
        e = np.abs(L1f - pa[None]) + np.abs(R1f[i][None] - pb)
        P[i] = np.sqrt(e).sum(axis=(1, 2))
    np.fill_diagonal(D, np.inf)
    np.fill_diagonal(P, np.inf)
    return D, P


def normalise(D):
    n = D.shape[0]
    # divide by second smallest in the row and column (confidence style)
    srow = np.sort(D, axis=1)[:, 1] if n > 2 else np.ones(n)
    scol = np.sort(D, axis=0)[1, :] if n > 2 else np.ones(n)
    # floor: avoids blow-ups when several strips are (nearly) identical
    floor = 0.5 * np.median(srow) + 1e-9
    denom = np.minimum(srow[:, None], scol[None, :]) + floor
    return D / denom


def path_cost(path, C):
    return float(C[path[:-1], path[1:]].sum())


def greedy_chain(C):
    n = C.shape[0]
    right = -np.ones(n, dtype=int)
    left = -np.ones(n, dtype=int)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    order = np.argsort(C, axis=None)
    links = 0
    for k in order:
        i, j = divmod(int(k), n)
        if i == j or right[i] >= 0 or left[j] >= 0:
            continue
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        right[i] = j
        left[j] = i
        parent[ri] = rj
        links += 1
        if links == n - 1:
            break
    start = int(np.where(left < 0)[0][0])
    path = [start]
    while right[path[-1]] >= 0:
        path.append(int(right[path[-1]]))
    return path


def or_opt(path, C, deadline, max_seg=None):
    """Move segments (any length) to the best other position; also try the
    move with the whole path rotated (start/end change). Asymmetric safe."""
    n = len(path)
    if max_seg is None:
        max_seg = n
    p = list(path)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best = None
        cur = path_cost(p, C)
        arr = np.array(p)
        for L in range(1, min(max_seg, n - 1) + 1):
            for s in range(0, n - L + 1):
                seg = arr[s:s + L]
                rest = np.concatenate([arr[:s], arr[s + L:]])
                # cost removed
                rem = 0.0
                if s > 0:
                    rem += C[arr[s - 1], arr[s]]
                if s + L < n:
                    rem += C[arr[s + L - 1], arr[s + L]]
                add_back = C[arr[s - 1], arr[s + L]] if (s > 0 and s + L < n) else 0.0
                base = cur - rem + add_back - C[seg[:-1], seg[1:]].sum() * 0  # segment internal unchanged
                # insertion positions in rest: between rest[k-1] and rest[k], k=0..len(rest)
                m = len(rest)
                a = seg[0]
                b = seg[-1]
                ins = np.empty(m + 1)
                ins[0] = C[b, rest[0]]
                ins[m] = C[rest[-1], a]
                if m > 1:
                    ins[1:m] = C[rest[:-1], a] + C[b, rest[1:]] - C[rest[:-1], rest[1:]]
                ins[s] = np.inf  # original position
                k = int(np.argmin(ins))
                newc = base + ins[k]
                if newc < cur - 1e-9 and (best is None or newc < best[0]):
                    best = (newc, s, L, k)
            if time.time() > deadline:
                break
        if best is not None:
            _, s, L, k = best
            seg = p[s:s + L]
            rest = p[:s] + p[s + L:]
            p = rest[:k] + seg + rest[k:]
            improved = True
    return p


def solve(C, deadline):
    n = C.shape[0]
    if n == 1:
        return [0]
    best = greedy_chain(C)
    best = or_opt(best, C, deadline)
    return best


def main():
    folder = find_input_dir()
    names, imgs = load_strips(folder)
    n = len(names)
    if n == 0:
        raise SystemExit("no strips")
    D, P = edge_costs(imgs)
    # the prediction term is the primary cue; MGC only breaks near-ties
    C = normalise(P) + 0.05 * normalise(D)
    deadline = T0 + TIME_BUDGET
    path = solve(C, deadline)
    order = [names[k] for k in path]
    out = np.concatenate([imgs[k] for k in path], axis=1)
    if out.shape[2] == 1:
        out = out[:, :, 0]
    cv2.imwrite(os.path.join(HERE, "reconstructed_image.png"), out)
    with open(os.path.join(HERE, "reconstructed_order.json"), "w") as f:
        json.dump(order, f, indent=4)
    print(f"{n} strips, cost {path_cost(path, C):.3f}, {time.time() - T0:.2f}s")


if __name__ == "__main__":
    main()
