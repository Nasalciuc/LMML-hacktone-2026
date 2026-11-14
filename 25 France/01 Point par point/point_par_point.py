# Point par point
# python point_par_point.py [paris_listings.csv] [out_dir]
import os
import string
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "task_france", "paris_listings.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else HERE
FONT = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
LAT_KM, LON_KM = 110.574, 111.320 * np.cos(np.radians(48.86))
K, EPS, MIN_PTS, MIN_LETTER, GROW = 10, 0.25, 6, 150, 0.30

df = pd.read_csv(CSV)
f = pd.DataFrame({
    "lat": df.latitude * LAT_KM,
    "lon": df.longitude * LON_KM,
    "price": np.log(df.price),
    "avail": df.availability_365 / 365,
    "rpm": np.log(df.reviews_per_month),
})
z = ((f - f.mean()) / f.std()).values

# distance to the 10th neighbour in (map + colour) space: two modes
d = NearestNeighbors(n_neighbors=K + 1).fit(z).kneighbors(z)[0][:, K]
logd = np.log(d)
h, e = np.histogram(logd, bins=160, range=(-3, 1))
c = (e[:-1] + e[1:]) / 2
hs = np.convolve(h, np.ones(5) / 5, mode="same")
i_main = int(np.argmax(hs))
i_left = int(np.argmax(np.where(c < c[i_main] - 0.6, hs, -1)))
thr = c[i_left + int(np.argmin(hs[i_left:i_main]))]
cand = np.where(logd < thr)[0]
print(f"modes {c[i_left]:.2f} / {c[i_main]:.2f}, valley {thr:.2f}, {len(cand)} candidates")

lab = DBSCAN(eps=EPS, min_samples=MIN_PTS).fit_predict(z[cand])
sizes = pd.Series(lab[lab >= 0]).value_counts()
ids = [k for k in sizes.index if sizes[k] >= MIN_LETTER]
print(f"{len(sizes)} clusters, {len(ids)} big ones, small: {sizes[sizes < MIN_LETTER].to_dict()}")
groups = {k: cand[lab == k] for k in ids}


def plane_dist(v):
    x = z[v]
    mu = x.mean(0)
    b = np.linalg.svd(x - mu, full_matrices=False)[2][:2]
    r = lambda y: np.linalg.norm(y - (y @ b.T) @ b, axis=1)
    return r(z - mu), r(x - mu)


# sparse parts of a letter sit just above the cut: add points close to its sheet
pool = logd < thr + GROW
for _ in range(5):
    taken = np.zeros(len(z), bool)
    for v in groups.values():
        taken[v] = True
    added = 0
    for k, v in groups.items():
        r_all, r_in = plane_dist(v)
        near = cKDTree(z[v]).query(z)[0] < EPS
        new = np.where(pool & ~taken & near & (r_all < 3 * np.median(r_in)))[0]
        if len(new):
            groups[k] = np.array(sorted(set(v) | set(new)))
            taken[new] = True
            added += len(new)
    if not added:
        break


def norm(p):
    p = p - p.mean(0)
    return p / np.sqrt((p ** 2).sum(1).mean())


def glyph(ch, n=400):
    img = Image.new("L", (300, 300), 0)
    ImageDraw.Draw(img).text((40, 10), ch, fill=255, font=ImageFont.truetype(FONT, 220))
    m = np.array(img) > 128
    try:
        from skimage.morphology import skeletonize
        m = skeletonize(m)
    except ImportError:
        pass
    yy, xx = np.nonzero(m)
    p = np.c_[xx, -yy].astype(float)
    if len(p) > n:
        p = p[np.random.default_rng(0).choice(len(p), n, replace=False)]
    return norm(p)


def icp(src, g, a, iters=35):
    # affine ICP, singular values kept >= half the largest so a shape can't collapse
    t = np.zeros(2)
    tg = cKDTree(g)
    for _ in range(iters):
        q = src @ a.T + t
        i1 = tg.query(q)[1]
        i2 = cKDTree(q).query(g)[1]
        s = np.r_[src, src[i2]]
        dd = np.r_[g[i1], g]
        sol = np.linalg.lstsq(np.c_[s, np.ones(len(s))], dd, rcond=None)[0]
        u, sv, vt = np.linalg.svd(sol[:2].T)
        a = u @ np.diag(np.maximum(sv, 0.5 * sv.max())) @ vt
        t = (dd - s @ a.T).mean(0)
    q = src @ a.T + t
    return 0.5 * (tg.query(q)[0].mean() + cKDTree(q).query(g)[0].mean()), a, t


def fit(src, g):
    best = None
    for refl in (1, -1):
        for th in np.radians(range(0, 360, 30)):
            rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            r = icp(src, g, rot @ np.diag([1.0, refl]))
            if best is None or r[0] < best[0]:
                best = r
    return best


def shape(v):
    # availability pinned at 0 or 365 folds the sheet, leave it out then
    av = df.availability_365.values[v]
    cols = [0, 1, 2, 4] if np.mean((av <= 10) | (av >= 355)) > 0.1 else [0, 1, 2, 3, 4]
    x = z[v][:, cols]
    x = x - x.mean(0)
    return norm(x @ np.linalg.svd(x, full_matrices=False)[2][:2].T)


# reading order = colour order: median nightly price
order = sorted(ids, key=lambda k: np.median(df.price.values[groups[k]]))
G = {ch: glyph(ch) for ch in string.ascii_uppercase}
res = []
for pos, k in enumerate(order, 1):
    s = shape(groups[k])
    fits = {ch: fit(s, g) for ch, g in G.items()}
    rank = sorted(fits, key=lambda ch: fits[ch][0])
    res.append((k, s, fits, rank))
    print(pos, len(groups[k]), f"{np.median(df.price.values[groups[k]]):.0f}",
          " ".join(f"{ch}={fits[ch][0]:.3f}" for ch in rank[:3]))

# N and Z are the same shape turned 90 degrees, keep near-ties
readings = [""]
for _, _, fits, rank in res:
    best = fits[rank[0]][0]
    readings = [w + ch for w in readings for ch in rank[:3] if fits[ch][0] <= best * 1.05]
print("readings:", ", ".join(readings))
WORD = "NEURON"  # the only real word among the readings
assert WORD in readings

pos = np.zeros(len(df), int)
for i, (k, *_) in enumerate(res, 1):
    pos[groups[k]] = i
letters = np.array(["-"] + list(WORD))[pos]
pd.DataFrame({
    "listing_id": df.listing_id,
    "cluster": pos - 1,
    "letter": letters,
    "label": [f"{l}{p}" if p else "background" for l, p in zip(letters, pos)],
}).to_csv(os.path.join(OUT, "labels.csv"), index=False)

cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(3, 6, height_ratios=[1.35, 1, 1])
ax = fig.add_subplot(gs[0, :4])
ax.scatter(df.longitude, df.latitude, s=0.4, c="#c8c8c8", linewidths=0)
for i, ((k, *_), col) in enumerate(zip(res, cols)):
    sub = df.iloc[groups[k]]
    ax.scatter(sub.longitude, sub.latitude, s=4, c=col, label=f"{i + 1}: {WORD[i]}")
ax.set_aspect(1 / np.cos(np.radians(48.86)))
ax.legend(markerscale=4)
ax.set_title("listings (grey) and the extra dots")
ax = fig.add_subplot(gs[0, 4:])
ax.hist(logd, bins=160, range=(-3, 1), color="#888")
ax.axvline(thr, color="r", ls="--")
ax.set_yscale("log")
ax.set_xlabel("log distance to 10th neighbour (lat, lon, price, availability, reviews/month)")
for i, ((k, s, fits, _), col) in enumerate(zip(res, cols)):
    _, a, t = fits[WORD[i]]
    for row, pts in ((1, s), (2, s @ a.T + t)):
        ax = fig.add_subplot(gs[row, i])
        ax.scatter(pts[:, 0], pts[:, 1], s=5, c=col)
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(f"{WORD[i]}  ({np.median(df.price.values[groups[k]]):.0f} EUR)", fontsize=16)
fig.suptitle(f"read cheap to expensive: {WORD}", fontsize=18)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "letters.png"), dpi=90)
print(f"SIGMOID_{WORD}")
