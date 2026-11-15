"""Independent reading of the letters from the archive's boxes.

    python -I letters.py <archive boxes.csv> <our decisions_tiered.json> <our crops.json> <out.png>

Different fonts (Windows bold sans) and a different score than the archive's letter_check.py:
two-sided distance between the points and the letter, as a share of the letter height
  points -> ink  (mean over the best 85 % of points; a point on a stroke costs 0)
  ink -> points  (mean over the letter's ink: strokes without any point cost a lot)
Lower is better. Point sets:
  all      the 26 archive box centres of the album
  dq       only the pages the archive decided by double-quantisation evidence
  hybrid   archive centres, replaced by our own point on the pages where our content reading
           (an impossible face value) is trusted
  agreed   dq pages + pages where the archive's box is on the stamp our content reading picked
"""
import csv
import json
import string
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BOX, OURS, CROPS, OUT = sys.argv[1:5]
FONTS = [f"C:/Windows/Fonts/{n}.ttf" for n in ("arialbd", "verdanab", "tahomabd", "segoeuib")]
GH, PAD = 200, 120


def glyph(ch, font):
    f = ImageFont.truetype(font, 400)
    im = Image.new("L", (900, 900), 0)
    ImageDraw.Draw(im).text((200, 100), ch, fill=255, font=f)
    a = np.array(im) > 127
    ys, xs = np.nonzero(a)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    w = max(8, int(round(a.shape[1] * GH / a.shape[0])))
    a = cv2.resize(a.astype(np.uint8), (w, GH), interpolation=cv2.INTER_AREA) > 0
    canvas = np.zeros((GH + 2 * PAD, w + 2 * PAD), np.uint8)
    canvas[PAD:PAD + GH, PAD:PAD + w] = a
    dist = cv2.distanceTransform(1 - canvas, cv2.DIST_L2, 5)  # distance to ink, glyph px
    iy, ix = np.nonzero(a[::6, ::6])
    return a, dist, np.stack([ix * 6.0, iy * 6.0], 1), w


GLYPHS = {(ch, f): glyph(ch, f) for ch in string.ascii_uppercase for f in FONTS}


def fit(P):
    x0, x1 = np.percentile(P[:, 0], [5, 95])
    y0, y1 = np.percentile(P[:, 1], [5, 95])
    cx, cy, bh = (x0 + x1) / 2, (y0 + y1) / 2, max(y1 - y0, 60.0)
    keep = max(3, int(round(0.85 * len(P))))
    best = {}
    for (ch, f), (a, dist, ink, gw) in GLYPHS.items():
        for s in (0.9, 1.0, 1.1, 1.2, 1.3):
            H = bh * s
            for asp in (0.8, 0.9, 1.0, 1.1, 1.25):
                W = H * gw / GH * asp
                for dx in (-0.1, -0.05, 0, 0.05, 0.1):
                    for dy in (-0.1, -0.05, 0, 0.05, 0.1):
                        gx, gy = cx + dx * H - W / 2, cy + dy * H - H / 2
                        u = (P[:, 0] - gx) / W * gw + PAD
                        v = (P[:, 1] - gy) / H * GH + PAD
                        ui = np.clip(u, 0, dist.shape[1] - 1).astype(int)
                        vi = np.clip(v, 0, dist.shape[0] - 1).astype(int)
                        d1 = dist[vi, ui] / GH + (np.abs(u - ui) + np.abs(v - vi)) / GH
                        d1 = np.sort(d1)[:keep].mean()
                        IX = gx + ink[:, 0] / gw * W
                        IY = gy + ink[:, 1] / GH * H
                        d2 = np.sqrt((IX[:, None] - P[None, :, 0]) ** 2 + (IY[:, None] - P[None, :, 1]) ** 2).min(1).mean() / H
                        sc = d1 + d2
                        if sc < best.get(ch, (9,))[0]:
                            best[ch] = (sc, f, gx, gy, W, H)
    return sorted(best.items(), key=lambda kv: kv[1][0])


rows = list(csv.DictReader(open(BOX)))
ours = {(d["album"], d["page"]): d for d in json.load(open(OURS))}
crops = {c["crop"][:-4]: c["box"] for c in json.load(open(CROPS))}
albums = sorted({r["album"] for r in rows})
panels = []
word = {}
for a in albums:
    sets = {"all": [], "dq": [], "hybrid": [], "agreed": []}
    draw = []
    for r in rows:
        if r["album"] != a:
            continue
        c = (int(r["x"]) + int(r["w"]) / 2, int(r["y"]) + int(r["h"]) / 2)
        o = ours[(a, r["page"])]
        trusted = o["why"] == "content"
        sx, sy, sw, sh = crops[o["stamp"]]
        same = sx <= c[0] <= sx + sw and sy <= c[1] <= sy + sh
        sets["all"].append(c)
        if r["source"] == "dq":
            sets["dq"].append(c)
        sets["hybrid"].append((o["cx"], o["cy"]) if trusted else c)
        if r["source"] == "dq" or (trusted and same):
            sets["agreed"].append(c)
        draw.append((c, r["source"], (o["cx"], o["cy"]) if trusted else None, same))
    print(f"\n{a}")
    for name, pts in sets.items():
        P = np.array(pts, float)
        top = fit(P)
        word.setdefault(name, []).append(top[0][0])
        print(f"  {name:7s} n={len(P):2d}  " + "  ".join(f"{ch} {v[0]:.3f}" for ch, v in top[:6]))
        if name == "all":
            best_all = top[0]
    # panel: archive centres (black = dq, grey = classifier), our trusted points (red ring), best glyph behind
    img = np.full((1200, 1600, 3), 255, np.uint8)
    ch, (sc, f, gx, gy, W, H) = best_all
    g = GLYPHS[(ch, f)][0].astype(np.uint8)
    gg = cv2.resize(g, (max(2, int(W)), max(2, int(H))), interpolation=cv2.INTER_NEAREST) > 0
    X0, Y0 = int(gx), int(gy)
    ys, xs = np.nonzero(gg)
    ok = (ys + Y0 >= 0) & (ys + Y0 < 1200) & (xs + X0 >= 0) & (xs + X0 < 1600)
    img[ys[ok] + Y0, xs[ok] + X0] = (215, 232, 255)
    for c, src, o, same in draw:
        col = (0, 0, 0) if src == "dq" else (140, 140, 140)
        cv2.circle(img, (int(c[0]), int(c[1])), 16, col, -1)
        if o is not None:
            cv2.circle(img, (int(o[0]), int(o[1])), 24, (0, 0, 220), 5)
            if not same:
                cv2.line(img, (int(c[0]), int(c[1])), (int(o[0]), int(o[1])), (0, 0, 220), 3)
    cv2.putText(img, f"{a}: {ch}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 5)
    panels.append(cv2.resize(img, (560, 420), interpolation=cv2.INTER_AREA))
print("\nbest letter per album, by point set:")
for name, w in word.items():
    print(f"  {name:7s} {''.join(w)}")
cv2.imwrite(OUT, np.hstack([np.pad(p, ((3, 3), (3, 3), (0, 0)), constant_values=110) for p in panels]))
print("saved", OUT)
