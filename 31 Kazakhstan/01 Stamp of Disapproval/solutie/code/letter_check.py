"""Objective check of the reading: for every capital letter, render it (bold sans fonts) over the
robust bounding box of an album's region centres and score
  precision = (fraction of centres inside the glyph - ink fraction) / (1 - ink fraction)
  coverage  = fraction of the glyph skeleton with a centre closer than 10 % of the letter height
  F         = harmonic mean of the two
Reported: the top letters by F and by the excess-over-chance inside score.

    python letter_check.py boxes.csv
"""
import sys, csv, string
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

FONTS = ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
         '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
         '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf']


def glyph(ch, font_path, size=400):
    f = ImageFont.truetype(font_path, size)
    im = Image.new('L', (size * 2, size * 2), 0)
    ImageDraw.Draw(im).text((size // 2, size // 4), ch, fill=255, font=f)
    a = np.array(im) > 127
    ys, xs = np.nonzero(a)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def letter_scores(P, fonts=FONTS):
    x0, x1 = np.percentile(P[:, 0], [3, 97]); y0, y1 = np.percentile(P[:, 1], [3, 97])
    inside_score, f_score = {}, {}
    for ch in string.ascii_uppercase:
        for f in fonts:
            g = glyph(ch, f)
            gd = cv2.dilate(g.astype(np.uint8), np.ones((int(g.shape[0] * 0.04) + 1,) * 2, np.uint8)) > 0
            sy, sx = np.nonzero(skeletonize(g))
            for pad in (0.0, 0.06, 0.12):
                for sxs in (0.9, 1.0, 1.1):
                    W = (x1 - x0) * (1 + 2 * pad) * sxs; H = (y1 - y0) * (1 + 2 * pad)
                    gx = (x0 + x1) / 2 - W / 2; gy = y0 - (y1 - y0) * pad
                    u = ((P[:, 0] - gx) / W * g.shape[1]).astype(int); v = ((P[:, 1] - gy) / H * g.shape[0]).astype(int)
                    ok = (u >= 0) & (u < g.shape[1]) & (v >= 0) & (v < g.shape[0])
                    inside = np.zeros(len(P), bool); inside[ok] = gd[v[ok], u[ok]]
                    ex = inside.mean() - gd.mean()
                    prec = ex / (1 - gd.mean())
                    SX = gx + sx / g.shape[1] * W; SY = gy + sy / g.shape[0] * H
                    d = np.sqrt((SX[:, None] - P[None, :, 0]) ** 2 + (SY[:, None] - P[None, :, 1]) ** 2).min(1)
                    cov = (d < 0.1 * H).mean()
                    F = 2 * prec * cov / (prec + cov) if prec > 0 else 0.0
                    inside_score[ch] = max(inside_score.get(ch, -9), ex)
                    f_score[ch] = max(f_score.get(ch, -9), F)
    by_f = sorted(f_score.items(), key=lambda kv: -kv[1])
    by_in = sorted(inside_score.items(), key=lambda kv: -kv[1])
    return by_f, by_in


def main(boxes_csv):
    rows = list(csv.DictReader(open(boxes_csv)))
    for a in sorted(set(r['album'] for r in rows)):
        P = np.array([[int(r['x']) + int(r['w']) / 2, int(r['y']) + int(r['h']) / 2] for r in rows if r['album'] == a])
        by_f, by_in = letter_scores(P)
        print(a, '| F:', '  '.join('%s %.2f' % kv for kv in by_f[:5]), '| inside-excess:', '  '.join('%s %.2f' % kv for kv in by_in[:5]))


if __name__ == '__main__':
    main(*sys.argv[1:])
