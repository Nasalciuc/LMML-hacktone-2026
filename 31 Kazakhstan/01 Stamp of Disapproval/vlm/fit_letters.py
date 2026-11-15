"""Try to read each album sheet as a letter: fit A-Z stroke templates to the confident dots.

Dots: strong content picks (weight 1) and year-missing picks (weight 0.6) from
decisions_tiered.json. Templates: Hershey simplex glyphs, thin strokes, scanned over scale
and offset. Score = symmetric chamfer: trimmed mean distance dot->stroke plus mean distance
stroke->nearest dot (so a big glyph with empty strokes is penalised). Lower is better.
A null score from random shuffles of the dots tells whether any fit means anything.

    python fit_letters.py
"""
import json, string
from pathlib import Path
import cv2, numpy as np

HERE = Path(__file__).parent
S = 4  # work at page/4 = 400x300


def glyph(ch, h):
    sc = h / 22.0
    (w, hh), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)
    im = np.zeros((hh + 8, w + 8), np.uint8)
    cv2.putText(im, ch, (4, hh + 4), cv2.FONT_HERSHEY_SIMPLEX, sc, 255, 1)
    ys, xs = np.nonzero(im)
    return np.stack([xs - xs.mean(), ys - ys.mean()], 1).astype(np.float32)


def fit(pts, w):
    best = {}
    c = (pts * w[:, None]).sum(0) / w.sum()
    for ch in string.ascii_uppercase:
        b = 1e9
        for h in (120, 160, 200, 240):
            g = glyph(ch, h)
            g = g[:: max(1, len(g) // 150)]
            for dx in range(-40, 41, 20):
                for dy in range(-40, 41, 20):
                    for ax in (0.8, 1.0, 1.3):
                        gg = g * [ax, 1] + c + [dx, dy]
                        D = np.linalg.norm(pts[:, None] - gg[None], axis=2)
                        d1 = np.sort(D.min(1) * w)[: int(len(pts) * 0.8)].mean()
                        d2 = D.min(0).mean()
                        b = min(b, d1 + d2)
        best[ch] = b
    return sorted(best.items(), key=lambda t: t[1])


def main():
    d = json.load(open(HERE / "decisions_tiered.json", encoding="utf-8"))
    rng = np.random.default_rng(0)
    for al in sorted({x["album"] for x in d}):
        ds = [x for x in d if x["album"] == al and x["why"] != "net"]
        pts = np.array([[x["cx"] / S, x["cy"] / S] for x in ds], np.float32)
        w = np.array([1.0 if x["why"] == "content" else 0.6 for x in ds], np.float32)
        r = fit(pts, w)
        # null: same number of points drawn uniformly in the dots' bounding box
        lo, hi = pts.min(0), pts.max(0)
        null = [fit(rng.uniform(lo, hi, pts.shape).astype(np.float32), w)[0][1] for _ in range(5)]
        print(f"{al}: n={len(pts)} best " + " ".join(f"{c}:{s:.1f}" for c, s in r[:5]) +
              f" | null best {np.mean(null):.1f} +- {np.std(null):.1f}", flush=True)


if __name__ == "__main__":
    main()
