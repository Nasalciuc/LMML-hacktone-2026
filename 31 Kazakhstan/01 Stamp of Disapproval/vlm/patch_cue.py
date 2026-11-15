"""Flat-patch cue for removals, normalised per stamp so stamps of one page can be compared.

Removed inscriptions are filled with flat colour, sometimes plus fresh white noise. On flat
16x16 windows (stride 8) of a stamp we measure the high-pass residual's std and lag-1
autocorrelation. A window is scored by how far it sits below the stamp's own flat windows
(robust z): too smooth (low std) or too white (low autocorrelation). Stamp score = strongest
window, with its position.

    python patch_cue.py            # exhibits check + patch_feats.json for the album crops
"""
import json
from pathlib import Path
import cv2, numpy as np

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "task_kazakhstan"
W, ST = 16, 8


def windows(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    res = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    blur = cv2.GaussianBlur(g, (0, 0), 2)
    out = []
    H, Wd = g.shape
    for y in range(12, H - W - 12, ST):
        for x in range(12, Wd - W - 12, ST):
            b = blur[y:y + W, x:x + W]
            if b.max() - b.min() > 6:        # not flat
                continue
            r = res[y:y + W, x:x + W]
            r = r - r.mean()
            s = r.std()
            ac = 0.5 * ((r[:, 1:] * r[:, :-1]).mean() + (r[1:] * r[:-1]).mean()) / (s * s + 1e-6)
            out.append((x + W // 2, y + W // 2, s, ac))
    return np.array(out, np.float32).reshape(-1, 4)


def rz(v):
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-3
    return (v - med) / mad


def score(img):
    """Returns (score, x, y, n_flat). Higher = more anomalous flat window."""
    w = windows(img)
    if len(w) < 30:
        return 0.0, img.shape[1] // 2, img.shape[0] // 2, len(w)
    z = np.maximum(-rz(w[:, 2]), -rz(w[:, 3]))
    # smooth over neighbours: an erased inscription covers several windows
    pts = w[:, :2]
    zs = np.array([np.sort(z[(np.abs(pts - p).max(1) <= ST)])[-3:].mean() if (np.abs(pts - p).max(1) <= ST).sum() >= 3 else 0 for p in pts])
    i = int(zs.argmax())
    return float(zs[i]), int(w[i, 0]), int(w[i, 1]), len(w)


def main():
    hit = n = 0
    for i in range(1, 31):
        img = cv2.imread(str(DATA / "exhibits" / f"exhibit_{i:02d}.jpg"))
        m = cv2.imread(str(DATA / "exhibits" / f"exhibit_{i:02d}_mask.png"), 0)
        s, x, y, k = score(img)
        md = cv2.dilate(m, np.ones((17, 17), np.uint8))
        ok = bool(md[y, x] > 127)
        hit += ok; n += 1
        print(f"exhibit_{i:02d} score {s:5.1f} flat {k:4d} peak_in_mask {ok}")
    print(f"peak in (dilated) mask: {hit}/{n}")
    out = {}
    for c in json.load(open(HERE / "crops.json")):
        s, x, y, k = score(cv2.imread(str(HERE / "crops" / c["crop"])))
        out[c["crop"][:-4]] = [s, x, y, k]
    json.dump(out, open(HERE / "patch_feats.json", "w"))
    print(len(out), "album stamps")


if __name__ == "__main__":
    main()
