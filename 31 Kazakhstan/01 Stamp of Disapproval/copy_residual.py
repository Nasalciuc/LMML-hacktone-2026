"""For stamps that have a verified copy of the same design elsewhere in the albums
(copies.json), align the copies onto the stamp, take the per-pixel median of the aligned copies
as a reference and score the smoothed colour residual. Pages where one stamp has a copy
get that stamp's residual peak; marks are drawn per album.

    python copy_residual.py
"""
import json
from collections import defaultdict

import cv2
import numpy as np

D = json.load(open("copies.json"))
ids = D["ids"]
pairs = [p for p in D["pairs"] if p["ncc"] > 0.9]
pages = {}
for k, (page, box) in enumerate(ids):
    pages.setdefault(page, None)

partners = defaultdict(list)  # i -> list of (j, H mapping j->i)
for p in pairs:
    H = np.array(p["H"])
    partners[p["i"]].append((p["j"], H))
    partners[p["j"]].append((p["i"], np.linalg.inv(H)))

cache = {}


def crop(k):
    if k not in cache:
        page, (x, y, w, h) = ids[k]
        img = cv2.imread("../task_kazakhstan/albums/" + page)
        cache[k] = img[y:y + h, x:x + w].astype(np.float32)
    return cache[k]


rows = []
for i, plist in partners.items():
    a = crop(i)
    hi, wi = a.shape[:2]
    warped, valids = [], []
    for j, H in plist:
        b = cv2.warpPerspective(crop(j), H, (wi, hi), flags=cv2.INTER_LINEAR)
        v = cv2.warpPerspective(np.ones(crop(j).shape[:2], np.float32), H, (wi, hi)) > 0.99
        # match global colour/contrast of the copy to the stamp
        for c in range(3):
            m = v
            sa, sb = a[..., c][m], b[..., c][m]
            b[..., c] = (b[..., c] - sb.mean()) / (sb.std() + 1e-6) * sa.std() + sa.mean()
        warped.append(b); valids.append(v)
    ref = np.median(np.stack(warped), 0)
    valid = np.all(np.stack(valids), 0)
    res = np.abs(cv2.GaussianBlur(a, (0, 0), 1.5) - cv2.GaussianBlur(ref, (0, 0), 1.5)).mean(2)
    res[~valid] = 0
    res[:12] = res[-12:] = 0; res[:, :12] = res[:, -12:] = 0
    sm = cv2.GaussianBlur(res, (0, 0), 6)
    cy, cx = np.unravel_index(int(sm.argmax()), sm.shape)
    page, (x, y, w, h) = ids[i]
    rows.append((page, int(i), len(plist), float(sm.max()), float(np.median(sm[valid])), int(x + cx), int(y + cy)))

rows.sort()
for r in rows:
    print(f"{r[0]:28s} stamp {r[1]:3d} copies {r[2]} peak {r[3]:6.2f} median {r[4]:5.2f} at ({r[5]},{r[6]})")
json.dump(rows, open("copy_residual.json", "w"))
