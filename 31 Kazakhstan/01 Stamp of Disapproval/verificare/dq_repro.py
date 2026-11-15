"""Re-run the archive's double-quantisation evidence on our copy of the task data.

    python -I dq_repro.py <archive code dir> <task_kazakhstan> <archive boxes.csv> <out.json> <what> [what ...]

what = "exhibits" or an album name (album_1 ...). Uses only segment.py and dq.py from the archive
(numpy + cv2). For pages it recomputes the strongest contradiction of the stamp's first JPEG
compression (the archive's dqL33 feature) and compares its value and place with boxes.csv.
"""
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

CODE, DATA, BOXES, OUT = sys.argv[1:5]
WHAT = sys.argv[5:]
sys.path.insert(0, CODE)
from dq import dq_features, dq_maps  # noqa: E402
from segment import stamp_masks  # noqa: E402

cv2.setNumThreads(1)
box = {(r["album"], r["page"]): r for r in csv.DictReader(open(BOXES))}
res = {"exhibits": [], "pages": []}
for what in WHAT:
    if what == "exhibits":
        for i in range(1, 31):
            im = cv2.imread(f"{DATA}/exhibits/exhibit_{i:02d}.jpg")
            m = cv2.imread(f"{DATA}/exhibits/exhibit_{i:02d}_mask.png", 0) > 127
            st = stamp_masks(im)
            M, info = dq_maps(im, st)
            keep = np.zeros(m.shape, bool)
            for _, s in st:
                keep |= cv2.distanceTransform(s.astype(np.uint8), cv2.DIST_L2, 5) > 12
            Ls = cv2.boxFilter(M["L"] * keep, -1, (33, 33), normalize=False)
            cy, cx = np.unravel_index(np.argmin(Ls), Ls.shape)
            hit = bool((cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)[cy, cx])
            res["exhibits"].append(dict(i=i, min=float(Ls[cy, cx]), x=int(cx), y=int(cy), inside=hit))
            print("exhibit", i, round(float(Ls[cy, cx]), 1), hit, flush=True)
        continue
    for f in sorted(os.listdir(f"{DATA}/albums/{what}")):
        if not f.endswith(".jpg"):
            continue
        t = time.time()
        im = cv2.imread(f"{DATA}/albums/{what}/{f}")
        st = stamp_masks(im)
        F, M, info = dq_features(im, st, prefix="dq", clip_hi=254, clip_lo=1, cf_llr=0.05)
        stamp = np.zeros(im.shape[:2], bool)
        for _, mk in st:
            stamp |= mk
        dist = cv2.distanceTransform(stamp.astype(np.uint8), cv2.DIST_L2, 5)
        cand = stamp & (dist > 6)
        sub = np.zeros_like(cand)
        sub[::2, ::2] = True
        L = np.where(cand & sub, F["dqL33"], np.inf)
        yy, xx = np.unravel_index(np.argmin(L), L.shape)
        r = box[(what, f[:-4])]
        x, y, w, h = (int(r[k]) for k in "xywh")
        # second strongest contradiction on another stamp (how unique is the first one)
        other = np.inf
        for (bx, by, bw, bh), mk in st:
            if not mk[yy, xx]:
                v = L[mk]
                if v.size:
                    other = min(other, float(v.min()))
        d = dict(album=what, page=f[:-4], n_stamps=len(st), dq_min=float(L[yy, xx]), x=int(xx), y=int(yy),
                 other_stamps_min=None if other == np.inf else other,
                 archive_dq_min=float(r["dq_min"]), archive_source=r["source"],
                 in_archive_box=bool(x - 20 <= xx <= x + w + 20 and y - 20 <= yy <= y + h + 20),
                 qf=[[g["qf"] for g in gs] for gs in info], sec=round(time.time() - t, 1))
        res["pages"].append(d)
        print(what, f[:-4], round(d["dq_min"], 1), d["archive_dq_min"], d["archive_source"], d["in_archive_box"], d["sec"], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("saved", OUT)
