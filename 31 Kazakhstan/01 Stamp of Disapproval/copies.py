"""Find copies of the same stamp design among all album stamps (ORB + RANSAC homography),
verify by correlation after alignment, and save the verified pairs."""
import glob, json
import cv2
import numpy as np
from forgery_net import stamps_on

orb = cv2.ORB_create(2000); bf = cv2.BFMatcher(cv2.NORM_HAMMING)
crops, ids = [], []
for f in sorted(glob.glob('../task_kazakhstan/albums/*/*.jpg')):
    p = cv2.imread(f)
    for b in stamps_on(p):
        x, y, w, h = b
        crops.append(p[y:y + h, x:x + w]); ids.append((f.replace(chr(92), '/').split('albums/')[1], [int(v) for v in b]))
gray = [cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for c in crops]
feats = [orb.detectAndCompute(g, None) for g in gray]
pairs = []
n = len(crops)
for i in range(n):
    ki, di = feats[i]
    for j in range(i + 1, n):
        kj, dj = feats[j]
        if di is None or dj is None:
            continue
        m = bf.knnMatch(di, dj, k=2)
        good = [a for a, b2 in (x for x in m if len(x) == 2) if a.distance < 0.7 * b2.distance]
        if len(good) < 25:
            continue
        src = np.float32([ki[a.queryIdx].pt for a in good]); dst = np.float32([kj[a.trainIdx].pt for a in good])
        H, inl = cv2.findHomography(dst, src, cv2.RANSAC, 3.0)  # maps j -> i
        if H is None or inl.sum() < 25:
            continue
        hi, wi = gray[i].shape
        wj = cv2.warpPerspective(gray[j], H, (wi, hi), flags=cv2.INTER_LINEAR, borderValue=0)
        valid = cv2.warpPerspective(np.ones_like(gray[j]), H, (wi, hi)) > 0
        if valid.mean() < 0.8:
            continue
        a = gray[i][valid].astype(np.float32); b = wj[valid].astype(np.float32)
        ncc = float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-6))
        pairs.append({"i": i, "j": j, "inliers": int(inl.sum()), "ncc": round(ncc, 4), "H": H.tolist(), "cover": round(float(valid.mean()), 3)})
json.dump({"ids": ids, "pairs": pairs}, open("copies.json", "w"))
good = [p for p in pairs if p["ncc"] > 0.9]
stamps = set([p["i"] for p in good] + [p["j"] for p in good])
print(f"{n} stamps, {len(pairs)} aligned pairs, {len(good)} with ncc>0.9 covering {len(stamps)} stamps")
