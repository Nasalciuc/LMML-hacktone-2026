"""Per-stamp features from the earlier forgery-net page heatmaps (../outputs_v1, ../outputs_v2).

For every crop in crops.json: the peak of the smoothed heatmap inside the stamp (cut edge
ignored) and where that peak is on the page. Written to net_feats.json.

    python net_feats.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent


def main():
    crops = json.load(open(HERE / "crops.json"))
    H = {v: np.load(ROOT / f"outputs_{v}" / "heatmaps_400x300.npz") for v in ("v1", "v2")}
    out = {}
    for c in crops:
        x, y, w, h = c["box"]
        f = {}
        for v, npz in H.items():
            hm = cv2.GaussianBlur(npz[f"{c['album']}__{c['page']}"].astype(np.float32), (0, 0), 2)
            # heatmap is page/4; stay 8 page-px inside the stamp edge
            x0, y0, x1, y1 = (x + 8) // 4, (y + 8) // 4, (x + w - 8) // 4, (y + h - 8) // 4
            sub = hm[y0:y1, x0:x1]
            if sub.size == 0:
                f[v] = [0.0, x + w // 2, y + h // 2]
                continue
            iy, ix = np.unravel_index(int(sub.argmax()), sub.shape)
            f[v] = [float(sub[iy, ix]), int((x0 + ix) * 4 + 2), int((y0 + iy) * 4 + 2)]
        out[c["crop"][:-4]] = f
    json.dump(out, open(HERE / "net_feats.json", "w"), indent=0)
    print(len(out), "stamps")


if __name__ == "__main__":
    main()
