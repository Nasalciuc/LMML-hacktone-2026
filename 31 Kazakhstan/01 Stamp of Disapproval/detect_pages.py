"""Run the forgery network on every album page, keep the strongest region per page,
write boxes.csv and draw one sheet per album (all page regions on one 1600x1200 sheet).

    python detect_pages.py --data ../task_kazakhstan --model forgery_net.pt
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from forgery_net import UNet, predict, stamps_on

PAGE_W, PAGE_H = 1600, 1200

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("../task_kazakhstan"))
ap.add_argument("--model", type=Path, nargs="+", default=[Path("forgery_net.pt")])
ap.add_argument("--out", type=Path, default=Path("outputs"))
a = ap.parse_args()
a.out.mkdir(exist_ok=True)
(a.out / "pages").mkdir(exist_ok=True)

models = []
for mp in a.model:
    m = UNet().cuda(); m.load_state_dict(torch.load(mp, map_location="cuda")); m.eval(); models.append(m)
heats = {}

rows = []
albums = sorted(p for p in (a.data / "albums").iterdir() if p.is_dir())
sheets = []
for album in albums:
    sheet = np.full((PAGE_H, PAGE_W, 3), 255, np.uint8)
    for f in sorted(album.glob("*.jpg")):
        page = cv2.imread(str(f))
        heat = np.zeros(page.shape[:2], np.float32)
        for x, y, w, h in stamps_on(page):
            p = np.mean([predict(m, page[y:y + h, x:x + w]) for m in models], 0)
            p[:6], p[-6:], p[:, :6], p[:, -6:] = 0, 0, 0, 0  # ignore the cut edge
            heat[y:y + h, x:x + w] = np.maximum(heat[y:y + h, x:x + w], p)
        heats[f"{album.name}__{f.stem}"] = cv2.resize(heat, (400, 300), interpolation=cv2.INTER_AREA).astype(np.float16)
        sm = cv2.GaussianBlur(heat, (0, 0), 8)
        cy, cx = np.unravel_index(int(sm.argmax()), sm.shape)
        score = float(sm[cy, cx])
        # box = connected region of the raw map around the peak
        reg = (heat > 0.5 * heat[max(0, cy - 15):cy + 15, max(0, cx - 15):cx + 15].max()).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(reg)
        k = lab[cy, cx]
        if k > 0 and st[k, 4] > 20:
            bx, by, bw, bh = (int(v) for v in st[k, :4])
        else:
            bx, by, bw, bh = cx - 25, cy - 25, 50, 50
        rows.append([album.name, f.stem, bx, by, bw, bh, cx, cy, round(score, 3)])
        cv2.rectangle(sheet, (bx, by), (bx + bw, by + bh), (0, 0, 0), -1)
        vis = page.copy()
        hm = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        vis = np.where(heat[..., None] > 0.2, (0.5 * vis + 0.5 * hm).astype(np.uint8), vis)
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
        cv2.imwrite(str(a.out / "pages" / f"{album.name}_{f.stem}.jpg"), cv2.resize(vis, None, fx=0.5, fy=0.5))
    sheets.append(sheet)
    print(album.name, "done", flush=True)

np.savez_compressed(a.out / "heatmaps_400x300.npz", **heats)
with open(a.out / "boxes.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["album", "page", "x", "y", "w", "h", "cx", "cy", "score"])
    w.writerows(rows)

small = [cv2.resize(s, (400, 300), interpolation=cv2.INTER_AREA) for s in sheets]
for s in small:
    cv2.rectangle(s, (0, 0), (399, 299), (180, 180, 180), 1)
cv2.imwrite(str(a.out / "album_sheets.png"), np.hstack(small))
for i, s in enumerate(sheets):
    cv2.imwrite(str(a.out / f"sheet_{albums[i].name}.png"), cv2.resize(s, (800, 600), interpolation=cv2.INTER_AREA))
print("wrote", a.out / "boxes.csv", a.out / "album_sheets.png")
