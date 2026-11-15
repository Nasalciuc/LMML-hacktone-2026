"""Figures for the write-up: DQ evidence on exhibits (with ground truth) and on one album page."""
import sys, os
import numpy as np, cv2
from segment import stamp_masks
from dq import dq_maps

def dq_view(im, st):
    M, info = dq_maps(im, st)
    keep = np.zeros(im.shape[:2], bool)
    for _, m in st:
        keep |= cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5) > 12
    Ls = cv2.boxFilter(M['L'] * keep, -1, (33, 33), normalize=False)
    v = cv2.applyColorMap(np.clip(128 + Ls * 0.8, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    sm = np.zeros(im.shape[:2], bool)
    for _, m in st: sm |= m
    v[~sm] = (40, 40, 40)
    return v, Ls, info

def main(data, out):
    rows = []
    for i in (2, 12, 17, 27, 9, 30):
        im = cv2.imread(f'{data}/exhibits/exhibit_{i:02d}.jpg')
        m = (cv2.imread(f'{data}/exhibits/exhibit_{i:02d}_mask.png', 0) > 127).astype(np.uint8)
        st = stamp_masks(im)
        v, Ls, info = dq_view(im, st)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        a = im.copy(); cv2.drawContours(a, cnts, -1, (0, 255, 0), 2); cv2.drawContours(v, cnts, -1, (255, 255, 255), 2)
        t = np.hstack([cv2.resize(a, (300, int(300 * im.shape[0] / im.shape[1]))), cv2.resize(v, (300, int(300 * im.shape[0] / im.shape[1])))])
        g = info[0][0] if info and info[0] else {}
        cv2.putText(t, 'ex%02d QF%s grid(%s,%s)' % (i, g.get('qf'), g.get('ox_page'), g.get('oy_page')), (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        rows.append(t)
    H = max(r.shape[0] for r in rows)
    rows = [np.pad(r, ((0, H - r.shape[0]), (0, 6), (0, 0)), constant_values=255) for r in rows]
    cv2.imwrite(f'{out}/dq_exhibits.jpg', np.vstack([np.hstack(rows[:3]), np.hstack(rows[3:])]))
    im = cv2.imread(f'{data}/albums/album_1/page_1759.jpg')
    st = stamp_masks(im)
    v, Ls, info = dq_view(im, st)
    cy, cx = np.unravel_index(np.argmin(Ls), Ls.shape)
    a = im.copy(); cv2.circle(a, (cx, cy), 45, (0, 255, 255), 4); cv2.circle(v, (cx, cy), 45, (255, 255, 255), 4)
    cv2.imwrite(f'{out}/dq_page_album1_1759.jpg', np.hstack([cv2.resize(a, (800, 600)), cv2.resize(v, (800, 600))]))

if __name__ == '__main__':
    main(*sys.argv[1:])
