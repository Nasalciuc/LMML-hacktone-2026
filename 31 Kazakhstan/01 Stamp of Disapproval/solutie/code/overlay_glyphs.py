"""Check figure: the regions of each album over the glyph of the letter read from them."""
import sys, csv
import numpy as np, cv2
from letter_check import glyph

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def main(boxes_csv, word, out_png):
    rows = list(csv.DictReader(open(boxes_csv)))
    albums = sorted(set(r['album'] for r in rows))
    tiles = []
    for a, ch in zip(albums, word):
        R = [r for r in rows if r['album'] == a]
        P = np.array([[int(r['x']) + int(r['w']) / 2, int(r['y']) + int(r['h']) / 2] for r in R])
        x0, x1 = np.percentile(P[:, 0], [3, 97]); y0, y1 = np.percentile(P[:, 1], [3, 97])
        g = glyph(ch, FONT).astype(np.uint8) * 255
        pad = 0.06
        W = int((x1 - x0) * (1 + 2 * pad)); H = int((y1 - y0) * (1 + 2 * pad))
        gm = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
        sheet = np.full((1200, 1600, 3), 255, np.uint8)
        gx = int(x0 - (x1 - x0) * pad); gy = int(y0 - (y1 - y0) * pad)
        sl = sheet[max(0, gy):gy + H, max(0, gx):gx + W]
        gm = gm[:sl.shape[0], :sl.shape[1]]
        sl[gm > 127] = (200, 225, 255)
        for r in R:
            x, y, w, h = (int(r[k]) for k in ('x', 'y', 'w', 'h'))
            col = (30, 30, 30) if r['source'] == 'dq' else (40, 40, 200)
            cv2.rectangle(sheet, (x, y), (x + w, y + h), col, -1)
        cv2.putText(sheet, f'{a}: {ch}', (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 150), 5)
        tiles.append(cv2.resize(sheet, (560, 420), interpolation=cv2.INTER_AREA))
    cv2.imwrite(out_png, np.hstack([np.pad(t, ((4, 4), (4, 4), (0, 0)), constant_values=120) for t in tiles]))


if __name__ == '__main__':
    main(*sys.argv[1:])
