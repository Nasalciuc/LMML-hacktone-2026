"""Draw the detected regions of all pages of each album on one sheet ("tracing paper")."""
import sys, os, csv
import numpy as np, cv2


def main(boxes_csv, out_dir, data=None):
    os.makedirs(out_dir, exist_ok=True)
    rows = list(csv.DictReader(open(boxes_csv)))
    albums = sorted(set(r['album'] for r in rows))
    sheets = []
    for a in albums:
        sheet = np.full((1200, 1600, 3), 245, np.uint8)
        for r in rows:
            if r['album'] != a:
                continue
            x, y, w, h = (int(r[k]) for k in ('x', 'y', 'w', 'h'))
            cv2.rectangle(sheet, (x, y), (x + w, y + h), (40, 40, 200), -1)
            cv2.rectangle(sheet, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(sheet, a, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.imwrite(f'{out_dir}/sheet_{a}.png', sheet)
        sheets.append(cv2.resize(sheet, (400, 300), interpolation=cv2.INTER_AREA))
        if data:  # overlay of all pages of the album with the boxes (lamp view)
            acc = None
            for r in rows:
                if r['album'] != a:
                    continue
                im = cv2.imread(f"{data}/albums/{a}/{r['page']}.jpg").astype(np.float32)
                acc = im if acc is None else acc + im
            acc = (acc / max(1, sum(1 for r in rows if r['album'] == a))).astype(np.uint8)
            for r in rows:
                if r['album'] != a:
                    continue
                x, y, w, h = (int(r[k]) for k in ('x', 'y', 'w', 'h'))
                cv2.rectangle(acc, (x, y), (x + w, y + h), (0, 255, 255), 3)
            cv2.imwrite(f'{out_dir}/lamp_{a}.jpg', acc)
    strip = np.hstack([np.pad(s, ((4, 4), (4, 4), (0, 0)), constant_values=120) for s in sheets])
    cv2.imwrite(f'{out_dir}/all_albums.png', strip)


if __name__ == '__main__':
    main(*sys.argv[1:])
