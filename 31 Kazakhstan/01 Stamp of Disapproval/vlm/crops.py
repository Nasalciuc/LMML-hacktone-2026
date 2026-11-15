"""Cut every album page into its three stamps; save crops and a json index."""
import json
from pathlib import Path
import cv2
import numpy as np

DATA = Path(__file__).resolve().parents[2] / "task_kazakhstan"
OUT = Path(__file__).parent / "crops"


def stamps_on(page):
    d = np.abs(page.astype(np.int16) - np.array([38, 36, 34])).max(2)
    m = cv2.morphologyEx((d > 30).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    for opening in (0, 5):  # opening only when two stamps touch
        mm = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((opening, opening), np.uint8)) if opening else m
        _, _, stats, _ = cv2.connectedComponentsWithStats(mm)
        # small strips are about 400 x 90 px
        boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h, a in stats[1:] if w > 60 and h > 60 and w * h > 15000]
        if len(boxes) >= 3:
            break
    return boxes


def main():
    OUT.mkdir(exist_ok=True)
    index = []
    for album in sorted((DATA / "albums").iterdir()):
        for f in sorted(album.glob("*.jpg")):
            page = cv2.imread(str(f))
            boxes = sorted(stamps_on(page), key=lambda b: (b[0], b[1]))
            if len(boxes) != 3:
                print("warning", album.name, f.stem, len(boxes))
            for k, (x, y, w, h) in enumerate(boxes):
                name = f"{album.name}_{f.stem}_{k}.png"
                cv2.imwrite(str(OUT / name), page[y:y + h, x:x + w])
                index.append(dict(album=album.name, page=f.stem, k=k, box=[x, y, w, h], crop=name))
    json.dump(index, open(Path(__file__).parent / "crops.json", "w"), indent=1)
    print(len(index), "stamps")


if __name__ == "__main__":
    main()
