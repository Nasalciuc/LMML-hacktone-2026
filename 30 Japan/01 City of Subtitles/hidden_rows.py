"""Does a JPEG hold more image data than its header admits? Rewrite the frame height in the SOF0 marker to a
larger value, decode, and measure how many extra rows decode to real content (not the grey padding)."""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

import prnu

ImageFile.LOAD_TRUNCATED_IMAGES = True


def patched(path, new_h):
    b = bytearray(open(path, "rb").read())
    i = b.find(b"\xff\xc0")
    h = int.from_bytes(b[i + 5:i + 7], "big")
    b[i + 5:i + 7] = new_h.to_bytes(2, "big")
    return bytes(b), h


def extra_rows(path, new_h=1440):
    data, h = patched(path, new_h)
    im = Image.open(io.BytesIO(data))
    im.load()
    a = np.asarray(im.convert("L"), dtype=np.float32)
    tail = a[h:]
    rows_content = int((tail.std(1) > 2).sum())
    return h, rows_content, tail


if __name__ == "__main__":
    out = []
    for p in sorted((prnu.DATA / "anonymous").glob("*.jpg"))[: int(sys.argv[1]) if len(sys.argv) > 1 else 999]:
        h, n, tail = extra_rows(p)
        out.append((p.name, n))
        if n:
            print(p.name, "extra rows with content:", n)
    print("checked", len(out), "photos; with extra content:", sum(1 for _, n in out if n))
