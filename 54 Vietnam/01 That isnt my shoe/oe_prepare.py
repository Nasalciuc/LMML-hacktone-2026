"""Build the image caches for the pretrained + outlier-exposure model.

train : the 735 labelled photos (uint8, 256x256)
hf    : ipogorelov/sneakers (Bing photos with brand labels), squashed to 256x256 like the
        task photos, near-duplicates of task photos dropped, capped per brand.
Output: cache/train.npz, cache/hf.npz  (cache/ is git-ignored)
"""
import collections, csv, glob, io, sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

HERE = Path(__file__).resolve().parent
TASK = HERE.parent / "task_vietnam"
HF_DIR = Path(r"C:/Users/user/data/vietnam_oe/hf_sneakers")
OUT = HERE / "cache"
CAP = 1200          # max images kept per HF brand
SIDE = 256
rng = np.random.default_rng(0)


def key(a):
    """32x32 grey, zero mean / unit variance: dot product / 1024 = correlation."""
    g = np.asarray(Image.fromarray(a).convert("L").resize((32, 32), Image.BILINEAR), np.float32).ravel()
    return (g - g.mean()) / (g.std() + 1e-6)


def main():
    OUT.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(TASK / "train-output.csv")))
    tr = np.stack([np.asarray(Image.open(TASK / "train-input" / r["filename"]).convert("RGB")) for r in rows])
    np.savez(OUT / "train.npz", x=tr, y=np.array([r["label"] for r in rows]), f=np.array([r["filename"] for r in rows]))
    tk = np.stack([key(a) for a in tr] + [key(a[:, ::-1]) for a in tr])   # photos + mirrors

    # pass 1: brand of every row, so the cap can be a random sample
    files = sorted(glob.glob(str(HF_DIR / "*.parquet")))
    idx = []
    for fi, f in enumerate(files):
        for ri, b in enumerate(pq.read_table(f, columns=["brand"])["brand"].to_pylist()):
            idx.append((b.strip().lower(), fi, ri))
    cnt = collections.Counter(b for b, _, _ in idx)
    print("brands:", cnt.most_common())
    keep = {}
    for b in cnt:
        rows_b = [(fi, ri) for bb, fi, ri in idx if bb == b]
        sel = rng.permutation(len(rows_b))[: int(CAP * 1.3)]   # extra for dropped ones
        for k in sel:
            keep[rows_b[k]] = b
    xs, bs, ms, n_dup, n_bad = [], [], [], 0, 0
    per = collections.Counter()
    for fi, f in enumerate(files):
        want = {ri: b for (ffi, ri), b in keep.items() if ffi == fi}
        if not want:
            continue
        t = pq.read_table(f)
        imgs, models = t["image"].to_pylist(), t["model"].to_pylist()
        for ri, b in sorted(want.items()):
            if per[b] >= CAP:
                continue
            try:
                im = Image.open(io.BytesIO(imgs[ri])).convert("RGB")
            except Exception:
                n_bad += 1
                continue
            if min(im.size) < 100:
                n_bad += 1
                continue
            a = np.asarray(im.resize((SIDE, SIDE), Image.BICUBIC))
            if (tk @ key(a)).max() / 1024 > 0.97:      # same photo as a task photo
                n_dup += 1
                continue
            xs.append(a)
            bs.append(b); ms.append(models[ri] or ""); per[b] += 1
        print(f"{fi+1}/{len(files)} kept {len(xs)}", flush=True)
    print("kept per brand:", per.most_common(), "near-dups of task photos dropped:", n_dup, "bad:", n_bad)
    np.savez(OUT / "hf.npz", x=np.stack(xs), brand=np.array(bs), model=np.array(ms))


if __name__ == "__main__":
    main()
