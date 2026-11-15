"""Locate the suspicious face value / year inside a stamp with Qwen2.5-VL grounding.

"Where is the alteration?" gives useless boxes (4/30 on the exhibits), but "box the printed
text 202" is a normal grounding query. For every stamp whose read has a strong value/year cue
we ask for the box of that text and store it in crop pixels.

    python ground.py --what exhibits   # check against the masks -> ground_exhibits.json
    python ground.py --what albums     # only stamps with a strong cue -> ground.json
"""
import argparse, json
from pathlib import Path

import cv2
from PIL import Image

import content_rules as CR
from decide import cues
from vlm_read import generate, load, parse_box, prep

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "task_kazakhstan"


def target(read_txt):
    c = cues(read_txt)
    if not c["strong"]:
        return None
    d, flags, _ = CR.score(read_txt)
    if flags.get("bad_year"):
        return str(flags["bad_year"][0])
    return c["value"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["exhibits", "albums"], required=True)
    a = ap.parse_args()
    if a.what == "exhibits":
        recs = json.load(open(HERE / "vlm_exhibits.json", encoding="utf-8"))
        out_path = HERE / "ground_exhibits.json"
    else:
        recs = json.load(open(HERE / "vlm_albums.json", encoding="utf-8"))
        out_path = HERE / "ground.json"
    jobs = [(r, target(r["read"])) for r in recs]
    jobs = [(r, t) for r, t in jobs if t]
    print(len(jobs), "stamps to ground", flush=True)
    model, proc = load()
    out = {}
    for r, t in jobs:
        im = prep(r["crop"])
        orig = Image.open(r["crop"]).size
        q = (f'Locate the printed number "{t}" on this stamp. '
             'Output its bounding box as JSON: {"bbox_2d": [x1, y1, x2, y2]}.')
        txt = generate(model, proc, im, q, 48)
        b = parse_box(txt, im, orig)
        out[r["name"]] = b
        print(r["name"], t, b, flush=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=0)
    if a.what == "exhibits":
        hit = 0
        for name, b in out.items():
            m = cv2.imread(str(DATA / "exhibits" / f"{name}_mask.png"), 0)
            if b is None:
                continue
            cx = min(max((b[0] + b[2]) // 2, 0), m.shape[1] - 1)
            cy = min(max((b[1] + b[3]) // 2, 0), m.shape[0] - 1)
            # count a hit if the box overlaps the mask at all
            sub = m[max(0, b[1]):b[3], max(0, b[0]):b[2]]
            ok = bool(sub.size and (sub > 127).any())
            hit += ok
            print(f"  {name} box overlaps mask: {ok} (centre in mask: {bool(m[cy, cx] > 127)})")
        print(f"grounded value box overlaps the mask on {hit}/{len(out)} flagged exhibits")


if __name__ == "__main__":
    main()
