"""Cross-check of the solution's boxes against our own earlier result.

    python -I compare.py <solution boxes.csv> <our decisions_tiered.json> <our crops.json>

On the pages where our older content reading found an impossible face value, is the solution's
box on the same stamp? Two more arguments (a new and an old Japan attribution table) run an
unrelated comparison that was done in the same session; without them that part is skipped.
"""
import collections
import csv
import json
import sys

import numpy as np

BOX, OURS, CROPS = sys.argv[1:4]
JNEW, JOLD = (sys.argv[4:6] if len(sys.argv) >= 6 else (None, None))

# ---------------------------------------------------------------- Kazakhstan
rows = list(csv.DictReader(open(BOX)))
box = {(r["album"], r["page"]): r for r in rows}
print("archive boxes:", len(rows), "| per album:", dict(collections.Counter(r["album"] for r in rows)),
      "| source:", dict(collections.Counter(r["source"] for r in rows)))
ours = json.load(open(OURS))
crops = json.load(open(CROPS))
print("crops.json type:", type(crops).__name__, "n:", len(crops))
ck = list(crops)[:1] if isinstance(crops, dict) else crops[:1]
print("crops sample:", json.dumps(ck if not isinstance(crops, dict) else {ck[0]: crops[ck[0]]})[:300])


def stamp_box(name):
    c = crops[name] if isinstance(crops, dict) else next(x for x in crops if x.get("crop") == name + ".png")
    if isinstance(c, dict):
        for keys in (("x", "y", "w", "h"), ("x0", "y0", "x1", "y1")):
            if all(k in c for k in keys):
                v = [c[k] for k in keys]
                return (v[0], v[1], v[0] + v[2], v[1] + v[3]) if keys[0] == "x" else tuple(v)
        if "box" in c:  # x, y, w, h
            x, y, w, h = c["box"]
            return (x, y, x + w, y + h)
    return tuple(c)


tot = collections.Counter()
hit_stamp = collections.Counter()
hit_box = collections.Counter()
dist = collections.defaultdict(list)
by_src = collections.Counter()
by_src_hit = collections.Counter()
for d in ours:
    r = box[(d["album"], d["page"])]
    x, y, w, h = (float(r[k]) for k in "xywh")
    cx, cy = x + w / 2, y + h / 2
    why = d["why"]
    tot[why] += 1
    try:
        sb = stamp_box(d["stamp"])
        same = sb[0] <= cx <= sb[2] and sb[1] <= cy <= sb[3]
    except Exception as e:  # unknown crop format
        same = None
    hit_stamp[why] += bool(same)
    m = 25
    inside = x - m <= d["cx"] <= x + w + m and y - m <= d["cy"] <= y + h + m
    hit_box[why] += inside
    dist[why].append(float(np.hypot(d["cx"] - cx, d["cy"] - cy)))
    if why == "content":
        by_src[r["source"]] += 1
        by_src_hit[r["source"]] += bool(same)
print("\nour pick vs archive box, by our tier (content = odd face value read by the VLM, our only trusted tier)")
for why in tot:
    print(f"  {why:8s} pages {tot[why]:3d} | archive box centre inside our chosen stamp {hit_stamp[why]:3d} "
          f"| our point within 25 px of archive box {hit_box[why]:3d} | median distance {np.median(dist[why]):.0f} px")
print("  content pages by archive decision source:", dict(by_src), "| same stamp:", dict(by_src_hit))
print("  chance level for 'same stamp' is about one in three")
for a in sorted({d["album"] for d in ours}):
    k = [d for d in ours if d["album"] == a and d["why"] == "content"]
    s = 0
    for d in k:
        r = box[(d["album"], d["page"])]
        x, y, w, h = (float(r[q]) for q in "xywh")
        sb = stamp_box(d["stamp"])
        s += sb[0] <= x + w / 2 <= sb[2] and sb[1] <= y + h / 2 <= sb[3]
    print(f"  {a}: content pages {len(k):2d}, same stamp {s:2d}")

# ---------------------------------------------------------------- Japan
if JNEW is None:
    sys.exit(0)
new = {r["file"]: r for r in csv.DictReader(open(JNEW))}
old = {r["file"]: r for r in csv.DictReader(open(JOLD))}
print("\nJapan: new rows", len(new), "| ours", len(old), "| same files", set(new) == set(old))
same_p = [f for f in new if new[f]["photographer"] == old[f]["photographer"]]
same_r = [f for f in new if str(new[f]["rotated_180"]) == str(old[f]["rotated180"])]
print("  same photographer:", len(same_p), "of", len(new), "| same rotation flag:", len(same_r))
for f in sorted(set(new) - set(same_p)):
    print("   differs:", f, "new", new[f]["photographer"], new[f]["z_owner"], "| ours", old[f]["photographer"], old[f]["z"], old[f].get("note", ""))
for f in sorted(set(new) - set(same_r)):
    print("   rotation differs:", f, new[f]["rotated_180"], old[f]["rotated180"])
cnt = collections.Counter((r["week"], r["photographer"]) for r in new.values())
names = sorted({r["photographer"] for r in new.values()})
print("  per week:", {w: [cnt[(w, n)] for n in names] for w in sorted({r["week"] for r in new.values()}, key=int)}, names)
weak = sorted((float(r["z_owner"]) - float(r["z_best_other"]), f, r["photographer"]) for f, r in new.items())[:10]
print("  weakest margins:", [(f, p, round(m, 2)) for m, f, p in weak])
