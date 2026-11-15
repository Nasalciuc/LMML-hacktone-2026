"""Decode an earned Island Duel star chart into the SIGMOID_<WORD> flag.

Reading order is geometric (the viewer scrambles paint order and prints no
numbers). This prints the letters under many plausible geometric orderings,
separating big "guide" stars (size>6) from small "companion" stars, so the
intended word is easy to spot. Usage: python decode_chart.py path/to/x.chart.json
"""

import json
import math
import sys


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return d


def order_report(stars, label):
    letters = "".join(s["text"] for s in stars)
    print(f"  {label:42s} -> {letters}")


def analyze(group, name):
    if not group:
        return
    print(f"\n== {name} ({len(group)} stars) ==")
    xs = [s["x"] for s in group]
    ys = [s["y"] for s in group]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    # principal axis via covariance
    sxx = sum((x - cx) ** 2 for x in xs)
    syy = sum((y - cy) ** 2 for y in ys)
    sxy = sum((x - cx) * (y - cy) for x, y in zip(xs, ys))
    theta = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ux, uy = math.cos(theta), math.sin(theta)

    def proj(s):
        return (s["x"] - cx) * ux + (s["y"] - cy) * uy

    orders = {
        "y then x (reading, top-left)": lambda s: (round(s["y"], 1), s["x"]),
        "x then y (columns, left-top)": lambda s: (round(s["x"], 1), s["y"]),
        "principal axis": proj,
        "principal axis reversed": lambda s: -proj(s),
        "angle around centroid": lambda s: math.atan2(s["y"] - cy, s["x"] - cx),
        "size desc then y": lambda s: (-s["size"], s["y"], s["x"]),
        "radius from centroid": lambda s: (s["x"] - cx) ** 2 + (s["y"] - cy) ** 2,
    }
    for lbl, key in orders.items():
        order_report(sorted(group, key=key), lbl)


def main(path):
    d = load(path)
    print("keys:", list(d.keys()))
    for k in ("title", "verse", "view_box"):
        if k in d:
            print(f"{k}: {d[k]}")
    stars = d["stars"]
    print(f"\n{len(stars)} stars total")
    for s in sorted(stars, key=lambda s: (s["y"], s["x"])):
        print(f"  x={s['x']:7.2f} y={s['y']:7.2f} size={s['size']:5.2f} "
              f"{'GUIDE' if s['size'] > 6 else 'comp '} '{s['text']}'")
    guide = [s for s in stars if s["size"] > 6]
    comp = [s for s in stars if s["size"] <= 6]
    analyze(stars, "ALL stars")
    analyze(guide, "GUIDE stars (big)")
    analyze(comp, "COMPANION stars (small)")
    # guide-first then companion, each in reading order
    both = (sorted(guide, key=lambda s: (s["y"], s["x"]))
            + sorted(comp, key=lambda s: (s["y"], s["x"])))
    print("\nGUIDE(reading)+COMP(reading):",
          "".join(s["text"] for s in both))
    print("COMP(reading)+GUIDE(reading):",
          "".join(s["text"] for s in (sorted(comp, key=lambda s: (s["y"], s["x"]))
                                      + sorted(guide, key=lambda s: (s["y"], s["x"])))))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluation.chart.json")
