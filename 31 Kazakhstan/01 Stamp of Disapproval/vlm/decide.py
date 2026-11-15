"""Per-page decision from the VLM reads (+ optional forgery-net features) and the album sheets.

Inputs (all in this folder):
  vlm_exhibits.json  read / p_alt / box for the 30 exhibits (masks = ground truth)
  vlm_albums.json    read / p_alt for every album stamp crop (vlm_fast.py)
  crops.json         album, page, k, box=[x,y,w,h] of each stamp on the 1600x1200 page
  net_feats.json     forgery-net peak per stamp (net_feats.py)
  ground.json        optional: box of the suspicious value/year inside a stamp (ground.py)

Per stamp the content cues are
  strong : a face value with a repeated / copied digit (55, 155, 202, 133...) or a year after 2025
  weak   : a face value that is not a multiple of 5 (32, 47...)
  noyear : no year read on the stamp
Modes:
  content : score = 2*strong + 1*weak + 0.5*noyear + 0.1*net  (net only breaks ties)
  net     : forgery-net peak only (the previous approach, for comparison)
  hybrid  : the stamp with a strong cue if exactly one stamp on the page has one, else net
  tiered  : as hybrid, then the stamp whose year of issue is clearly missing (yearq.py,
            P(no year) >= 0.9 and 0.3 above the next stamp of the page), then net
Sheet colours: black = strong content cue, blue = only no-year stamp, grey = net fallback.
Dot: grounded value box when the pick came from a value cue, else the net peak inside
the chosen stamp, else the stamp centre.

    python decide.py --exhibits
    python decide.py --mode hybrid
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np

import content_rules as CR

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "task_kazakhstan"
PAGE_W, PAGE_H = 1600, 1200


def load(p, default=None):
    p = HERE / p
    return json.load(open(p, encoding="utf-8")) if p.exists() else default


def cues(read_txt):
    d, flags, best = CR.score(read_txt)
    v = flags.get("value", "")
    strong = ("repeat_digit" in v or "copied_digit" in v) or bool(flags.get("bad_year"))
    weak = (not strong) and "not_mult5" in v
    reason = []
    if v: reason.append("value " + v)
    if flags.get("bad_year"): reason.append("year %s" % flags["bad_year"])
    if flags.get("no_year"): reason.append("no year")
    return dict(strong=int(strong), weak=int(weak), noyear=int(bool(flags.get("no_year"))),
                reason="; ".join(reason), value=v.split(":")[0] if v else "")


# ---------------- exhibits ----------------
DIGIT = {2, 7, 9, 12, 13, 17, 22, 24, 25, 26}   # from the mask tiles: a digit was overwritten
CLONE = {15, 19}                                   # a small element was copied


def exhibits_report():
    recs = {r["name"]: r for r in load("vlm_exhibits.json", [])}
    rows = []
    for i in range(1, 31):
        r = recs.get(f"exhibit_{i:02d}")
        if r is None:
            continue
        c = cues(r["read"])
        kind = "digit" if i in DIGIT else "clone" if i in CLONE else "removal"
        m = cv2.imread(str(DATA / "exhibits" / f"exhibit_{i:02d}_mask.png"), 0)
        inside = False
        if r.get("box"):
            b = r["box"]
            cx = min(max((b[0] + b[2]) // 2, 0), m.shape[1] - 1)
            cy = min(max((b[1] + b[3]) // 2, 0), m.shape[0] - 1)
            inside = bool(m[cy, cx] > 127)
        rows.append((i, kind, c, r.get("p_alt", 0), inside))
    print("=== EXHIBITS (all 30 altered) ===")
    for i, kind, c, p, inside in rows:
        print(f"  ex{i:02d} {kind:8s} strong={c['strong']} weak={c['weak']} noyear={c['noyear']} "
              f"p_alt={p:.2f} vlm_box_in_mask={inside}  {c['reason']}")
    for kind in ("digit", "clone", "removal"):
        rs = [x for x in rows if x[1] == kind]
        print(f"  {kind:8s} n={len(rs):2d}  strong={sum(x[2]['strong'] for x in rs)}  "
              f"strong|weak={sum(x[2]['strong'] or x[2]['weak'] for x in rs)}  "
              f"noyear={sum(x[2]['noyear'] for x in rs)}  vlm_box_in_mask={sum(x[4] for x in rs)}")
    print(f"  all      n={len(rows)}  strong={sum(x[2]['strong'] for x in rows)}  "
          f"strong|weak={sum(x[2]['strong'] or x[2]['weak'] for x in rows)}  "
          f"vlm_box_in_mask={sum(x[4] for x in rows)}  p_alt>0.5={sum(x[3] > 0.5 for x in rows)}")


# ---------------- albums ----------------
YQ = {}
def stamp_table():
    global YQ
    YQ = load("yearq_albums.json", {})
    crops = load("crops.json")
    vlm = {r["name"]: r for r in load("vlm_albums.json", [])}
    net = load("net_feats.json", {})
    gr = load("ground.json", {})
    pages = {}
    for c in crops:
        name = c["crop"][:-4]
        if name not in vlm:
            continue
        s = dict(name=name, album=c["album"], page=c["page"], k=c["k"], box=c["box"],
                 p_alt=vlm[name]["p_alt"], read=vlm[name]["read"], **cues(vlm[name]["read"]))
        n = net.get(name, {})
        cx, cy = c["box"][0] + c["box"][2] // 2, c["box"][1] + c["box"][3] // 2
        s["net2"], s["net1"] = n.get("v2", [0, cx, cy])[0], n.get("v1", [0, cx, cy])[0]
        s["net_xy"] = n.get("v2", [0, cx, cy])[1:]
        s["ground"] = gr.get(name)
        s["ym"] = YQ.get(name, {}).get("year_missing", float(s["noyear"]))
        pages.setdefault((c["album"], c["page"]), []).append(s)
    return pages


def netscore(s):
    return 0.5 * (s["net2"] + s["net1"])


def pick(stamps, mode):
    if mode == "net":
        return max(stamps, key=netscore), "net"
    if mode == "content":
        sc = lambda s: 2 * s["strong"] + s["weak"] + 0.5 * s["noyear"] + 0.1 * netscore(s)
        b = max(stamps, key=sc)
        return b, ("content" if (b["strong"] or b["weak"] or b["noyear"]) else "net")
    if mode == "hybrid":
        st = [s for s in stamps if s["strong"]]
        if len(st) == 1:
            return st[0], "content"
        return max(st if st else stamps, key=netscore), "net"
    if mode == "tiered":
        st = [s for s in stamps if s["strong"]]
        if len(st) == 1:
            return st[0], "content"
        # year of issue missing (direct question, P(No)): take it when one stamp stands out
        r = sorted(stamps, key=lambda s: -s["ym"])
        if r[0]["ym"] >= 0.9 and (len(r) == 1 or r[0]["ym"] - r[1]["ym"] >= 0.3):
            return r[0], "noyear"
        return max(st or stamps, key=netscore), "net"
    raise ValueError(mode)


def dot(s, why):
    x, y, w, h = s["box"]
    if why == "content" and s.get("ground"):
        g = s["ground"]
        return int(x + (g[0] + g[2]) / 2), int(y + (g[1] + g[3]) / 2), "ground"
    if s.get("net_xy"):
        return int(s["net_xy"][0]), int(s["net_xy"][1]), "net_peak"
    return int(x + w / 2), int(y + h / 2), "centre"


def albums(mode, tag):
    pages = stamp_table()
    dec = []
    for (al, pg), ss in sorted(pages.items()):
        s, why = pick(ss, mode)
        cx, cy, where = dot(s, why)
        dec.append(dict(album=al, page=pg, k=s["k"], stamp=s["name"], cx=cx, cy=cy, why=why,
                        where=where, reason=s["reason"], n_strong=sum(t["strong"] for t in ss)))
    json.dump(dec, open(HERE / f"decisions_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(HERE / f"boxes_{tag}.csv", "w") as fh:
        fh.write("album,page,x,y,w,h\n")
        for d in dec:
            fh.write(f"{d['album']},{d['page']},{d['cx'] - 25},{d['cy'] - 25},50,50\n")
    panels = []
    for al in sorted({d["album"] for d in dec}):
        sheet = np.full((PAGE_H, PAGE_W, 3), 255, np.uint8)
        for d in dec:
            if d["album"] != al:
                continue
            col = {"content": (0, 0, 0), "noyear": (200, 90, 0)}.get(d["why"], (170, 170, 170))
            cv2.circle(sheet, (d["cx"], d["cy"]), 30, col, -1)
        cv2.putText(sheet, al, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
        cv2.rectangle(sheet, (0, 0), (PAGE_W - 1, PAGE_H - 1), (180, 180, 180), 4)
        panels.append(cv2.resize(sheet, (400, 300), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(HERE / f"album_sheets_{tag}.png"), np.hstack(panels))
    for al in sorted({d["album"] for d in dec}):
        ds = [d for d in dec if d["album"] == al]
        print(f"  {al}: {len(ds)} pages, {sum(d['why'] == 'content' for d in ds)} picked by content, "
              f"pages with exactly one strong stamp: {sum(d['n_strong'] == 1 for d in ds)}")
    print("wrote", f"album_sheets_{tag}.png", f"decisions_{tag}.json", f"boxes_{tag}.csv")
    return dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hybrid", choices=["content", "net", "hybrid", "tiered"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--exhibits", action="store_true")
    a = ap.parse_args()
    if a.exhibits:
        exhibits_report()
        return
    albums(a.mode, a.tag or a.mode)


if __name__ == "__main__":
    main()
