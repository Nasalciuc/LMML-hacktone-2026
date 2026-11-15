"""Joint letter fit per album over candidate edit points of all 26 pages.

Idea: the altered stamp tends to be the most central one of the page (26/14/2 on the 42
strong-cue pages) and inside a stamp an edit sits in few places. So instead of deciding page
by page we ask which letter, placed somewhere on the page, is touched by most pages.

Candidates per page (page pixels):
  * strong-cue page: centre of the grounded odd number, weight 3, nothing else;
  * other pages: the most central stamp (weight 1), the second most central (0.5) and the
    year-missing pick of decide.py (at least 1; 1.5 if it is also the most central).
    Points inside a stamp: grounded face value, grounded year (elements.json), the two
    places where the removals of the exhibits sit (bottom-left 0.23/0.83, top-right
    0.82/0.24 of the stamp) and the 2 strongest flat-patch windows (patch_cue).
Score of a placed letter = sum over pages of the largest weight among the page's candidates
that lie on the (dilated) letter. All offsets are scanned exactly, on a page/4 grid.
Templates: A-Z x 4 fonts (Hershey strokes, Arial, Arial Bold, Courier Bold filled) x 7 heights
x 4 aspect ratios.
Controls: shuffled albums - same number of strong and other pages, drawn from the other four
albums. A letter's z = (score - mean of that letter on the controls) / std.

    python joint_fit.py            # resumable; results in joint_fit_results.json
    python joint_fit.py --report   # table + joint_fit_overlays.png from the saved results
"""
import argparse, json, string
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
S = 4                       # page/4 -> 400 x 300
PW, PH = 400, 300
HEIGHTS = (110, 135, 160, 185, 210, 235, 260)
ASPECTS = (0.7, 0.85, 1.0, 1.2)
FONTS = ("hershey", "arial.ttf", "arialbd.ttf", "courbd.ttf")
R_FONT = {"hershey": 9}     # tolerance in page/4 px; filled fonts use 5
N_CTRL = 10
LETTERS = string.ascii_uppercase
PAD = 10


# ---------------- candidates ----------------
def cdist(s):
    x, y, w, h = s["box"]
    return float(np.hypot(x + w / 2 - 800, y + h / 2 - 600))


def build_candidates():
    import patch_cue
    from decide import stamp_table
    P = stamp_table()
    el = json.load(open(HERE / "elements.json"))
    dec = {(d["album"], d["page"]): d for d in json.load(open(HERE / "decisions_tiered.json", encoding="utf-8"))}
    pages = []
    for (al, pg), ss in sorted(P.items()):
        st = [s for s in ss if s["strong"]]
        c = []
        if len(st) == 1 and st[0].get("ground"):
            g, b = st[0]["ground"], st[0]["box"]
            c.append([b[0] + (g[0] + g[2]) / 2, b[1] + (g[1] + g[3]) / 2, 3.0, "odd value", st[0]["name"]])
            pages.append(dict(album=al, page=pg, strong=True, cands=c))
            continue
        d = dec[(al, pg)]
        for r, s in enumerate(sorted(ss, key=cdist)):
            w = (1.0, 0.5, 0.0)[min(r, 2)]
            if d["why"] == "noyear" and d["k"] == s["k"]:
                w = max(1.0, w + 0.5)
            if w <= 0:
                continue
            x, y, bw, bh = s["box"]
            e = el.get(s["name"], {})
            for k in ("value", "year"):
                if e.get(k):
                    g = e[k]
                    c.append([x + (g[0] + g[2]) / 2, y + (g[1] + g[3]) / 2, w, k, s["name"]])
            c.append([x + 0.23 * bw, y + 0.83 * bh, w, "bottom-left", s["name"]])
            c.append([x + 0.82 * bw, y + 0.24 * bh, w, "top-right", s["name"]])
            img = cv2.imread(str(HERE / "crops" / (s["name"] + ".png")))
            wn = patch_cue.windows(img)
            if len(wn) >= 30:
                z = np.maximum(-patch_cue.rz(wn[:, 2]), -patch_cue.rz(wn[:, 3]))
                got = []
                for i in np.argsort(-z):
                    p = wn[i, :2]
                    if all(np.abs(p - q).max() > 40 for q in got):
                        got.append(p)
                    if len(got) == 2:
                        break
                for p in got:
                    c.append([x + float(p[0]), y + float(p[1]), w, "flat patch", s["name"]])
        pages.append(dict(album=al, page=pg, strong=False, cands=c))
    json.dump(pages, open(HERE / "candidates.json", "w"), indent=0)
    return pages


# ---------------- templates ----------------
def base_glyph(font, ch):
    if font == "hershey":
        im = np.zeros((420, 420), np.uint8)
        cv2.putText(im, ch, (60, 330), cv2.FONT_HERSHEY_SIMPLEX, 10, 255, 3)
    else:
        f = ImageFont.truetype(font, 300)
        pil = Image.new("L", (520, 520), 0)
        ImageDraw.Draw(pil).text((80, 40), ch, 255, font=f)
        im = np.array(pil)
    ys, xs = np.nonzero(im > 127)
    return (im[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 127).astype(np.uint8)


_T = None


def templates():
    """List of (letter, font_i, h, aspect_i, flipped dilated mask as float32, r)."""
    global _T
    if _T is not None:
        return _T
    out = []
    for ch in LETTERS:
        for fi, font in enumerate(FONTS):
            g = base_glyph(font, ch)
            r = R_FONT.get(font, 5)
            for h in HEIGHTS:
                for ai, asp in enumerate(ASPECTS):
                    w = max(6, int(round(h * g.shape[1] / g.shape[0] * asp)))
                    if w + 2 * r > PW + 2 * PAD - 2:
                        continue
                    m = cv2.resize(g * 255, (w, h), interpolation=cv2.INTER_AREA) > 60
                    m = np.pad(m.astype(np.uint8), r)
                    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
                    out.append((ch, fi, h, ai, np.ascontiguousarray(m[::-1, ::-1]).astype(np.float32), r))
    _T = out
    return out


def render(ch, fi, h, ai, ox, oy):
    """Glyph mask (not dilated) on the page/4 canvas, for overlays."""
    g = base_glyph(FONTS[fi], ch)
    r = R_FONT.get(FONTS[fi], 5)
    w = max(6, int(round(h * g.shape[1] / g.shape[0] * ASPECTS[ai])))
    m = cv2.resize(g * 255, (w, h), interpolation=cv2.INTER_AREA) > 60
    canvas = np.zeros((PH + 2 * PAD, PW + 2 * PAD), bool)
    canvas[oy + r:oy + r + h, ox + r:ox + r + w] = m
    return canvas[PAD:PAD + PH, PAD:PAD + PW]


# ---------------- scoring ----------------
def score_set(job):
    name, pages = job           # pages: list of float arrays (n,3): x/4+PAD, y/4+PAD, weight
    T = templates()
    best = {ch: (-1.0, None) for ch in LETTERS}
    CH, CW = PH + 2 * PAD, PW + 2 * PAD
    for ch, fi, h, ai, Df, r in T:
        Hd, Wd = Df.shape
        GH, GW = CH - Hd + 1, CW - Wd + 1
        if GH <= 0 or GW <= 0:
            continue
        total = np.zeros((GH, GW), np.float32)
        pm = np.zeros((GH, GW), np.float32)
        for c in pages:
            pm[:] = 0
            for cx, cy, w in c:
                cx, cy = int(cx), int(cy)
                a, b = max(0, cy - Hd + 1), min(GH - 1, cy)
                a2, b2 = max(0, cx - Wd + 1), min(GW - 1, cx)
                if a > b or a2 > b2:
                    continue
                sl = pm[a:b + 1, a2:b2 + 1]
                np.maximum(sl, w * Df[a + Hd - 1 - cy:b + Hd - cy, a2 + Wd - 1 - cx:b2 + Wd - cx], out=sl)
            total += pm
        i = int(total.argmax())
        s = float(total.flat[i])
        if s > best[ch][0]:
            best[ch] = (s, [fi, h, ai, i % GW, i // GW])
    return name, best


def to_arrays(pages):
    out = []
    for p in pages:
        a = np.array([[c[0] / S + PAD, c[1] / S + PAD, c[2]] for c in p["cands"]], np.float32).reshape(-1, 3)
        a[:, 0] = np.clip(a[:, 0], 0, PW + 2 * PAD - 1)
        a[:, 1] = np.clip(a[:, 1], 0, PH + 2 * PAD - 1)
        out.append(a)
    return out


def jobs(pages):
    albums = sorted({p["album"] for p in pages})
    out = []
    for al in albums:
        mine = [p for p in pages if p["album"] == al]
        out.append((al, to_arrays(mine)))
    for al in albums:
        mine = [p for p in pages if p["album"] == al]
        rest_s = [p for p in pages if p["album"] != al and p["strong"]]
        rest_n = [p for p in pages if p["album"] != al and not p["strong"]]
        ns = sum(p["strong"] for p in mine)
        rng = np.random.default_rng(int(al[-1]) * 1000)
        for k in range(N_CTRL):
            pick = [rest_s[i] for i in rng.choice(len(rest_s), ns, replace=False)] + \
                   [rest_n[i] for i in rng.choice(len(rest_n), len(mine) - ns, replace=False)]
            out.append((f"{al}_ctrl_{k:02d}", to_arrays(pick)))
    return out


def run(workers):
    cp = HERE / "candidates.json"
    pages = json.load(open(cp)) if cp.exists() else build_candidates()
    print(len(pages), "pages,", sum(len(p["cands"]) for p in pages), "candidate points", flush=True)
    rp = HERE / "joint_fit_results.json"
    res = json.load(open(rp)) if rp.exists() else {}
    todo = [j for j in jobs(pages) if j[0] not in res]
    print(len(todo), "sets to score,", len(res), "already saved", flush=True)
    with Pool(workers) as pool:
        for name, best in pool.imap_unordered(score_set, todo):
            res[name] = best
            json.dump(res, open(rp, "w"))
            top = sorted(best.items(), key=lambda t: -t[1][0])[:3]
            print(name, " ".join(f"{c}:{v[0]:.1f}" for c, v in top), flush=True)


def report():
    pages = json.load(open(HERE / "candidates.json"))
    res = json.load(open(HERE / "joint_fit_results.json"))
    albums = sorted({p["album"] for p in pages})
    panels, summary = [], {}
    for al in albums:
        if al not in res:
            continue
        mine = [p for p in pages if p["album"] == al]
        ns = sum(p["strong"] for p in mine)
        maxs = 3 * ns + sum(max((c[2] for c in p["cands"]), default=0) for p in mine if not p["strong"])
        ctr = [res[k] for k in sorted(res) if k.startswith(al + "_ctrl_")]
        rows = []
        for ch in LETTERS:
            s = res[al][ch][0]
            nul = np.array([c[ch][0] for c in ctr]) if ctr else np.array([0.0])
            rows.append((ch, s, nul.mean(), nul.std(), (s - nul.mean()) / (nul.std() + 0.5)))
        best_raw = max(r[1] for r in rows)
        null_best = [max(v[0] for v in c.values()) for c in ctr]
        byz = sorted(rows, key=lambda r: -r[4])
        print(f"{al}: {ns} strong pages, max possible {maxs:.1f}, {len(ctr)} controls")
        print("   top by z   : " + "  ".join(f"{c} {s:.1f} (ctrl {m:.1f}+-{sd:.1f}, z {z:+.1f})" for c, s, m, sd, z in byz[:5]))
        print("   top by raw : " + "  ".join(f"{c} {s:.1f}" for c, s, _, _, _ in sorted(rows, key=lambda r: -r[1])[:5]))
        if null_best:
            print(f"   best raw over all letters {best_raw:.1f}; on controls {np.mean(null_best):.1f}+-{np.std(null_best):.1f} "
                  f"(max {max(null_best):.1f}); controls >= real: {sum(n >= best_raw for n in null_best)}/{len(null_best)}")
        summary[al] = dict(top_z=[(c, round(s, 1), round(float(m), 1), round(float(z), 2)) for c, s, m, sd, z in byz[:5]],
                           best_raw=best_raw, control_best=[round(n, 1) for n in null_best])
        for ch, s, m, sd, z in byz[:3]:
            fi, h, ai, ox, oy = res[al][ch][1]
            g = render(ch, fi, h, ai, ox, oy)
            v = np.full((PH * 2, PW * 2, 3), 255, np.uint8)
            v[cv2.resize(g.astype(np.uint8), (PW * 2, PH * 2), interpolation=cv2.INTER_NEAREST) > 0] = (190, 230, 190)
            for p in mine:
                for c in p["cands"]:
                    q = (int(c[0] / 2), int(c[1] / 2))
                    if c[2] >= 3:
                        cv2.circle(v, q, 9, (0, 0, 0), -1)
                    else:
                        cv2.circle(v, q, 4, (200, 90, 0) if c[2] >= 1 else (190, 190, 190), -1)
            cv2.rectangle(v, (0, 0), (PW * 2 - 1, PH * 2 - 1), (0, 0, 0), 2)
            cv2.putText(v, f"{al}  {ch}  {s:.1f} (ctrl {m:.1f}, z {z:+.1f})", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            panels.append(v)
    json.dump(summary, open(HERE / "joint_fit_summary.json", "w"), indent=1)
    if panels:
        rows = [np.hstack(panels[i:i + 3]) for i in range(0, len(panels), 3)]
        cv2.imwrite(str(HERE / "joint_fit_overlays.png"), np.vstack(rows))
        print("wrote joint_fit_overlays.png (black = strong anchors, blue = weight >= 1 candidates, grey = weight 0.5)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.report:
        report()
    else:
        run(a.workers)
        report()
