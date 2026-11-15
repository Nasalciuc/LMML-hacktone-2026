import json, math, itertools
d = json.load(open("evals/attempt-1.chart.json", encoding="utf-8"))
stars = d["stars"]
guides = [s for s in stars if s["size"] > 6]
comps = [s for s in stars if s["size"] <= 6]

def dist(a, b): return math.hypot(a["x"]-b["x"], a["y"]-b["y"])

# ---- English-likeness score via bigram frequencies (log) ----
COMMON = ("THE AND ING ION ENT STAR HOME WARD NAVIG VOYAG CROSS SOUTH NORTH "
          "WIND WAVE SEA OCEAN ISLAND PATH WAY FIND POLAR GUIDE LIGHT NIGHT SKY "
          "SAIL MOON SUN FETU VASA SAMOA PACIF COMPASS CHART MESSAGE HORIZON").split()
BIG = {}
for w in ("the be to of and a in that have i it for not on with he as you do at this but his by "
          "from navigation voyager wayfinding southern cross polaris constellation homeward "
          "mariner compass horizon starlight celestial astronomy latitude").split():
    w = w.upper()
    for i in range(len(w)-1):
        BIG[w[i:i+2]] = BIG.get(w[i:i+2], 0) + 1
def score(s):
    sc = sum(BIG.get(s[i:i+2], 0) for i in range(len(s)-1))
    vowels = sum(c in "AEIOU" for c in s)
    r = vowels/len(s)
    bonus = sum(5 for w in COMMON if w in s and len(w) >= 4)
    bonus += 20 if any(w in s for w in COMMON if len(w) >= 5) else 0
    pen = -3 if (r < 0.15 or r > 0.75) else 0
    return sc + bonus + pen

# ---- companion orderings ----
assign = {}
for c in comps:
    g = min(guides, key=lambda g: dist(c, g))
    assign.setdefault(id(g), []).append(c)

def group_order(order_guides, with_guides=False):
    seq = []
    for g in order_guides:
        if with_guides: seq.append(g["text"])
        for c in sorted(assign.get(id(g), []), key=lambda c: dist(c, g)):
            seq.append(c["text"])
    return "".join(seq)

top = min(guides, key=lambda s: s["y"]); bottom = max(guides, key=lambda s: s["y"])
left = min(guides, key=lambda s: s["x"]); right = max(guides, key=lambda s: s["x"])
faint = min(guides, key=lambda s: s["size"])
geo = [bottom, left, top, right, faint]  # alpha..epsilon by cross position

orders = {}
orders["geo"] = group_order(geo)
orders["geo_g"] = group_order(geo, True)
for nm, key in [("bry", lambda s:(-s["size"], s["y"])), ("brx", lambda s:(-s["size"], s["x"])),
                ("bra", lambda s:(-s["size"], math.atan2(s["y"]-100, s["x"]-100)))]:
    orders[nm] = group_order(sorted(guides, key=key))
orders["raw"] = "".join(c["text"] for c in comps)
orders["yx"] = "".join(c["text"] for c in sorted(comps, key=lambda c:(c["y"], c["x"])))
orders["xy"] = "".join(c["text"] for c in sorted(comps, key=lambda c:(c["x"], c["y"])))
orders["yx_rev"] = orders["yx"][::-1]
# guides only orderings
orders["guides_geo"] = "".join(g["text"] for g in geo)
orders["guides_yx"] = "".join(g["text"] for g in sorted(guides, key=lambda s:(s["y"], s["x"])))
# all stars geo-grouped
orders["all_geo"] = group_order(geo, True)

# ---- ciphers ----
def caesar(t, k): return "".join(chr((ord(c)-65-k) % 26 + 65) for c in t)
def vig(t, key, sign=-1):
    return "".join(chr((ord(c)-65 + sign*(ord(key[i % len(key)])-65)) % 26 + 65) for i, c in enumerate(t))
def prog(t, start, step, sign=-1):
    return "".join(chr((ord(c)-65 + sign*(start+step*i)) % 26 + 65) for i, c in enumerate(t))

KEYS = ["SIGMOID", "SAMOA", "HOME", "STAR", "FETU", "VASA", "CROSS", "SOUTH",
        "NETCI", "TNECI", "CENTI", "APIA", "HOMEWARD", "NIGHT", "SKY", "WATCH",
        "GREEK", "FIVE", "LIGHTS", "COMPASS", "VOYAGE", "POLARIS"]

cands = []
for onm, t in orders.items():
    for k in range(26):
        cands.append((f"{onm}|caesar-{k}", caesar(t, k)))
    for key in KEYS:
        cands.append((f"{onm}|vigdec-{key}", vig(t, key, -1)))
        cands.append((f"{onm}|vigenc-{key}", vig(t, key, +1)))
    for start in range(0, 6):
        for step in (1, -1, 2):
            cands.append((f"{onm}|prog{start},{step}", prog(t, start, step, -1)))

cands = [(nm, s) for nm, s in cands]
cands.sort(key=lambda x: -score(x[1]))
print("ciphertext orders:")
for onm, t in orders.items():
    print(f"  {onm:12s} {t}")
print("\nTOP 40 candidates by English-likeness:")
seen = set()
n = 0
for nm, s in cands:
    if s in seen: continue
    seen.add(s)
    print(f"  {score(s):4d}  {nm:22s} {s}")
    n += 1
    if n >= 40: break
