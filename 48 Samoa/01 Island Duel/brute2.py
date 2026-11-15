import json, math, itertools
d = json.load(open("evals/attempt-1.chart.json", encoding="utf-8"))
stars = d["stars"]
guides = [s for s in stars if s["size"] > 6]
comps = [s for s in stars if s["size"] <= 6]
def dist(a, b): return math.hypot(a["x"]-b["x"], a["y"]-b["y"])

# English quad/bigram-ish scorer using common English bigram log-weights
BIGRAMS = {
 'TH':3.6,'HE':3.1,'IN':2.4,'ER':2.1,'AN':2.0,'RE':1.7,'ON':1.6,'AT':1.5,'EN':1.4,
 'ND':1.4,'TI':1.4,'ES':1.4,'OR':1.3,'TE':1.3,'OF':1.3,'ED':1.3,'IS':1.3,'IT':1.2,
 'AL':1.2,'AR':1.2,'ST':1.2,'TO':1.2,'NT':1.1,'NG':1.1,'SE':1.1,'HA':1.1,'AS':1.1,
 'OU':1.0,'IO':1.0,'LE':1.0,'VE':1.0,'CO':0.9,'ME':0.9,'DE':0.9,'HI':0.9,'RI':0.9,
 'RO':0.9,'IC':0.8,'NE':0.8,'EA':0.8,'RA':0.8,'CE':0.8,'LI':0.8,'CH':0.7,'LL':0.7,
 'BE':0.7,'MA':0.7,'SI':0.7,'OM':0.7,'UR':0.7,'CA':0.6,'EL':0.6,'TA':0.6,'LA':0.6,
 'NS':0.6,'DI':0.6,'FO':0.6,'HO':0.6,'PE':0.6,'EC':0.6,'PR':0.6,'NA':0.6,'SO':0.6,
 'UT':0.5,'WA':0.5,'GE':0.5,'SS':0.5,'PO':0.5,'WE':0.5,'DA':0.5,'EE':0.5,'EO':0.5,
 'MO':0.5,'UN':0.5,'IG':0.4,'VI':0.4,'GA':0.4,'YA':0.3,'AY':0.4,'KE':0.4,'SK':0.3,
}
THEME = ("STAR HOME WARD NAVIGAT VOYAG CROSS SOUTH NORTH WIND WAVE OCEAN ISLAND "
         "PATH WAYFIND POLAR GUIDE LIGHT NIGHT SAIL MOON COMPASS CHART HORIZON "
         "CELEST MARINER VASA FETU SAMOA PACIFIC APIA OVER").split()
def score(s):
    sc = sum(BIGRAMS.get(s[i:i+2], 0) for i in range(len(s)-1))
    v = sum(c in "AEIOU" for c in s) / len(s)
    sc += -4 if (v < 0.18 or v > 0.72) else 0
    for w in THEME:
        if w in s and len(w) >= 4:
            sc += len(w) * 2
    return sc

assign = {}
for c in comps:
    g = min(guides, key=lambda g: dist(c, g))
    assign.setdefault(id(g), []).append(c)

top = min(guides, key=lambda s: s["y"]); bottom = max(guides, key=lambda s: s["y"])
left = min(guides, key=lambda s: s["x"]); right = max(guides, key=lambda s: s["x"])
faint = min(guides, key=lambda s: s["size"])
geo = [bottom, left, top, right, faint]

def grp(order_g, withg=False, far_first=False):
    seq = []
    for g in order_g:
        if withg: seq.append(g["text"])
        cs = sorted(assign.get(id(g), []), key=lambda c: dist(c, g), reverse=far_first)
        seq += [c["text"] for c in cs]
    return "".join(seq)

orders = {}
orders["geo"] = grp(geo)
orders["geo_far"] = grp(geo, far_first=True)
orders["geo_rev"] = grp(geo[::-1])
orders["geo_g"] = grp(geo, withg=True)
for perm in itertools.permutations(guides):
    orders["gp_" + "".join(g["text"] for g in perm)] = grp(perm)
orders["yx"] = "".join(c["text"] for c in sorted(comps, key=lambda c:(c["y"], c["x"])))
orders["xy"] = "".join(c["text"] for c in sorted(comps, key=lambda c:(c["x"], c["y"])))
orders["raw"] = "".join(c["text"] for c in comps)

def caesar(t, k): return "".join(chr((ord(c)-65-k) % 26 + 65) for c in t)
def vig(t, key, sign):
    return "".join(chr((ord(c)-65+sign*(ord(key[i % len(key)])-65)) % 26 + 65) for i, c in enumerate(t))

guide_letters = [g["text"] for g in guides]
keyset = set()
for perm in itertools.permutations(guide_letters):
    keyset.add("".join(perm))
keyset |= {"SIGMOID", "SAMOA", "HOMEWARD", "HOME", "STAR", "FETU", "VASA",
           "CROSS", "SOUTH", "GREEK", "OVER", "WATCH", "COMPASS", "POLARIS",
           "SIGMOIDSAMOA"}

cands = []
for onm, t in orders.items():
    for k in range(26):
        cands.append((f"{onm}|C{k}", caesar(t, k)))
    for key in keyset:
        cands.append((f"{onm}|Vd-{key}", vig(t, key, -1)))
        cands.append((f"{onm}|Ve-{key}", vig(t, key, +1)))

best = {}
for nm, s in cands:
    if s not in best or score(s) > best[s][0]:
        best[s] = (score(s), nm)
ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
print("TOP 50:")
for s, (sc, nm) in ranked[:50]:
    print(f"  {sc:6.1f}  {nm:26s} {s}")
