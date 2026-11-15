import json, math, itertools
d = json.load(open("evals/attempt-1.chart.json", encoding="utf-8"))
stars = d["stars"]
guides = [s for s in stars if s["size"] > 6]
comps = [s for s in stars if s["size"] <= 6]
def dist(a, b): return math.hypot(a["x"]-b["x"], a["y"]-b["y"])
assign = {}
for c in comps:
    g = min(guides, key=lambda g: dist(c, g))
    assign.setdefault(id(g), []).append(c)
top = min(guides, key=lambda s: s["y"]); bottom = max(guides, key=lambda s: s["y"])
left = min(guides, key=lambda s: s["x"]); right = max(guides, key=lambda s: s["x"])
faint = min(guides, key=lambda s: s["size"])
geo = [bottom, left, top, right, faint]

def grp(order_g, withg=False, far=False):
    seq = []
    for g in order_g:
        if withg: seq.append(g["text"])
        for c in sorted(assign.get(id(g), []), key=lambda c: dist(c, g), reverse=far):
            seq.append(c["text"])
    return "".join(seq)

orders = {}
orders["geo"] = grp(geo)
orders["geo_g"] = grp(geo, withg=True)
orders["geo_far"] = grp(geo, far=True)
orders["geo_rev"] = grp(geo[::-1])
orders["geo_rev_g"] = grp(geo[::-1], withg=True)
orders["raw"] = "".join(c["text"] for c in comps)
orders["raw_all"] = "".join(s["text"] for s in stars)
orders["yx"] = "".join(c["text"] for c in sorted(comps, key=lambda c:(c["y"], c["x"])))
orders["xy"] = "".join(c["text"] for c in sorted(comps, key=lambda c:(c["x"], c["y"])))
orders["yx_all"] = "".join(s["text"] for s in sorted(stars, key=lambda c:(c["y"], c["x"])))

THEME = ("STAR HOME WARD NAVIGAT VOYAG CROSS SOUTH NORTH WIND WAVE OCEAN ISLAND "
         "PATH WAYFIND POLAR GUIDE LIGHT NIGHT SAIL MOON COMPASS CHART HORIZON "
         "CELEST MARINER VASA FETU SAMOA PACIFIC APIA FOLAU MALAGA TAUTAI TAPUITEA "
         "MATAGI HOKULE AVANOA FETUAO SIGMOID THE AND ETU AITU VAA").split()
BIG = {'TH':3.6,'HE':3.1,'IN':2.4,'ER':2.1,'AN':2.0,'RE':1.7,'ON':1.6,'AT':1.5,'EN':1.4,
 'ND':1.4,'TI':1.4,'ES':1.4,'OR':1.3,'TE':1.3,'ED':1.3,'IS':1.3,'IT':1.2,'AL':1.2,
 'AR':1.2,'ST':1.2,'TO':1.2,'NT':1.1,'NG':1.1,'SE':1.1,'HA':1.1,'AS':1.1,'OU':1.0,
 'IO':1.0,'LE':1.0,'VE':1.0,'CO':0.9,'ME':0.9,'DE':0.9,'HI':0.9,'RI':0.9,'RO':0.9,
 'IC':0.8,'NE':0.8,'EA':0.8,'RA':0.8,'CE':0.8,'LI':0.8,'CH':0.7,'LL':0.7,'BE':0.7,
 'MA':0.7,'SI':0.7,'OM':0.7,'UR':0.7,'CA':0.6,'EL':0.6,'TA':0.6,'LA':0.6,'NA':0.6,
 'SO':0.6,'HO':0.6,'FO':0.6,'UT':0.5,'WA':0.5,'GE':0.5,'SS':0.5,'MO':0.5,'UN':0.5,
 'AU':0.6,'AI':0.6,'UE':0.4,'AO':0.4,'AA':0.3,'OA':0.4,'AG':0.4,'GA':0.4}
def score(s):
    sc = sum(BIG.get(s[i:i+2], 0) for i in range(len(s)-1))
    for w in THEME:
        if w in s and len(w) >= 3: sc += len(w)*2.2
    v = sum(c in "AEIOU" for c in s)/len(s)
    sc += -4 if (v < 0.2 or v > 0.75) else 0
    return sc

def prog(t, a, b, sign):
    return "".join(chr((ord(c)-65 + sign*(a+b*i)) % 26 + 65) for i, c in enumerate(t))

best = {}
for onm, t in orders.items():
    for a in range(26):
        for b in range(26):
            for sign in (-1, 1):
                s = prog(t, a, b, sign)
                key = s
                sc = score(s)
                if key not in best or sc > best[key][0]:
                    best[key] = (sc, f"{onm}|a{a},b{b},{'+' if sign>0 else '-'}")
ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
print("orders:")
for k, v in orders.items():
    print(f"  {k:10s} {v}")
print("\nTOP 45 progressive:")
for s, (sc, nm) in ranked[:45]:
    print(f"  {sc:5.1f} {nm:22s} {s}")
