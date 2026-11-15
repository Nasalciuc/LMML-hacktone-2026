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

def grp(order_g, withg=False, far=False):
    seq = []
    for g in order_g:
        if withg: seq.append(g["text"])
        for c in sorted(assign.get(id(g), []), key=lambda c: dist(c, g), reverse=far):
            seq.append(c["text"])
    return "".join(seq)

orders = {}
# all permutations of guides, grouped companions, both near-first and far-first, with/without guides
for perm in itertools.permutations(guides):
    tag = "".join(g["text"] for g in perm)
    orders["c_" + tag] = grp(perm)
    orders["cf_" + tag] = grp(perm, far=True)
    orders["g_" + tag] = grp(perm, withg=True)
# global sorts
orders["yx"] = "".join(c["text"] for c in sorted(comps, key=lambda c:(c["y"], c["x"])))
orders["xy"] = "".join(c["text"] for c in sorted(comps, key=lambda c:(c["x"], c["y"])))
orders["yx_all"] = "".join(s["text"] for s in sorted(stars, key=lambda c:(c["y"], c["x"])))
orders["xy_all"] = "".join(s["text"] for s in sorted(stars, key=lambda c:(c["x"], c["y"])))
# add reversed versions
for k in list(orders):
    orders[k + "_R"] = orders[k][::-1]

THEME = ("STAR HOME WARD NAVIGAT VOYAG CROSS SOUTH NORTH WIND WAVE OCEAN ISLAND PATH "
         "WAYFIND POLAR GUIDE LIGHT NIGHT SAIL MOON COMPASS CHART HORIZON CELEST MARINER "
         "VASA FETU SAMOA PACIFIC APIA FOLAU MALAGA TAUTAI TAPUITEA MATAGI HOKULE AVANOA "
         "FETUAO SEXTANT ASTRO LATITUD PACIFI TUSITALA VAA AITU ATUA MANAIA TALA "
         "RETURN SKY DAWN").split()
BIG = {'TH':3.6,'HE':3.1,'IN':2.4,'ER':2.1,'AN':2.0,'RE':1.7,'ON':1.6,'AT':1.5,'EN':1.4,
 'ND':1.4,'TI':1.4,'ES':1.4,'OR':1.3,'TE':1.3,'ED':1.3,'IS':1.3,'IT':1.2,'AL':1.2,'AR':1.2,
 'ST':1.2,'TO':1.2,'NT':1.1,'NG':1.1,'SE':1.1,'HA':1.1,'AS':1.1,'OU':1.0,'IO':1.0,'LE':1.0,
 'VE':1.0,'CO':0.9,'ME':0.9,'DE':0.9,'HI':0.9,'RI':0.9,'RO':0.9,'IC':0.8,'NE':0.8,'EA':0.8,
 'RA':0.8,'CE':0.8,'LI':0.8,'CH':0.7,'LL':0.7,'MA':0.7,'SI':0.7,'UR':0.7,'CA':0.6,'EL':0.6,
 'TA':0.6,'LA':0.6,'NA':0.6,'SO':0.6,'HO':0.6,'FO':0.6,'AU':0.6,'AI':0.6,'OA':0.5}
def score(s):
    sc = sum(BIG.get(s[i:i+2], 0) for i in range(len(s)-1))
    for w in THEME:
        if w in s and len(w) >= 3: sc += len(w)*2.5
    v = sum(c in "AEIOU" for c in s)/len(s)
    sc += -5 if (v < 0.22 or v > 0.72) else 0
    return sc

def vig_dec(t, k): return "".join(chr((ord(c)-ord(k[i % len(k)])) % 26 + 65) for i, c in enumerate(t))
def vig_enc(t, k): return "".join(chr((ord(c)+ord(k[i % len(k)])-130) % 26 + 65) for i, c in enumerate(t))
def beaufort(t, k): return "".join(chr((ord(k[i % len(k)])-ord(c)) % 26 + 65) for i, c in enumerate(t))

KEYS = ["SIGMOID", "DIOMGIS", "SAMOA", "AOMAS", "HOMEWARD", "HOME", "STAR", "FETU",
        "VASA", "CROSS", "SOUTH", "GREEK", "OVER", "WATCH", "COMPASS", "POLARIS",
        "NIGHT", "NAVIGATOR", "VOYAGER", "FOUNDATION", "CHART", "FIVE", "LIGHTS"]
best = {}
for onm, t in orders.items():
    for k in KEYS:
        for fn, fnm in ((vig_dec, "Vd"), (vig_enc, "Ve"), (beaufort, "Bf")):
            s = fn(t, k)
            if s not in best or score(s) > best[s][0]:
                best[s] = (score(s), f"{onm}|{fnm}-{k}")
ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
print("TOP 45 (key families x all orders):")
for s, (sc, nm) in ranked[:45]:
    print(f"  {sc:5.1f} {nm:26s} {s}")
