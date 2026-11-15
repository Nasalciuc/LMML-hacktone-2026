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
geo = [bottom, left, top, right, faint]  # alpha..epsilon

groups = []  # list of lists of companion letters, near->far, in alpha..epsilon order
for g in geo:
    cs = sorted(assign.get(id(g), []), key=lambda c: dist(c, g))
    groups.append([c["text"] for c in cs])
print("groups (alpha..epsilon):", groups)
print("guide letters alpha..eps:", [g["text"] for g in geo])

BIG = {
 'TH':3.6,'HE':3.1,'IN':2.4,'ER':2.1,'AN':2.0,'RE':1.7,'ON':1.6,'AT':1.5,'EN':1.4,
 'ND':1.4,'TI':1.4,'ES':1.4,'OR':1.3,'TE':1.3,'OF':1.3,'ED':1.3,'IS':1.3,'IT':1.2,
 'AL':1.2,'AR':1.2,'ST':1.2,'TO':1.2,'NT':1.1,'NG':1.1,'SE':1.1,'HA':1.1,'AS':1.1,
 'OU':1.0,'IO':1.0,'LE':1.0,'VE':1.0,'CO':0.9,'ME':0.9,'DE':0.9,'HI':0.9,'RI':0.9,
 'RO':0.9,'IC':0.8,'NE':0.8,'EA':0.8,'RA':0.8,'CE':0.8,'LI':0.8,'CH':0.7,'LL':0.7,
 'BE':0.7,'MA':0.7,'SI':0.7,'OM':0.7,'UR':0.7,'CA':0.6,'EL':0.6,'TA':0.6,'LA':0.6,
 'NS':0.6,'DI':0.6,'FO':0.6,'HO':0.6,'PE':0.6,'EC':0.6,'PR':0.6,'NA':0.6,'SO':0.6,
 'UT':0.5,'WA':0.5,'GE':0.5,'SS':0.5,'PO':0.5,'WE':0.5,'DA':0.5,'EE':0.5,'MO':0.5,
 'UN':0.5,'ID':0.5,'GA':0.4,'AY':0.4,'KE':0.4,'VA':0.4,'YF':0.2,'WY':0.2,
}
def bgw(a, b): return BIG.get(a+b, 0.0)
def shifted(group, s): return [chr((ord(c)-65-s) % 26 + 65) for c in group]

# precompute each group's string + its first/last char for each shift
gvar = []
for grp in groups:
    vs = []
    for s in range(26):
        sg = shifted(grp, s)
        inner = sum(bgw(sg[i], sg[i+1]) for i in range(len(sg)-1))
        vs.append(("".join(sg), sg[0], sg[-1], inner))
    gvar.append(vs)

best = []
for s0 in range(26):
    a = gvar[0][s0]
    for s1 in range(26):
        b = gvar[1][s1]
        for s2 in range(26):
            c = gvar[2][s2]
            for s3 in range(26):
                dd = gvar[3][s3]
                for s4 in range(26):
                    e = gvar[4][s4]
                    sc = (a[3]+b[3]+c[3]+dd[3]+e[3]
                          + bgw(a[2], b[1]) + bgw(b[2], c[1])
                          + bgw(c[2], dd[1]) + bgw(dd[2], e[1]))
                    if sc > 9.0:
                        best.append((sc, (s0, s1, s2, s3, s4),
                                     a[0]+b[0]+c[0]+dd[0]+e[0]))
best.sort(reverse=True)
print(f"\n{len(best)} combos scored >9.0; top 40:")
for sc, sh, w in best[:40]:
    print(f"  {sc:5.1f} shifts={sh} {w}")
