import json, math, sys
d = json.load(open("evals/attempt-1.chart.json", encoding="utf-8"))
stars = d["stars"]
guides = [s for s in stars if s["size"] > 6]
comps = [s for s in stars if s["size"] <= 6]

def dist(a, b): return math.hypot(a["x"]-b["x"], a["y"]-b["y"])

# geometric Southern-Cross greek assignment
def by(f): return sorted(guides, key=f)
top = min(guides, key=lambda s: s["y"])
bottom = max(guides, key=lambda s: s["y"])
left = min(guides, key=lambda s: s["x"])
right = max(guides, key=lambda s: s["x"])
faint = min(guides, key=lambda s: s["size"])
geo = {"alpha": bottom, "beta": left, "gamma": top, "delta": right, "epsilon": faint}
print("geometric greek:", {k: v["text"] for k, v in geo.items()})

# assign companions to nearest guide
assign = {}
for c in comps:
    g = min(guides, key=lambda g: dist(c, g))
    assign.setdefault(id(g), []).append(c)

def reading(order_guides):
    seq = []
    for g in order_guides:
        for c in sorted(assign.get(id(g), []), key=lambda c: dist(c, g)):
            seq.append(c["text"])
    return "".join(seq)

orders = {}
# geometric greek order alpha..epsilon
orders["geo_abcde"] = reading([geo["alpha"], geo["beta"], geo["gamma"], geo["delta"], geo["epsilon"]])
# brightness desc, tiebreak y
orders["bright_y"] = reading(sorted(guides, key=lambda s: (-s["size"], s["y"])))
orders["bright_x"] = reading(sorted(guides, key=lambda s: (-s["size"], s["x"])))
# raw chart order of companions
orders["raw"] = "".join(c["text"] for c in comps)
# pure global y-then-x and x-then-y of companions
orders["y_x"] = "".join(c["text"] for c in sorted(comps, key=lambda c: (c["y"], c["x"])))
orders["x_y"] = "".join(c["text"] for c in sorted(comps, key=lambda c: (c["x"], c["y"])))
# include guides+companions in geo reading (guide then its comps)
def reading_with_guides(order_guides):
    seq = []
    for g in order_guides:
        seq.append(g["text"])
        for c in sorted(assign.get(id(g), []), key=lambda c: dist(c, g)):
            seq.append(c["text"])
    return "".join(seq)
orders["geo_with_guides"] = reading_with_guides([geo["alpha"], geo["beta"], geo["gamma"], geo["delta"], geo["epsilon"]])

def vig_dec(text, key):
    return "".join(chr((ord(c)-ord(key[i % len(key)])) % 26 + 65) for i, c in enumerate(text))
def vig_enc(text, key):
    return "".join(chr((ord(c)+ord(key[i % len(key)])-130) % 26 + 65) for i, c in enumerate(text))

for key in ["SIGMOID", "SIGMOIDWORD"]:
    print(f"\n########## key={key} ##########")
    for name, txt in orders.items():
        print(f"  {name:16s} cipher={txt}")
        print(f"      vig_dec -> {vig_dec(txt, key)}")
        print(f"      vig_enc -> {vig_enc(txt, key)}")
