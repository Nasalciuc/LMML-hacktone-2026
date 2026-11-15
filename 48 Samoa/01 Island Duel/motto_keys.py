"""Test Samoa's motto 'Fa'avae i le Atua Samoa' (fa'avae = foundation) and the
1962 independence date as repeating keys over every derived reading order."""
import json, math, itertools
W = r"C:/Users/user/Projects/LMML-hacktone-2026/17 Cuba/01 If you know you know/index/words.txt"
DICT = set()
for l in open(W, encoding="utf-8", errors="ignore"):
    t = l.split()
    if t and t[0].strip().upper().isalpha():
        DICT.add(t[0].strip().upper())
d = json.load(open("evals/attempt-1.chart.json", encoding="utf-8"))
stars = d["stars"]; guides = [s for s in stars if s["size"] > 6]; comps = [s for s in stars if s["size"] <= 6]
dist = lambda a, b: math.hypot(a["x"]-b["x"], a["y"]-b["y"])
assign = {}
for c in comps:
    assign.setdefault(id(min(guides, key=lambda g: dist(c, g))), []).append(c)

# orders as lists of (letter, group_index_in_order)
orders = {}
for perm in itertools.permutations(guides):
    tag = "".join(g["text"] for g in perm)
    for far in (False, True):
        for withg in (False, True):
            seq = []
            for gi, g in enumerate(perm):
                if withg: seq.append((g["text"], gi))
                for c in sorted(assign[id(g)], key=lambda c: dist(c, g), reverse=far):
                    seq.append((c["text"], gi))
            orders[f"{'cf' if far else 'c'}{'g' if withg else ''}_{tag}"] = seq
    # tier (rank-major)
    seq = []
    for r in range(3):
        for gi, g in enumerate(perm):
            cs = sorted(assign[id(g)], key=lambda c: dist(c, g))
            if r < len(cs): seq.append((cs[r]["text"], gi))
    orders["t_" + tag] = seq
for nm, key in (("yx", lambda c: (c["y"], c["x"])), ("xy", lambda c: (c["x"], c["y"]))):
    orders[nm] = [(c["text"], 0) for c in sorted(comps, key=key)]
orders["raw"] = [(c["text"], 0) for c in comps]
orders["raw_all"] = [(s["text"], 0) for s in stars]
for k in list(orders):
    orders[k + "R"] = orders[k][::-1]

ALPHA_KEYS = ["FAAVAEILEATUASAMOA", "FAAVAEILEATUA", "FAAVAE", "ATUA", "SAMOA",
              "FAAVAEILEATUASAMOA"[::-1], "FAAVAEIATUA", "AVAE", "LEATUA"]
NUM_KEYS = {"1962": [1, 9, 6, 2], "111962": [1, 1, 1, 9, 6, 2], "01011962": [0, 1, 0, 1, 1, 9, 6, 2],
            "19620101": [1, 9, 6, 2, 0, 1, 0, 1]}
keys = {k: [ord(ch) - 65 for ch in k] for k in ALPHA_KEYS}
keys.update(NUM_KEYS)
MODES = {"vdec": lambda c, k: c - k, "venc": lambda c, k: c + k, "beau": lambda c, k: k - c}

def words_in(s):
    best = ""
    for i in range(len(s)):
        for j in range(i + 4, len(s) + 1):
            if s[i:j] in DICT and j - i > len(best): best = s[i:j]
    return best, s in DICT

hits = []
def consider(tag, s, grpseq):
    b, f = words_in(s)
    # double constraint: positions of the two Y companions (same group, both 'Y')
    if f or len(b) >= 7:
        hits.append((f, len(b), tag, s, b))

for onm, seq in orders.items():
    letters = [ord(ch) - 65 for ch, _ in seq]
    for kn, kv in keys.items():
        L = len(kv)
        for mn, fn in MODES.items():
            # repeating key with every start offset
            for off in range(L):
                s = "".join(chr(fn(c, kv[(i + off) % L]) % 26 + 65) for i, c in enumerate(letters))
                consider(f"{onm}|{mn}|{kn}|off{off}", s, seq)
            # per-group: whole group shifted by key letter at group index
            if not onm.startswith(("yx", "xy", "raw")):
                for off in range(L):
                    s = "".join(chr(fn(c, kv[(gi + off) % L]) % 26 + 65) for c, (_, gi) in zip(letters, seq))
                    consider(f"{onm}|{mn}|{kn}|pergroup{off}", s, seq)
                # key restarts at each group
                pos = {}
                out = []
                for c, (_, gi) in zip(letters, seq):
                    p = pos.get(gi, 0); pos[gi] = p + 1
                    out.append(chr(fn(c, kv[p % L]) % 26 + 65))
                consider(f"{onm}|{mn}|{kn}|restart", "".join(out), seq)

hits.sort(reverse=True)
print("exact dictionary words:", sum(1 for h in hits if h[0]))
seen = set(); n = 0
for f, bl, tag, s, b in hits:
    if s in seen: continue
    seen.add(s); n += 1
    print(f"{'EXACT' if f else '     '} {tag:40s} {s}  [{b}]")
    if n >= 40: break
if not hits: print("no hits (no word >=7 letters)")
