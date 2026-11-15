"""Solve the homeward-chart cipher.

Verse:
  1 "The five lights of home keep the Greek watch."   -> 5 guide stars (size>6)
     get Greek letters alpha..epsilon by brightness (size); they are the key.
  2 "Each keeps its nearer companions before the farther." -> each companion
     belongs to its nearest guide; within a guide read nearest->farthest.
  3 "Our foundation turns the alphabet, over and over." -> a Caesar rotation,
     applied cumulatively.
  4 "For the homeward message, undo those turns."      -> decrypt.
"""

import json
import math
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "evals/attempt-1.chart.json",
                   encoding="utf-8"))
stars = d["stars"]
guides = [s for s in stars if s["size"] > 6]
comps = [s for s in stars if s["size"] <= 6]
print("prefix", d["prefix"])
print("guides:", [(s["text"], s["size"], s["x"], s["y"]) for s in guides])
print("comps :", [(s["text"]) for s in comps], len(comps))


def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def dec(ch, shift):
    return chr((ord(ch) - 65 - shift) % 26 + 65)


def enc(ch, shift):
    return chr((ord(ch) - 65 + shift) % 26 + 65)


def greek_order(tiebreak):
    # brightest first (largest size); tie-break per option
    keys = {
        "x": lambda s: (-s["size"], s["x"]),
        "y": lambda s: (-s["size"], s["y"]),
        "angle": lambda s: (-s["size"], math.atan2(s["y"] - cy, s["x"] - cx)),
        "letter": lambda s: (-s["size"], s["text"]),
        "x-y": lambda s: (-s["size"], s["x"], s["y"]),
    }
    return sorted(guides, key=keys[tiebreak])


cx = sum(s["x"] for s in guides) / len(guides)
cy = sum(s["y"] for s in guides) / len(guides)

# assign each companion to nearest guide
assign = {}
for c in comps:
    g = min(guides, key=lambda g: dist(c, g))
    assign.setdefault(id(g), []).append(c)

for tb in ["x", "y", "angle", "letter", "x-y"]:
    go = greek_order(tb)
    # build reading order: guides in greek order; within each, companions near->far
    seq = []       # (letter, greek_index(1..5), order_in_guide(1..))
    for gi, g in enumerate(go, 1):
        cs = sorted(assign.get(id(g), []), key=lambda c: dist(c, g))
        for oi, c in enumerate(cs, 1):
            seq.append((c["text"], gi, oi))
    cipher = "".join(t for t, _, _ in seq)
    greek_letters = "".join(g["text"] for g in go)
    print(f"\n=== tiebreak={tb}  greek(bright->faint)={greek_letters} ===")
    print("companion cipher (reading order):", cipher)
    # scheme A: fixed caesar, all shifts
    print(" fixed shifts:")
    for sh in range(1, 26):
        print(f"   -{sh:2d}: {''.join(dec(t, sh) for t,_,_ in seq)}")
    # scheme B: shift = greek index (1..5) per companion
    print(" shift=greek idx:", "".join(dec(t, gi) for t, gi, _ in seq))
    print(" shift=greek idx (enc):", "".join(enc(t, gi) for t, gi, _ in seq))
    # scheme C: cumulative shift by position (1,2,3,...) "over and over"
    print(" shift=position:", "".join(dec(t, i + 1) for i, (t, _, _) in enumerate(seq)))
    print(" shift=position(enc):", "".join(enc(t, i + 1) for i, (t, _, _) in enumerate(seq)))
    # scheme D: shift = greek*order
    print(" shift=greek*order:", "".join(dec(t, gi * oi) for t, gi, oi in seq))
    # scheme E: cumulative sum of greek indices
    acc = 0
    outp = []
    for t, gi, oi in seq:
        acc += gi
        outp.append(dec(t, acc))
    print(" shift=cumsum(greek):", "".join(outp))


print("\n\n########## Vigenere / per-guide-letter schemes ##########")


def vig_dec(text, key):
    return "".join(chr((ord(c) - ord(key[i % len(key)])) % 26 + 65) for i, c in enumerate(text))


def vig_enc(text, key):
    return "".join(chr((ord(c) + ord(key[i % len(key)]) - 2 * 65) % 26 + 65) for i, c in enumerate(text))


for tb in ["x", "y", "angle", "letter", "x-y"]:
    go = greek_order(tb)
    key = "".join(g["text"] for g in go)
    seq = []
    for g in go:
        cs = sorted(assign.get(id(g), []), key=lambda c: dist(c, g))
        for c in cs:
            seq.append((c["text"], g["text"]))
    cipher = "".join(t for t, _ in seq)
    print(f"\n=== tb={tb} key(greek)={key} cipher={cipher} ===")
    print("  vig_dec whole-key :", vig_dec(cipher, key))
    print("  vig_enc whole-key :", vig_enc(cipher, key))
    # per-companion shift by ITS guide letter (A=0)
    print("  per-guide dec(A=0):", "".join(chr((ord(t) - (ord(gl) - 65) - 65) % 26 + 65) for t, gl in seq))
    print("  per-guide enc(A=0):", "".join(chr((ord(t) + (ord(gl) - 65) - 65) % 26 + 65) for t, gl in seq))
    # per-companion shift by ITS guide letter (A=1)
    print("  per-guide dec(A=1):", "".join(chr((ord(t) - (ord(gl) - 64) - 65) % 26 + 65) for t, gl in seq))
