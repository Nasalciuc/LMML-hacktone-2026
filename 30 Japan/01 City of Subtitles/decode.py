"""Read the orientation of toorisugari's anonymous photos as a message.

Lists toorisugari's photos in posting order (week, then counter) with the orientation bit
(1 = shot upside down, sensor pattern turned 180 degrees) and tries the usual readings."""
import csv
import string
import sys

f = sys.argv[1] if len(sys.argv) > 1 else "attribution.csv"
rows = [r for r in csv.DictReader(open(f)) if r["photographer"] == "toorisugari"]
rows.sort(key=lambda r: r["file"])
for r in rows:
    print(r["file"], r["rotated180"], r["z_best"], r["z_margin"])
bits = "".join(r["rotated180"] for r in rows)
print(len(bits), bits)


def ascii8(b):
    return "".join(chr(int(b[i:i + 8], 2)) if 32 <= int(b[i:i + 8], 2) < 127 else "?" for i in range(0, len(b) - 7, 8))


def a1z26_5(b):
    out = ""
    for i in range(0, len(b) - 4, 5):
        v = int(b[i:i + 5], 2)
        out += string.ascii_uppercase[v - 1] if 1 <= v <= 26 else "?"
    return out


inv = "".join("1" if c == "0" else "0" for c in bits)
for nm, b in (("bits", bits), ("inverted", inv)):
    for off in range(8):
        print(nm, "offset", off, "ascii8:", ascii8(b[off:]), "| 5bit:", a1z26_5(b[off:]))
weeks = {}
for r in rows:
    weeks.setdefault(r["file"][:3], []).append(r["rotated180"])
for w, b in weeks.items():
    s = "".join(b)
    print(w, s, len(s), int(s, 2) if s else "")
