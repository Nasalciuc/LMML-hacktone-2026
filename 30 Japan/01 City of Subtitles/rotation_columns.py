"""Toorisugari's 45 orientation bits as a 5 x 9 grid (week x photo), read as 5-bit letters by rows and by
columns, MSB/LSB first, plain and inverted. Result: nothing readable (columns give WET?MXDLT at best)."""
import csv
rows = [r for r in csv.DictReader(open("attribution_table.csv", encoding="utf-8"))]
T = sorted([r for r in rows if r["photographer"] == "toorisugari"], key=lambda r: r["file"])
bits = [int(r["rotated180"]) for r in T]
W = [bits[i * 9:(i + 1) * 9] for i in range(5)]
L = lambda v: chr(64 + v) if 1 <= v <= 26 else "?"
for inv in (0, 1):
    for msb in (True, False):
        cols = []
        for c in range(9):
            b = [W[k][c] ^ inv for k in range(5)]
            cols.append(int("".join(map(str, b if msb else b[::-1])), 2))
        print("columns inv", inv, "msb", msb, "".join(L(v) for v in cols))
    s = "".join(str(b ^ inv) for b in bits)
    print("rows    inv", inv, "".join(L(int(s[i:i + 5], 2)) for i in range(0, 45, 5)))
