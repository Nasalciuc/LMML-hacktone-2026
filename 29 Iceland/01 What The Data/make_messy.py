# builds a messier export (more duplicates, broken names, F readings, longer preamble) to test main.py
import random
import sys
from decimal import Decimal, ROUND_HALF_UP

random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
stations = ["Geysir", "Mývatn", "Hveragerði", "Reykjanes", "Hveravellir", "Landmannalaugar",
            "Þingvellir", "Deildartunguhver", "Krýsuvík", "Seltún", "Kerlingarfjöll", "Ásbyrgi",
            "Hrafntinnusker", "Grábrók", "Öskjuvatn", "Ísafjörður", "Hólmavík"]


def garble(s, times, codec):
    for _ in range(times):
        b = s.encode("utf-8")
        s = b.decode(codec, errors="strict") if codec == "latin-1" else "".join(
            bytes([x]).decode("cp1252") if x not in (0x81, 0x8D, 0x8F, 0x90, 0x9D) else chr(x) for x in b)
    return s


rows, truth = [], []
for rid in range(1, 81):
    st = random.choice(stations)
    c = Decimal(random.uniform(35, 98)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    date = f"2026-{random.randint(6, 8):02d}-{random.randint(1, 28):02d}"
    truth.append(f"{rid},{st},{c},{date}")
    for _ in range(random.choice([1, 1, 2, 3])):
        if random.random() < 0.45:
            f = (c * 9 / 5 + 32).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
            raw, unit = str(f), "F"
        else:
            raw, unit = str(c), "C"
        if random.random() < 0.4:
            raw = raw.replace(".", ",")
        name = garble(st, random.choice([0, 0, 1, 2, 3]), random.choice(["cp1252", "latin-1"]))
        rows.append(f"{rid};{name};{raw};{unit};{date}")
random.shuffle(rows)
pre = ["# IMO Geothermal Monitoring Network -- Legacy Export Format v1"] + \
      [f"# relay note {i}" for i in range(random.randint(3, 9))] + ["", "#", ""]
open("test-input.csv", "w", encoding="utf-8", newline="\r\n").write(
    "\n".join(pre + ["reading_id;station_name;temperature_raw;unit;reading_date"] + rows) + "\n")
open("expected.csv", "w", encoding="utf-8", newline="").write(
    "reading_id,station_name,temperature_c,reading_date\n" + "\n".join(truth) + "\n")
