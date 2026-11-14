import csv
import io
from decimal import Decimal, ROUND_HALF_UP

SRC, DST = "test-input.csv", "test-output.csv"


def to_bytes(s):
    # undo a wrong cp1252 (or latin-1) decode, char by char
    out = bytearray()
    for ch in s:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) > 255:
                raise
            out.append(ord(ch))
    return bytes(out)


def fix_name(s):
    s = s.strip()
    for _ in range(5):
        try:
            t = to_bytes(s).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if t == s:
            break
        s = t
    return s


def to_celsius(raw, unit):
    v = Decimal(raw.strip().replace(",", "."))
    if "F" in unit.strip().upper():
        v = (v - 32) * 5 / 9
    return v.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def main():
    data = open(SRC, "rb").read()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252")

    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().lower().startswith("reading_id"))
    rows = csv.DictReader(io.StringIO("\n".join(lines[start:])), delimiter=";")
    rows.fieldnames = [f.strip().lower() for f in rows.fieldnames]

    seen = {}
    for r in rows:
        if not r.get("reading_id") or not r["reading_id"].strip():
            continue
        rid = int(r["reading_id"].strip())
        rec = (fix_name(r["station_name"]), to_celsius(r["temperature_raw"], r["unit"]), r["reading_date"].strip())
        native_c = "F" not in r["unit"].strip().upper()
        # keep one row per reading; if copies disagree, trust the one sent in Celsius
        if rid not in seen or (native_c and not seen[rid][1]):
            seen[rid] = (rec, native_c)

    with open(DST, "w", encoding="utf-8", newline="") as f:
        f.write("reading_id,station_name,temperature_c,reading_date\n")
        for rid in sorted(seen):
            name, temp, date = seen[rid][0]
            f.write(f"{rid},{name},{temp},{date}\n")


if __name__ == "__main__":
    main()
