"""Content checks on what the VLM read from a stamp.

The read is a JSON-ish string with country / denominations / years / other_text.
Rules (scores add up):
  odd_value   face value that is not a plausible tenge rate (not a multiple of 5, or a
              "copied digit" pattern such as 55, 66, 133, 202, 330, 2211)
  bad_year    a printed year after 2026 (older years are biographies/anniversaries, so not flagged)
  no_year     no year printed at all (most modern issues print it; erased years are the most
              common removal in the exhibits)
  no_country  neither QAZAQSTAN nor KAZAKHSTAN read
"""
import json
import re

ROUND = {1, 2, 3, 5}  # not used directly; kept for reference


def parse_read(txt):
    s = txt.strip()
    s = re.sub(r"^```(json)?", "", s).strip().rstrip("`").strip()
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    d = {}
    for k in ["country", "denominations", "years", "other_text"]:
        m = re.search(r'"%s"\s*:\s*\[(.*?)\]' % k, s, re.S)
        d[k] = re.findall(r'"([^"]*)"|(\d+)', m.group(1)) if m else []
        d[k] = [a or b for a, b in d[k]]
    return d


def numbers(vals):
    out = []
    for v in vals or []:
        v = str(v)
        for m in re.finditer(r"\d+(?:[.,]\d+)?", v):
            out.append(m.group(0))
    return out


def value_flags(v):
    """v: digit string as printed. Returns a suspicion score and a reason."""
    digits = re.sub(r"\D", "", v.split(".")[0].split(",")[0])
    if not digits or digits.startswith("0"):  # leading zero = misread (e.g. "00900")
        return 0.0, ""
    n = int(digits)
    if 1992 <= n <= 2030 and len(digits) == 4:
        return 0.0, ""  # a year in the denomination list
    s = 0.0
    why = []
    if n % 5 != 0 and n > 10:
        s += 1.0
        why.append("not_mult5")
    # a digit repeated where a real value would have a round tail: 55, 66, 99, 133, 155, 202, 330, 110
    core = digits.rstrip("0") or "0"
    if len(digits) >= 2 and len(set(core)) < len(core):
        s += 1.0
        why.append("repeat_digit")
    elif len(digits) >= 3 and digits[0] == digits[-1] and digits[-1] != "0":
        s += 1.0
        why.append("copied_digit")
    return s, "+".join(why)


def score(read_txt):
    d = parse_read(read_txt)
    flags = {}
    best = 0.0
    for v in numbers(d.get("denominations")):
        s, why = value_flags(v)
        if s > best:
            best, flags["value"] = s, f"{v}:{why}"
    years = [int(y) for y in numbers(d.get("years")) if len(re.sub(r"\D", "", y)) == 4]
    all_years = years + [int(y) for y in numbers(d.get("other_text")) if len(y) == 4 and y.isdigit()]
    # years before 1992 are genuine (biographies, anniversaries); a forged year digit gives a
    # year after the issue period, e.g. 2211 from 2011
    bad = [y for y in all_years if y > 2026]  # 2026 occurs on genuine stamps (ranges, centenaries)
    if bad:
        flags["bad_year"] = bad
    if not all_years:
        flags["no_year"] = True
    ctry = " ".join(map(str, d.get("country") or [])).upper()
    if not re.search(r"KAZ|ҚАЗ|КАЗ|QAZ", ctry):
        flags["no_country"] = True
    return d, flags, best
