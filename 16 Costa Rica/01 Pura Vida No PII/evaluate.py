"""Local scorer.

python evaluate.py INPUT GOLD PRED [--items ITEMS] [-v]

Ground-truth PII items come from aligning INPUT with GOLD ([CATEGORY] placeholders),
or from ITEMS ({doc_id: [[cat, value], ...]}) when given.
"""
import argparse
import datetime
import json
import re
import unicodedata
from collections import Counter

PH = re.compile(r'\[([A-Z_]+)\]')
PARTICLES = {'de', 'del', 'la', 'las', 'los', 'y', 'van', 'von', 'da', 'di'}
MONTHS_EN = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september',
             'october', 'november', 'december']
MONTHS_ES = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre',
             'octubre', 'noviembre', 'diciembre']


def fold(s):
    s = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in s if not unicodedata.combining(c)).lower()


def norm(s):
    return re.sub(r'[^a-z0-9]', '', fold(s))


def extract_items(src, gold):
    pieces = PH.split(gold)
    pat, cats = '', []
    for i, p in enumerate(pieces):
        if i % 2 == 0:
            pat += re.escape(p)
        else:
            pat += '(.+?)'
            cats.append(p)
    m = re.fullmatch(pat, src, re.S)
    if not m:
        raise ValueError('gold does not align with input')
    seen = {}
    for c, v in zip(cats, m.groups()):
        seen.setdefault((c, v), None)
    return list(seen)


def parse_date(v):
    s = fold(v)
    m = re.search(r'(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})', s)
    if m:
        return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})', s)
    if m:
        return datetime.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    y = re.search(r'\d{4}', s)
    mon = None
    for names in (MONTHS_EN, MONTHS_ES):
        for i, n in enumerate(names):
            if re.search(r'\b' + n[:3], s):
                mon = i + 1
                break
        if mon:
            break
    if 'setiembre' in s:
        mon = 9
    d = re.search(r'\b(\d{1,2})\b', re.sub(r'\d{4}', '', s))
    if y and mon and d:
        return datetime.date(int(y.group(0)), mon, int(d.group(1)))
    return None


def date_forms(dt):
    d, m, y = dt.day, dt.month, dt.year
    en, es = MONTHS_EN[m - 1], MONTHS_ES[m - 1]
    return {
        f'{en} {d}, {y}', f'{d} {en} {y}', f'{d} de {es} de {y}', f'{es} {d}, {y}',
        f'{d:02d}/{m:02d}/{y}', f'{m:02d}/{d:02d}/{y}', f'{d}/{m}/{y}', f'{y}-{m:02d}-{d:02d}',
    }


def is_city_chunk(c, last):
    return last and not re.search(r'\d', c) and len(c.split()) <= 3


def variants(cat, val):
    v = {val}
    digits = re.sub(r'\D', '', val)
    if cat == 'NAME':
        for p in re.split(r'[\s-]+', val):
            if len(norm(p)) >= 2 and fold(p) not in PARTICLES:
                v.add(p)
    elif cat == 'EMAIL':
        v.add(val.split('@')[0])
    elif cat == 'PHONE':
        v.add(digits)
        if digits.startswith('00506'):
            digits = digits[2:]
        if digits.startswith('506') and len(digits) == 11:
            v.add(digits[3:])
        if len(digits) == 8:
            v.add('506' + digits)
    elif cat == 'ID':
        v.add(digits)
    elif cat == 'ACCOUNT':
        v.add(norm(val))
    elif cat == 'DOB':
        dt = parse_date(val)
        if dt:
            v |= date_forms(dt)
    elif cat == 'ADDRESS':
        chunks = [c.strip() for c in val.split(',') if c.strip()]
        for i, c in enumerate(chunks):
            if not is_city_chunk(c, i == len(chunks) - 1 and i > 0):
                v.add(c)
    return {x for x in v if norm(x)}


def survives(variant, doc_fold, doc_sq, mode):
    n = norm(variant)
    if mode == 'sq':
        return n in doc_sq
    pat = '(?<![a-z0-9])' + '[^a-z0-9]*'.join(re.escape(c) for c in n) + '(?![a-z0-9])'
    return re.search(pat, doc_fold) is not None


TOPICS = [
    ('complaint', r'complain|complaint|queja|reclamo|unacceptable|disappointed|poor service|bad experience|inaceptable|decepcionad'),
    ('refund', r'refund|reimburs|reembolso|devoluci|devuelvan'),
    ('reschedule', r'reschedul|postpone|change the date|new dates|reprogram|cambiar la fecha|posponer'),
    ('payment', r'invoice|payment|charge|billing|wire transfer|factura|pago|cobro|transferencia'),
    ('booking', r'\bbook|reserv|itinerar|excursion|tour|hike'),
]
URG = [
    ('high', r'urgent|immediately|as soon as possible|priority|time-sensitive|urgente|de inmediato|cuanto antes|lo antes posible'),
    ('low', r'no rush|no hurry|whenever|convenience|sin prisa|cuando puedan|no es urgente'),
]


def toy_pipeline(text):
    t = fold(text)
    topic = next((k for k, rx in TOPICS if re.search(rx, t)), 'other')
    urg = next((k for k, rx in URG if re.search(rx, t)), 'normal')
    refs = tuple(sorted(re.findall(r'PV-[A-Z0-9]{3,10}', text)))
    return topic, urg, refs


def canon(t):
    t = PH.sub('<P>', t)
    t = re.sub(r'<P>(?:[ \t-]*<P>)+', '<P>', t)
    return re.sub(r'\s+', ' ', t).strip()


def wordbag(t):
    return Counter(re.findall(r'[^\W_]+', fold(PH.sub(' ', t))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('gold')
    ap.add_argument('pred')
    ap.add_argument('--items')
    ap.add_argument('-v', action='store_true')
    a = ap.parse_args()
    src = {d['id']: d['text'] for d in json.load(open(a.input, encoding='utf-8'))['documents']}
    gold = json.load(open(a.gold, encoding='utf-8'))['redacted_documents']
    pred = json.load(open(a.pred, encoding='utf-8'))['redacted_documents']
    items = json.load(open(a.items, encoding='utf-8')) if a.items else None

    tot = Counter()
    miss = {'tok': Counter(), 'sq': Counter()}
    misses = []
    util_ok = exact = 0
    kept = total_words = 0
    diffs = []
    for did, text in src.items():
        out = pred.get(did, '')
        if not out.strip():
            out = text
        its = [tuple(x) for x in items[did]] if items else extract_items(text, gold[did])
        df, ds = fold(out), norm(out)
        for cat, val in its:
            tot[cat] += 1
            vs = variants(cat, val)
            for mode in ('tok', 'sq'):
                hit = [x for x in vs if survives(x, df, ds, mode)]
                if hit:
                    miss[mode][cat] += 1
                    if mode == 'tok':
                        misses.append((did, cat, val, hit))
        g = gold[did]
        if toy_pipeline(out) == toy_pipeline(g) and toy_pipeline(out)[2] == toy_pipeline(text)[2]:
            util_ok += 1
        if canon(out) == canon(g):
            exact += 1
        else:
            diffs.append((did, canon(g), canon(out)))
        gb, pb = wordbag(g), wordbag(out)
        total_words += sum(gb.values())
        kept += sum(min(c, pb[w]) for w, c in gb.items())

    n = len(src)
    T = sum(tot.values())
    print(f'docs {n}  pii items {T}')
    for mode, label in (('tok', 'privacy (token-bounded)'), ('sq', 'privacy (squashed substring)')):
        print(f'{label}: {1 - sum(miss[mode].values()) / max(T, 1):.4f}')
    print('per category (token-bounded):',
          ', '.join(f'{c} {1 - miss["tok"][c] / tot[c]:.3f} ({tot[c]})' for c in sorted(tot)))
    print('per category (squashed):',
          ', '.join(f'{c} {1 - miss["sq"][c] / tot[c]:.3f}' for c in sorted(tot)))
    print(f'utility proxy (toy topic/urgency/ref): {util_ok / n:.4f}')
    print(f'exact match vs gold (placeholders unified): {exact / n:.4f}')
    print(f'gold non-PII words kept: {kept / max(total_words, 1):.4f}')
    if a.v:
        for did, cat, val, hit in misses[:60]:
            print('MISS', did, cat, repr(val), hit)
        for did, g, o in diffs[:15]:
            print('DIFF', did)
            print('  gold:', g[:600])
            print('  pred:', o[:600])


if __name__ == '__main__':
    main()
