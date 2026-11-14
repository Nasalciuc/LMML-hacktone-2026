"""Plain English -> Call of the Square register, driven by the rule tables.

to_register(text) returns the register reply (starting with "Sem,") or None when the text
cannot be converted cleanly (unknown capitalised name, too long a sentence, decimals ...).
Every returned reply is verified with the shipped style_checker.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent / "task_algeria"
sys.path.insert(0, str(TASK))
from style_checker import check_reply, load_vocab, MAX_WORDS  # noqa: E402

import lexicon  # noqa: E402

V = load_vocab()
SUFFIX = {k: c["suffix"] for k, c in V["classes"].items()}
SEAL = {k: c["seal"] for k, c in V["classes"].items()}
NEUTRAL = V["neutral_seal"]
DIGITS = V["digits"]
FORBIDDEN = V["forbidden_english"]           # surface form (lower) -> table word
TABLE = V["words"]                           # table word -> {stem, class}
STEMS = set(V["_stem_class"])
SEALS = V["_seals"]


def table_form(word_lower: str) -> str | None:
    base = FORBIDDEN.get(word_lower)
    if base is None:
        return None
    info = TABLE[base]
    return f"{info['stem']}-{SUFFIX[info['class']]}"


def numeral(n: int | str) -> str:
    return "-".join(DIGITS[int(ch)] for ch in str(int(n))) + "-" + SUFFIX["D"]


# ---------------------------------------------------------------- number words
UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
ONE_BLOCK_PREV = {"the", "this", "that", "each", "every", "any", "no", "which", "someone", "anyone", "everyone",
                  "a", "an", "only", "another", "special", "favourite", "favorite", "right", "good", "best", "first",
                  "last", "next", "other", "same", "little", "big", "new", "old", "great", "young"}
ONE_BLOCK_NEXT = {"of", "another", "who", "that", "which", "can", "could", "should", "must", "will", "would", "is",
                  "has", "was", "does", "may", "might", "day", "way", "thing", "another", "or", "and", "to", "at",
                  "in", "on", "by", "for", "with", "from", "it", "you", "s", "", "more", "less"}


def _num_value(words: list[str]) -> int | None:
    total, cur = 0, 0
    seen = False
    for w in words:
        if w in UNITS:
            cur += UNITS[w]
        elif w in TENS:
            cur += TENS[w]
        elif w == "hundred":
            cur = max(cur, 1) * 100
        elif w in ("thousand", "million", "billion"):
            total += max(cur, 1) * SCALES[w]
            cur = 0
        elif w == "and":
            continue
        else:
            return None
        seen = True
    return total + cur if seen else None


NUMWORD = set(UNITS) | set(TENS) | set(SCALES)


# ---------------------------------------------------------------- tokenising
TOKEN_RE = re.compile(r"[A-Za-z0-9À-ɏ][A-Za-z0-9À-ɏ'’\-]*|[,\.\?]")
ASCII_FOLD = str.maketrans("àáâäãåèéêëìíîïòóôöõùúûüçñÀÁÂÄÃÅÈÉÊËÌÍÎÏÒÓÔÖÕÙÚÛÜÇÑ",
                           "aaaaaaeeeeiiiiooooouuuucnAAAAAAEEEEIIIIOOOOOUUUUCN")

LANG_E_NEXT = {"language", "languages", "speakers", "speaker", "word", "words", "alphabet", "script", "literature",
               "grammar", "dialect", "dialects", "poetry", "poem", "poems", "is", "was", "and", "or", "as",
               "because", "but", "too", "also", "the", "with", "in", "at", "for", "to", "while", "which", "name",
               "names", "letters", "letter", "song", "songs", "music", "text", "texts", "version", "translation",
               "teacher", "class", "classes", "lessons", "course", "book", "books", "novel", "novels", "writing",
               "writer", "writers", "name", "term", "spelling", "phrase", "phrases", "expression", "idiom"}
LANG_E_PREV = {"speak", "speaks", "spoke", "spoken", "speaking", "in", "learn", "learning", "learned", "learnt",
               "study", "studying", "studied", "into", "from", "called", "and", "or", "is", "are", "was", "were",
               "write", "writes", "written", "translated", "translate", "teach", "teaches", "taught", "understand",
               "use", "uses", "used", "know", "knows", "language", "languages", "both", "as", "of", "official",
               "native", "mother", "modern", "classical", "standard", "spoke", "fluent", "fluently", "like", "say"}

# capitalised words that are also ordinary words: at the start of a sentence they are left alone
AMBIG_START = set("""
May March Turkey China Orange Yellow Tell Nice Horn Don Ural Superior Grand Great Little Old New North South East West
Northern Southern Eastern Western Central Upper Lower Lake Mount First Second Third Fourth Fifth Sixth Seventh Eighth
Ninth Tenth Hundred Thousand Modern Ancient Classical Romantic Medieval Rock Blues Opera Jazz Ballet Constitution
Requiem David Emma Carmen Inferno Ulysses Basilica Mausoleum Internet Victoria Jordan Chad Georgia Lima Bell Wright
Ali Mars Sun Celtic Roman Greek Muslim Christian Jewish Catholic Hindu Buddhist Orthodox Sufi Sunni Shia Christians
Muslims Jews Hindus Buddhists Catholics Protestants Arctic Tropical Alpine Prophet Pope Saint Sultan Emir Caliph
Pharaoh Bey Dey Pasha Khan Tsar Czar Kaiser Shah Imam Sheikh Mr Mrs Dr Sir Lady Lord Queen Pharaohs Sultans Caliphs
Emirs Algebra Geometry Biology Physics Astronomy Philosophy Geography Economics Medicine Calligraphy Ottoman Byzantine
Apollo Amazon Apple Ford Mercedes Concorde Hubble Champagne Real Pound Euro Dollar Gospel Psalms Odyssey Iliad Messiah
Depression Prohibition Enlightenment Reformation Renaissance Carnival Ottomans Byzantines Moors Arabs Turks Franks
Goths Vandals Huns Saxons Rai Malouf Gnawa Andalusi Chaabi Flamenco Tango Salsa Samba Reggae Hip
""".split())

CONNECTORS = {"of", "the", "da", "de", "del", "van", "von", "al", "el", "es", "la", "le", "and", "y", "di", "du",
              "des", "bin", "ibn", "ben", "a", "with", "in"}

COMMON_UPPER_OK = {"I", "I'm", "I've", "I'd", "I'll"}


class ConvertError(Exception):
    pass


def _cap(t: str) -> bool:
    return t[:1].isupper()


def _split_poss(t: str) -> tuple[str, str]:
    if len(t) > 2 and t[-2:] in ("'s", "’s"):
        return t[:-2], "'s"
    if len(t) > 1 and t[-1:] in ("'", "’") and t[-2:-1] == "s":
        return t[:-1], "'"
    return t, ""


def _mark_name_word(w: str, cls: str) -> str:
    """One word of a name: table words become their stem, others take the class suffix."""
    tf = table_form(w.lower())
    if tf:
        return tf
    if w.lower() in STEMS:
        raise ConvertError(f"name word equals a stem: {w}")
    return f"{w}-{SUFFIX[cls]}"


def _language_class(word: str, prev: str, nxt: str) -> str:
    if nxt in LANG_E_NEXT or nxt in (",", ".", "?", "") or prev in LANG_E_PREV:
        return "E"
    return lexicon.LANG_ADJ_CLASS.get(word, "A")


def preclean(text: str) -> str:
    t = text.translate(ASCII_FOLD)
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"\[[^\]]*\]", "", t)
    t = t.replace("**", "").replace("*", "").replace("#", "").replace("`", "")
    t = re.sub(r"\s*[—–]\s*", ", ", t)
    t = re.sub(r"\s+-\s+", ", ", t)
    t = t.replace("!", ".").replace(";", ",").replace(":", ",").replace('"', "")
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)
    t = re.sub(r"(\d)\s*%", r"\1 percent", t)
    t = re.sub(r"(\d)\s*°\s*C\b", r"\1 degrees Celsius", t)
    t = re.sub(r"(\d)\s*°\s*F\b", r"\1 degrees Fahrenheit", t)
    t = t.replace("°", " degrees")
    t = re.sub(r"(\d)\s*km\b", r"\1 kilometres", t)
    t = re.sub(r"(\d)\s*kg\b", r"\1 kilograms", t)
    t = re.sub(r"(\d)\s*cm\b", r"\1 centimetres", t)
    t = re.sub(r"(\d)\s*m\b", r"\1 metres", t)
    t = t.replace("&", "and").replace("/", " or ")
    t = re.sub(r"\.\.+", ".", t)
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r",\s*,", ",", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def split_plain_sentences(text: str) -> list[str]:
    text = preclean(text)
    if re.search(r"\b(e\.g|i\.e|etc|vs|Mr|Mrs|Dr|St|U\.S|a\.m|p\.m)\.", text):
        raise ConvertError("abbreviation")
    if re.search(r"\d\.\d", text):
        raise ConvertError("decimal")
    parts = re.split(r"(?<=[.?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def convert_sentence(sent: str, first: bool) -> str:
    """Convert one plain sentence (ending with . or ?) to the register (with seal)."""
    end = "?" if sent.rstrip().endswith("?") else "."
    toks = TOKEN_RE.findall(sent)
    toks = [t for t in toks if t not in (".", "?")]
    while toks and toks[-1] == ",":
        toks.pop()
    if not toks:
        raise ConvertError("empty")
    if len([t for t in toks if t != ","]) < 2:
        raise ConvertError("fragment")
    out: list[str] = []   # output tokens (words or ',')
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        if t == ",":
            if out and out[-1] != ",":
                out.append(",")
            i += 1
            continue
        word_pos = sum(1 for o in out if o != ",")
        at_start = word_pos == 0
        prev = toks[i - 1].lower() if i > 0 else ""
        nxt = toks[i + 1].lower() if i + 1 < n else ""
        # ---- digits
        core, poss = _split_poss(t)
        if any(ch.isdigit() for ch in core):
            m = re.fullmatch(r"(\d+)(st|nd|rd|th|s)?", core)
            if not m:
                raise ConvertError(f"odd digit token {t}")
            out.append(numeral(m.group(1)) + poss)
            i += 1
            continue
        # ---- number words
        low = core.lower()
        if low in NUMWORD or ("-" in low and all(p in NUMWORD for p in low.split("-"))):
            j = i
            seq = []
            while j < n:
                lw = toks[j].lower()
                if lw in NUMWORD:
                    seq.append(lw)
                elif "-" in lw and all(p in NUMWORD for p in lw.split("-")):
                    seq.extend(lw.split("-"))
                elif lw == "and" and seq and seq[-1] in SCALES and j + 1 < n and toks[j + 1].lower() in NUMWORD:
                    seq.append("and")
                else:
                    break
                j += 1
            nxt2 = toks[j].lower() if j < n else ""
            if seq == ["one"] and (prev in ONE_BLOCK_PREV or nxt2 in ONE_BLOCK_NEXT):
                out.append(core if not at_start else core)
                i += 1
                continue
            if seq in (["hundred"], ["thousand"], ["million"], ["billion"]) and prev not in ("a", "one"):
                # "hundreds of", "thousands of": keep as a plain word
                out.append(core)
                i += 1
                continue
            val = _num_value(seq)
            if val is None:
                raise ConvertError("number parse")
            if prev == "a" and out and out[-1].lower() == "a" and seq[0] in SCALES:
                out.pop()
            out.append(numeral(val))
            i = j
            continue
        if low in ("hundreds", "thousands", "millions", "billions", "dozens", "dozen", "tens"):
            out.append(core + poss)
            i += 1
            continue
        # ---- multiword names (longest match)
        if _cap(core):
            matched = False
            for L in range(min(lexicon.MAX_MULTI, n - i), 1, -1):
                span = tuple(_split_poss(x)[0] for x in toks[i:i + L])
                if span in lexicon.MULTI:
                    cls = lexicon.MULTI[span]
                    for k, w in enumerate(toks[i:i + L]):
                        wc, wp = _split_poss(w)
                        if _cap(wc):
                            out.append(_mark_name_word(wc, cls) + wp)
                        else:
                            if wc.lower() in STEMS:
                                raise ConvertError("stem in name")
                            out.append(table_form(wc.lower()) or wc)
                    i += L
                    matched = True
                    break
            if matched:
                continue
        # ---- table words (any case)
        tf = table_form(low)
        if tf:
            out.append(tf + poss)
            i += 1
            continue
        if low in STEMS:
            raise ConvertError(f"bare stem word {t}")
        # ---- capitalised words
        if _cap(core):
            if core in COMMON_UPPER_OK:
                out.append(core + poss)
                i += 1
                continue
            if at_start and (core in AMBIG_START or core not in lexicon.SINGLE) and core not in lexicon.LANGUAGE_WORDS:
                out.append((core.lower() if first and core.lower() != "i" else core) + poss)
                i += 1
                continue
            if core in lexicon.LANGUAGE_WORDS:
                cls = _language_class(core, prev, nxt)
                out.append(_mark_name_word(core, cls) + poss)
                i += 1
                continue
            if core in lexicon.SINGLE:
                # a run of unknown-but-capitalised words inherits the class of its known neighbour
                out.append(_mark_name_word(core, lexicon.SINGLE[core]) + poss)
                i += 1
                continue
            if "-" in core:
                head = core.split("-")[0]
                if head in lexicon.SINGLE:
                    out.append(f"{core}-{SUFFIX[lexicon.SINGLE[head]]}" + poss)
                    i += 1
                    continue
            raise ConvertError(f"unknown name {core}")
        # ---- plain lowercase word
        if low in SEALS and i == n - 1:
            raise ConvertError("ends with a seal word")
        if "-" in core and core.split("-")[0].lower() in DIGITS:
            raise ConvertError("digit-like hyphen word")
        if "-" in core and core.rsplit("-", 1)[1].lower() in SUFFIX.values():
            raise ConvertError("suffix-like hyphen word")
        out.append(core + poss)
        i += 1
    # ---- seal
    words = [o for o in out if o != ","]
    last = None
    for w in words:
        c = _class_of(w)
        if c:
            last = c
    seal = SEAL[last] if last else NEUTRAL
    if len(words) + 1 > MAX_WORDS:
        raise ConvertError("too long")
    s = ""
    for o in out:
        if o == ",":
            s += ","
        else:
            s += (" " if s else "") + o
    s = s.rstrip(",")
    return f"{s} {seal}{end}"


def _class_of(w: str) -> str | None:
    w = _split_poss(w.strip(".,?"))[0]
    if "-" not in w:
        return None
    parts = w.lower().split("-")
    if parts[0] in DIGITS and all(p in DIGITS for p in parts[:-1]) and parts[-1] == SUFFIX["D"]:
        return "D"
    tail = parts[-1]
    for k, suf in SUFFIX.items():
        if tail == suf:
            return k
    return None


SUBJECTS = {"it", "they", "this", "these", "you", "we", "he", "she", "people", "there", "many", "most", "some", "i"}
CUT_WORDS = {"often", "such", "including", "which", "making", "allowing", "typically", "usually", "especially",
             "like", "and", "but", "while", "as", "where", "who", "or", "so", "because", "with", "from", "for",
             "although", "though", "when", "since", "giving", "helping", "providing", "creating", "leading"}
SPLIT_RE = re.compile(r",\s+(and|but|so)\s+")
TRIM_RE = re.compile(r",\s+")


def _split_long(sent: str, depth: int = 0) -> list[str]:
    """Shorten an over-long sentence: split at ', and/but/so <subject>' or drop a trailing comma clause."""
    limit = MAX_WORDS - 2
    if len(sent.split()) <= limit or depth > 2:
        return [sent]
    end = "?" if sent.endswith("?") else "."
    body = sent.rstrip(" .?")
    best = None
    for m in SPLIT_RE.finditer(body):
        left, right = body[: m.start()], body[m.end():]
        if len(left.split()) >= 4 and len(right.split()) >= 3 and right.split()[0].lower() in SUBJECTS:
            score = abs(len(left.split()) - len(right.split()))
            if best is None or score < best[0]:
                best = (score, left, m.group(1), right)
    if best:
        _, left, conj, right = best
        right = ("But " + right) if conj == "but" else ("So " + right) if conj == "so" else right[0].upper() + right[1:]
        return _split_long(left.rstrip(" ,") + ".", depth + 1) + _split_long(right.rstrip(" ,") + end, depth + 1)
    # drop the trailing clause after the last comma that leaves a sentence of a good length
    cuts = [m.start() for m in TRIM_RE.finditer(body)
            if body[m.end():].split(" ", 1)[0].lower() in CUT_WORDS]
    for c in reversed(cuts):
        left = body[:c]
        if 5 <= len(left.split()) <= limit:
            return [left.rstrip(" ,") + end]
    return [sent]


def to_register(text: str, max_sentences: int = 4, strict: bool = True, split_long: bool = True) -> str | None:
    try:
        sents = split_plain_sentences(text)
    except ConvertError:
        return None
    outs = []
    for k, s in enumerate(sents):
        if not s.endswith((".", "?")):
            s = s + "."
        pieces = [s]
        if split_long and len(s.split()) > MAX_WORDS - 2:
            pieces = _split_long(s)
        ok = True
        for p in pieces:
            try:
                outs.append(convert_sentence(p, first=(len(outs) == 0)))
            except ConvertError:
                ok = False
                break
        if not ok:
            if strict or not outs:
                return None
            break
        if len(outs) >= max_sentences:
            break
    if not outs:
        return None
    reply = "Sem, " + " ".join(outs)
    res = check_reply(reply, V)
    if res["conformance"] < 1.0:
        return None
    return reply


if __name__ == "__main__":
    tests = [
        "The capital of Algeria is Algiers. The city lies on the Mediterranean coast.",
        "Algiers is the capital of Algeria.",
        "There are seven days in a week.",
        "The Algerian War of Independence began in 1954.",
        "The largest planet in our solar system is Jupiter.",
        "The Mona Lisa is a painting by Leonardo da Vinci.",
        "In Austria, people speak German.",
        "The Sahara Desert covers most of Algeria!",
        "A standard guitar has 6 strings.",
        "The boiling point of water is 100 degrees Celsius.",
        "Learning a new language takes time, and you should practice every day with native speakers to improve.",
        "The French Revolution began in 1789.",
        "Algeria's main export is oil and natural gas.",
        "Yves Saint Laurent was born in Oran in 1936.",
        "15 plus 27 equals 42.",
        "One tip is to practice speaking every day.",
    ]
    for t in tests:
        print(to_register(t), "|", t)
