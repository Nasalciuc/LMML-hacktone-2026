"""Two Solitudes: train W, b for the vault from the synthetic corpus.

    python train.py            -> model.npz

Steps
  1. split every synthetic report into header and sentences, guess the language of each
  2. find the hazard sentences: a sentence classifier trained on first sentences, cross-fitted by event
  3. rebuild training reports from clean parts: hazard sentence(s) + boilerplate drawn at random,
     equal numbers per hazard and language, with word swaps across English and French
  4. zero the fingerprint cells that only carry function words or hash collisions
  5. average eight logistic regressions trained on differently augmented sets, add a season term

Uses the sealed fingerprints only as unlabeled marginals in step 4 (how often each cell is non-zero).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
TASK = HERE.parent / "task_canada"
sys.path.insert(0, str(TASK))
from featurize import bucket, featurize, tokens  # noqa: E402
from open_vault import CLASSES, load_sealed  # noqa: E402
from lexicon import EXTRA, GENERAL, PAIRS, SYN  # noqa: E402

# ----------------------------------------------------------------------------- parsing
MERGED = re.compile(r"^(?:On|Le|Du|From|In|Between)\b[^.]*?\d{4}, (?:(?:in|near|près de|dans|à)\s+)?"
                    r"(?P<place>[^,:]*?(?:\([^)]*\))?)[,:]\s+(?P<rest>.*)$", re.S)
FR = re.compile(r"\b(le|la|les|des|du|est|sont|une|dans|avec|pour|sur)\b", re.I)
EN = re.compile(r"\b(the|was|were|and|with|of|in|from|at)\b", re.I)


def lang_of(t):
    f, e = len(FR.findall(t)), len(EN.findall(t))
    if f == e:
        return "fr" if re.search(r"[éèêàâçùûôî]", t) else "en"
    return "fr" if f > e else "en"


def split_doc(t):
    t = re.sub(r"\(([^)]*)\)", lambda m: "(" + m.group(1).replace(".", "") + ")", t)
    t = re.sub(r"\b(St|Ste)\. ", r"\1 ", t)
    parts = [p.strip() for p in re.split(r"(?<=\.)\s+(?=[A-ZÀ-Ý0-9])", t) if p.strip()]
    head = parts[0]
    if re.search(r"\d{4}\.?$", head) or (re.match(r"^[^,]+,\s+(?:du |le )?[\w\s\-éû]*\d", head)
                                         and head.count(",") <= 2 and len(head.split()) < 10):
        return head, parts[1:]
    m = MERGED.match(head)
    if m:
        rest = m.group("rest")
        return head[:m.start("rest")].strip().rstrip(",:") + ".", [rest[0].upper() + rest[1:]] + parts[1:]
    return "", parts


def load_corpus(path=TASK / "corpus.csv"):
    d = pd.read_csv(path)
    d["y"] = d.label.map(CLASSES.index)
    sp = [split_doc(t) for t in d.text]
    d["head"] = [h for h, _ in sp]
    d["body"] = [b for _, b in sp]
    d["lang"] = d.text.map(lang_of)
    d["event"] = d["head"].map(lambda h: re.sub(r"[\d\s,.]+", " ", h).strip().lower())
    return d[d.body.map(len) > 0].copy()


def sentences(d):
    rows = [(i, j, s) for i, b in zip(d.index, d.body) for j, s in enumerate(b)]
    S = pd.DataFrame(rows, columns=["doc", "pos", "text"])
    S["y"] = d.y.loc[S.doc].values
    S["lang"] = S.text.map(lang_of)
    S["k"] = S.text.map(lambda s: re.sub(r"\d+(?:[ ,.]\d+)*", "0", s))
    S["event"] = d.event.loc[S.doc].values
    return S


# ----------------------------------------------------------------------------- hazard sentences
def balance_weights(y, lang):
    n = pd.DataFrame({"y": y, "l": lang}).groupby(["y", "l"]).y.transform("size").values
    w = 1.0 / n
    return w * len(w) / w.sum()


def fit_lr(X, y, w=None, C=3.0):
    return LogisticRegression(C=C, max_iter=5000).fit(X, y, sample_weight=w)


def template_roles(S):
    """Sentence template -> 'gen' (boilerplate seen across many hazards), 'haz' (opens reports of one hazard), 'unk'."""
    first = S[S.pos == 0]
    c1 = pd.crosstab(first.k, first.y).reindex(columns=range(9), fill_value=0)
    ca = pd.crosstab(S.k, S.y).reindex(columns=range(9), fill_value=0)
    n = ca.sum(1)
    n1 = c1.sum(1).reindex(ca.index).fillna(0)
    sh1 = (c1.max(1) / c1.sum(1)).reindex(ca.index).fillna(0)
    p = ca.values / n.values[:, None]
    H = -np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0).sum(1) / np.log(9)
    sh = ca.max(1) / n
    nlab = (ca > 0).sum(1)
    gen = ((n >= 20) & (H >= 0.45)) | ((n1 >= 3) & (sh1 < 0.6)) | ((n1 < 3) & (nlab >= 4) & (sh < 0.6))
    haz = ~gen & (n1 >= 3) & (sh1 >= 0.8)
    return pd.Series(np.where(haz, "haz", np.where(gen, "gen", "unk")), index=ca.index)


def sentence_classes(S, XS, tau=0.5, rounds=2):
    """Hazard 0..8 per sentence, -1 boilerplate, -2 unsure. Ten-class model, out-of-fold by event."""
    role = S.k.map(template_roles(S)).values
    y, lang, ev = S.y.values, S.lang.values, S.event.values
    s1 = (S.pos.values == 0) & (role != "gen")
    target = np.where(role == "gen", 9, y)
    lab = s1 | (role == "gen")
    for _ in range(rounds):
        P = np.zeros((len(S), 10))
        for tr, te in GroupKFold(5).split(XS, target, ev):
            m = lab.copy()
            m[te] = False
            idx = np.flatnonzero(m)
            mdl = fit_lr(XS[idx], target[idx], balance_weights(target[idx], lang[idx]), C=10.0)
            P[np.ix_(te, mdl.classes_)] = mdl.predict_proba(XS[te])
        pmax, pred = P.max(1), P.argmax(1)
        cls = np.full(len(S), -2)
        ok = pmax >= tau
        cls[ok] = np.where(pred[ok] == 9, -1, pred[ok])
        agree = s1 & (pred == y)
        cls[agree] = y[agree]
        target = np.where(cls == -1, 9, cls)
        lab = cls >= -1
    return cls


SYN_RE = {(c, l): re.compile(r"\b(" + "|".join(sorted(map(re.escape, v), key=len, reverse=True)) + r")\b", re.I)
          for c, g in SYN.items() for l, v in g.items()}
SHARED = {"fire": ("wildfire", "industrial"), "fires": ("wildfire", "industrial"), "incendie": ("wildfire", "industrial"),
          "incendies": ("wildfire", "industrial"), "feu": ("wildfire", "industrial"), "storm": ("storm", "winter"),
          "tempête": ("storm", "winter"), "snow": ("winter", "ground"), "neige": ("winter", "ground"),
          "avalanche": ("winter", "ground"), "explosion": ("industrial", "transport"),
          "collision": ("transport", "industrial"), "train": ("transport", "industrial")}


def lexicon_classes(text):
    out = set()
    for c in range(9):
        for l in ("en", "fr"):
            for m in SYN_RE[(CLASSES[c], l)].finditer(text):
                w = m.group(0).lower()
                out |= {CLASSES.index(x) for x in SHARED[w]} if w in SHARED else {c}
    return out


def lexicon_check(S, cls):
    """Drop hazard sentences whose hazard words point elsewhere, later sentences with no hazard word,
    and boilerplate that names a hazard."""
    kw = S.text.map(lexicon_classes).values
    later = S.pos.values > 0
    bad = np.array([bool((c >= 0 and k and c not in k) or (c == -1 and k) or (c >= 0 and lt and not k))
                    for c, k, lt in zip(cls, kw, later)])
    out = cls.copy()
    out[bad] = -2
    return out


# ----------------------------------------------------------------------------- rebuilding reports
NAME_RE = re.compile(r"(?<=[\w,;:] )(?!(?:River|Nation|Prairies|F\d)\b)([A-ZÀ-Ý][\w'’\-]+)")


def build_translators():
    per_class = {}
    for c, pairs in PAIRS.items():
        for src, tgt, side in [(0, 1, "en"), (1, 0, "fr")]:
            m = {}
            for p in pairs:
                m.setdefault(p[src].lower(), []).append(p[tgt])
            rx = re.compile(r"\b(" + "|".join(sorted(map(re.escape, m), key=len, reverse=True)) + r")\b", re.I)
            per_class[(CLASSES.index(c), side)] = (rx, m)
    general = {}
    allpairs = list(GENERAL) + [p for ps in PAIRS.values() for p in ps]
    for src, tgt, side in [(0, 1, "en"), (1, 0, "fr")]:
        m = {}
        for p in allpairs:
            m.setdefault(p[src].lower(), set()).add(p[tgt])
        m = {k: sorted(v) for k, v in m.items()}
        rx = re.compile(r"\b(" + "|".join(sorted(map(re.escape, m), key=len, reverse=True)) + r")\b", re.I)
        general[side] = (rx, m)
    return per_class, general


TR_CLASS, TR_GENERAL = build_translators()


def synonym_swap(text, c, rng):
    name, l = CLASSES[c], lang_of(text)
    hits = list(SYN_RE[(name, l)].finditer(text))
    if not hits:
        return text
    h = hits[rng.integers(len(hits))]
    return text[:h.start()] + SYN[name][l][rng.integers(len(SYN[name][l]))] + text[h.end():]


def translate_terms(text, c, rng):
    rx, m = TR_CLASS[(c, lang_of(text))]
    return rx.sub(lambda h: m[h.group(0).lower()][rng.integers(len(m[h.group(0).lower()]))], text)


def translate_general(text, rng, p_word=0.5):
    rx, m = TR_GENERAL[lang_of(text)]

    def rep(h):
        if rng.random() >= p_word:
            return h.group(0)
        opts = m[h.group(0).lower()]
        return opts[rng.integers(len(opts))]
    return rx.sub(rep, text)


def word_dropout(text, rng, p):
    keep = [x for x in text.split(" ") if rng.random() >= p]
    return " ".join(keep) if keep else text


def pools(d, S, cls):
    nat = {(c, l): S.text.values[(cls == c) & (S.lang.values == l)] for c in range(9) for l in ("en", "fr")}
    gen = {l: S.text.values[(cls == -1) & (S.lang.values == l)] for l in ("en", "fr")}
    heads = {l: d["head"].values[(d.lang.values == l) & (d["head"].values != "")] for l in ("en", "fr")}
    names = []
    for v in nat.values():
        for s in v:
            names += NAME_RE.findall(s)
    return nat, gen, heads, np.array(sorted(set(names)))


EXTRA_POOL = {(CLASSES.index(c), l): np.array(v) for c, g in EXTRA.items() for l, v in g.items()}


def generate(P, n_per, rng, p_syn=0.3, p_tr=0.35, p_gtr=0.5, p_drop=0.3, p_extra=0.15, p_names=0.5,
             p_mix=0.15, p_second=0.4, p_distract=0.3, max_gen=4, p_head=0.85):
    """Reports with a known hazard: its sentence(s), random boilerplate, a random header and,
    sometimes, a sentence about another hazard. Equal counts per hazard and language."""
    nat, gen, heads, names = P
    other = {"en": "fr", "fr": "en"}
    pick = lambda arr: arr[rng.integers(len(arr))]
    texts, ys = [], []
    for c in range(9):
        for L in ("en", "fr"):
            for _ in range(n_per):
                lh = other[L] if rng.random() < p_mix else L
                if len(nat[(c, lh)]) == 0:
                    lh = L
                main = [pick(EXTRA_POOL[(c, lh)])] if rng.random() < p_extra else [pick(nat[(c, lh)])]
                if rng.random() < p_second:
                    main.append(pick(nat[(c, L)]))
                main = [synonym_swap(s, c, rng) if rng.random() < p_syn else s for s in main]
                body = [translate_terms(s, c, rng) if rng.random() < p_tr else s for s in main]
                for _ in range(rng.integers(0, max_gen + 1)):
                    body.append(pick(gen[other[L] if rng.random() < p_mix else L]))
                if rng.random() < p_distract:
                    body.append(pick(nat[((c + 1 + rng.integers(8)) % 9, L)]))
                body = [translate_general(s, rng) if rng.random() < p_gtr else s for s in body]
                body = [NAME_RE.sub(lambda m: pick(names), s) if rng.random() < p_names else s for s in body]
                rng.shuffle(body)
                body = [word_dropout(s, rng, p_drop) for s in body]
                h = [pick(heads[L])] if rng.random() < p_head else []
                texts.append(" ".join(h + body))
                ys.append(c)
    return texts, np.array(ys)


# ----------------------------------------------------------------------------- cells to ignore
STOP = set("""the a an of in on at to for from by with and or but was were is are be been being had has have it its as
that this these those which who after before during near over into up down out than more most about some nearly
approximately almost several many other others all also there their they he she his her not no only when where while
le la les de des du un une et ou en à au aux dans sur par pour avec sont est a ont été être qui que se son sa ses leur
leurs plus près environ quelque quelques plusieurs ce cette ces lors après avant pendant entre vers chez ne pas il elle
ils elles on y dont 0""".split())


def function_cells(texts, min_df=0.002):
    pairs = {}
    for t in texts:
        tk = tokens(t)
        for p in {a + " " + b for a, b in zip(tk, tk[1:]) if a in STOP and b in STOP}:
            pairs[p] = pairs.get(p, 0) + 1
    m = np.zeros(4096, bool)
    m[[bucket(w) for w in STOP]] = True
    m[[bucket(p) for p, n in pairs.items() if n >= min_df * len(texts)]] = True
    return m


def lexicon_cells():
    terms = set()
    for g in SYN.values():
        for v in g.values():
            terms |= set(v)
    for ps in PAIRS.values():
        for a, b in ps:
            terms |= {a, b}
    return {bucket(t.lower()) for t in terms}


def language_of_rows(X):
    fr = ((X[:, bucket("de")] > 0) & (X[:, bucket("la")] > 0)) | ((X[:, bucket("les")] > 0) & (X[:, bucket("des")] > 0))
    en = (X[:, bucket("the")] > 0) & ~fr
    return en, fr


def collision_cells(Xt, lang_t, Xs):
    """Cells far more active in the real reports than in training: in general (common real words), or in the
    language that never uses the training words of that cell (a hash collision with that language)."""
    rs, rt = (Xs > 0).mean(0), (Xt > 0).mean(0)
    shift = (rs > 0.05) & (rs > 5 * rt) & ~np.isin(np.arange(4096), list(lexicon_cells()))
    en, fr = language_of_rows(Xs)
    rt_en, rt_fr = (Xt[lang_t == "en"] > 0).mean(0), (Xt[lang_t == "fr"] > 0).mean(0)
    rs_en, rs_fr = (Xs[en] > 0).mean(0), (Xs[fr] > 0).mean(0)
    frac_fr = rt_fr / np.maximum(rt_en + rt_fr, 1e-9)
    cross = ((frac_fr > 0.85) & (rs_en > 3 * rt_en + 0.02)) | ((frac_fr < 0.15) & (rs_fr > 3 * rt_fr + 0.02))
    return shift | cross


# ----------------------------------------------------------------------------- season
MONTHS_EN = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
             "november", "december"]
MONTHS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
             "novembre", "décembre"]


def season_term(corpus, lam=1.5):
    """Month cells get lam * log P(month | hazard), centred, from the corpus dates. Weather hazards only:
    for industrial, transport and epidemic the corpus months look like artefacts and are left flat."""
    rx = re.compile(r"\b(" + "|".join(MONTHS_EN + MONTHS_FR) + r")\b", re.I)

    def month(t):
        m = rx.search(t)
        if not m:
            return -1
        w = m.group(1).lower()
        return MONTHS_EN.index(w) if w in MONTHS_EN else MONTHS_FR.index(w)
    mo = corpus.text.map(month)
    ct = pd.crosstab(corpus.label, mo).reindex(index=CLASSES, columns=range(12), fill_value=0).values.astype(float)
    P = 0.7 * ct / ct.sum(1, keepdims=True) + 0.3 / 12
    for c in ("industrial", "transport", "epidemic"):
        P[CLASSES.index(c)] = 1 / 12
    L = np.log(P)
    L -= L.mean(0, keepdims=True)
    W = np.zeros((9, 4096))
    for i in range(12):
        for w in (MONTHS_EN[i], MONTHS_FR[i]):
            W[:, bucket(w)] += lam * L[:, i]
    return W


# ----------------------------------------------------------------------------- the whole run
CONFIGS = [dict(p_syn=0.3, p_tr=0.35, p_gtr=0.5, p_drop=0.3, C=3), dict(p_syn=0.5, p_tr=0.5, p_gtr=0.5, p_drop=0.2, C=3),
           dict(p_syn=0.3, p_tr=0.35, p_gtr=1.0, p_drop=0.3, C=3), dict(p_syn=0.2, p_tr=0.2, p_gtr=0.3, p_drop=0.1, C=3),
           dict(p_syn=0.4, p_tr=0.4, p_gtr=0.5, p_drop=0.4, C=10), dict(p_syn=0.3, p_tr=0.35, p_gtr=0.5, p_drop=0.3, C=1),
           dict(p_syn=0.3, p_tr=0.6, p_gtr=0.7, p_drop=0.3, C=3), dict(p_syn=0.3, p_tr=0.35, p_gtr=0.5, p_drop=0.3, C=3,
                                                                     p_extra=0.3)]


def prepare(verbose=True):
    d = load_corpus()
    S = sentences(d)
    cls = lexicon_check(S, sentence_classes(S, featurize(S.text.tolist())))
    if verbose:
        print("sentences", len(S), "hazard", int((cls >= 0).sum()), "boilerplate", int((cls == -1).sum()),
              "dropped", int((cls == -2).sum()))
    return d, S, cls, pools(d, S, cls)


def train(d, P, Xs, n_per=800, verbose=True):
    ref_txt, _ = generate(P, n_per, np.random.default_rng(1), **{k: v for k, v in CONFIGS[0].items() if k != "C"})
    Xref = featurize(ref_txt)
    lang_ref = np.tile(np.repeat(["en", "fr"], n_per), 9)
    mask = function_cells(d.text.tolist()) | collision_cells(Xref, lang_ref, Xs)
    Wsum, members = 0, []
    for i, cfg in enumerate(CONFIGS):
        cfg = dict(cfg)
        C = cfg.pop("C")
        tx, ty = generate(P, n_per, np.random.default_rng(100 + i), **cfg)
        X = featurize(tx)
        X[:, mask] = 0
        m = fit_lr(X, ty, C=C)
        W = np.zeros((9, 4096))
        W[m.classes_] = m.coef_
        W[:, mask] = 0
        W /= (X @ W.T).std()
        Wsum = Wsum + W
        members.append(W)
        if verbose:
            print(f"member {i}: train acc {(m.predict(X) == ty).mean():.3f}", flush=True)
    W = Wsum / len(CONFIGS) + season_term(pd.read_csv(TASK / "corpus.csv"))
    return W, np.zeros(9), mask, members


def main():
    d, S, cls, P = prepare()
    Xs = load_sealed()
    W, b, mask, _ = train(d, P, Xs)
    np.savez(HERE / "model.npz", W=W, b=b)
    p = (Xs @ W.T + b).argmax(1)
    print("masked cells", int(mask.sum()))
    print("sealed reports per predicted hazard:", dict(zip(CLASSES, np.bincount(p, minlength=9).tolist())))
    print("saved", HERE / "model.npz")


if __name__ == "__main__":
    main()
