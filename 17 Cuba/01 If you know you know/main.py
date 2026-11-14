"""Zemí: hybrid passage retriever (BM25 + static embeddings) over the Cuba library."""
from __future__ import annotations

import json
import os
import re
import unicodedata

import numpy as np
import Stemmer

STOP = set("""a an the of and or to in on at by for from with as is are was were be been being this that these those
it its into than then there their they them he she his her him we you your our not no nor but if so such which who whom
whose what when where why how all any each more most other some very can could would should may might will shall do does
did done have has had having about over under between through during before after up down out off again once here only
own same too just also i me my us""".split())

QUERY_STOP = set("""part bit chapter section article book reading passage tell tells told story whole full history
describe describes described description read us please want know remember thing things la el los las donde parte que
de del y en una un""".split())

SPANISH = {
    "ano": "year", "anos": "years", "gente": "people", "mujeres": "women", "mujer": "woman", "hombres": "men",
    "hombre": "man", "iglesia": "church", "ciudad": "city", "pueblo": "town", "guerra": "war", "rey": "king",
    "nino": "child", "ninos": "children", "agua": "water", "tierra": "land", "azucar": "sugar", "tabaco": "tobacco",
    "esclavos": "slaves", "esclavo": "slave", "barco": "ship", "barcos": "ships", "dinero": "money",
    "musica": "music", "libro": "book", "muerte": "death", "muertos": "dead", "soldados": "soldiers",
    "ejercito": "army", "poblacion": "population", "comida": "food", "fiesta": "festival", "baile": "dance",
    "cancion": "song", "montanas": "mountains", "montana": "mountain", "playas": "beaches", "gobierno": "government",
    "ley": "law", "leyes": "laws", "viejo": "old", "primero": "first", "primera": "first",
    "revolucion": "revolution", "esclavitud": "slavery", "caballo": "horse", "caballos": "horses", "noche": "night",
    "dia": "day", "historia": "history", "sacerdote": "priest", "medico": "doctor", "enfermos": "sick",
    "hijos": "sons", "hijo": "son", "padre": "father", "madre": "mother", "diosa": "goddess", "dios": "god",
    "huelga": "strike", "lluvia": "rain", "pescado": "fish", "carne": "meat", "arroz": "rice", "vino": "wine",
}

STEM = Stemmer.Stemmer("english")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text)


def analyse(text: str) -> list[str]:
    return STEM.stemWords([w for w in normalise(text).split() if w not in STOP])


def osa(a: str, b: str, cap: int) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            c = prev[j - 1] + (a[i - 1] != b[j - 1])
            c = min(c, prev[j] + 1, cur[j - 1] + 1)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                c = min(c, prev2[j - 2] + 1)
            cur[j] = c
        if min(cur) > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[-1]


class Speller:
    """Fixes words the corpus has never seen: typos by edit distance, glued words by splitting."""

    def __init__(self, path: str, known: set[str]) -> None:
        self.freq: dict[str, int] = {}
        self.words: list[str] = []
        self.grams: dict[str, list[int]] = {}
        self.known = known
        with open(path, encoding="utf-8") as f:
            for line in f:
                w, n = line.split("\t")
                self.freq[w] = int(n)
                i = len(self.words)
                self.words.append(w)
                for g in self._grams(w):
                    self.grams.setdefault(g, []).append(i)

    @staticmethod
    def _grams(w: str) -> set[str]:
        w = f"^{w}$"
        return {w[i:i + 2] for i in range(len(w) - 1)}

    def _is_word(self, w: str) -> bool:
        forms = [w] + ([w[:-1], w[:-2]] if w.endswith("s") else [])
        return any(f in self.freq or f in self.known for f in forms)

    def _nearest(self, w: str, cap: int) -> str | None:
        counts: dict[int, int] = {}
        for g in self._grams(w):
            for i in self.grams.get(g, ()):
                counts[i] = counts.get(i, 0) + 1
        best, best_key = None, None
        for i in sorted(counts, key=counts.get, reverse=True)[:150]:
            c = self.words[i]
            if self.freq[c] < 3:
                continue
            dist = osa(w, c, cap)
            if dist <= cap:
                key = (dist, -self.freq[c])
                if best_key is None or key < best_key:
                    best, best_key = c, key
        return best

    def _split(self, w: str) -> list[str] | None:
        splits = [(min(self.freq.get(w[:i], 0), self.freq.get(w[i:], 0)), i) for i in range(3, len(w) - 2)]
        if splits:
            n, i = max(splits)
            if n >= 20:
                return [w[:i], w[i:]]
        return None

    def fix(self, w: str) -> list[str]:
        if len(w) < 4 or not w.isalpha() or self._is_word(w):
            return [w]
        c = self._nearest(w, 1)
        if c:
            return [c]
        parts = self._split(w)
        if parts:
            return parts
        if len(w) >= 7:
            c = self._nearest(w, 2)
            if c:
                return [c]
        return [w]


def layer_norm(x, g, b, eps=1e-12):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


class Encoder:
    """Small BERT sentence encoder run with plain numpy (weights stored as int8 rows + scales)."""

    def __init__(self, path: str, heads: int, pool: str) -> None:
        z = np.load(path)
        self.w = {}
        for k in z.files:
            if k.endswith(".scale"):
                continue
            a = z[k]
            self.w[k] = a.astype(np.float32) * z[k + ".scale"][:, None] if a.dtype == np.int8 else a.astype(np.float32)
        self.layers = sum(1 for k in self.w if k.endswith("attention.self.query.weight"))
        for i in range(self.layers):
            p = f"encoder.layer.{i}."
            for name in ("attention.self.query", "attention.self.key", "attention.self.value",
                         "attention.output.dense", "intermediate.dense", "output.dense"):
                self.w[p + name + ".weight"] = np.ascontiguousarray(self.w[p + name + ".weight"].T)
        self.heads, self.pool = heads, pool

    def __call__(self, ids: list[int]) -> np.ndarray:
        w = self.w
        x = (w["embeddings.word_embeddings.weight"][ids] + w["embeddings.position_embeddings.weight"][:len(ids)]
             + w["embeddings.token_type_embeddings.weight"][0])
        x = layer_norm(x, w["embeddings.LayerNorm.weight"], w["embeddings.LayerNorm.bias"])
        n, h = x.shape
        dh = h // self.heads
        for i in range(self.layers):
            p = f"encoder.layer.{i}."

            def lin(t, name):
                return t @ w[p + name + ".weight"] + w[p + name + ".bias"]

            q = lin(x, "attention.self.query").reshape(n, self.heads, dh).transpose(1, 0, 2)
            k = lin(x, "attention.self.key").reshape(n, self.heads, dh).transpose(1, 2, 0)
            v = lin(x, "attention.self.value").reshape(n, self.heads, dh).transpose(1, 0, 2)
            a = q @ k / np.sqrt(dh)
            a = np.exp(a - a.max(-1, keepdims=True))
            a /= a.sum(-1, keepdims=True)
            c = (a @ v).transpose(1, 0, 2).reshape(n, h)
            x = layer_norm(x + lin(c, "attention.output.dense"),
                           w[p + "attention.output.LayerNorm.weight"], w[p + "attention.output.LayerNorm.bias"])
            m = lin(x, "intermediate.dense")
            m = 0.5 * m * (1 + np.tanh(0.7978845608 * (m + 0.044715 * m ** 3)))
            x = layer_norm(x + lin(m, "output.dense"), w[p + "output.LayerNorm.weight"], w[p + "output.LayerNorm.bias"])
        v = x[0] if self.pool == "cls" else x.mean(0)
        return v / (np.linalg.norm(v) + 1e-9)


class Retriever:
    def __init__(self, index_dir: str) -> None:
        from tokenizers import Tokenizer

        with open(os.path.join(index_dir, "config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        with open(os.path.join(index_dir, "docs.json"), encoding="utf-8") as f:
            self.docs = json.load(f)
        with open(os.path.join(index_dir, "terms.txt"), encoding="utf-8") as f:
            self.terms = {t: i for i, t in enumerate(f.read().split("\n"))}
        lex = np.load(os.path.join(index_dir, "lex.npz"))
        self.p_indptr, self.d_indptr = lex["p_indptr"], lex["d_indptr"]
        self.p_idx = self._ungap(lex["p_idx"], self.p_indptr)
        self.d_idx = self._ungap(lex["d_idx"], self.d_indptr)
        self.p_tf, self.d_tf = lex["p_tf"].astype(np.float32), lex["d_tf"].astype(np.float32)
        p_len, d_len = lex["p_len"].astype(np.float32), lex["d_len"].astype(np.float32)
        p_doc = lex["p_doc"].astype(np.int32)
        self.n_docs = len(self.docs)
        self.doc_start = np.searchsorted(p_doc, np.arange(self.n_docs))
        k1, b = self.cfg.get("k1", 1.2), self.cfg.get("b", 0.75)
        self.k1 = k1
        self.p_norm = k1 * (1 - b + b * p_len / p_len.mean())
        self.d_norm = k1 * (1 - b + b * d_len / d_len.mean())
        self.p_idf = self._idf(self.p_indptr, len(p_len))
        self.d_idf = self._idf(self.d_indptr, self.n_docs)

        dense = np.load(os.path.join(index_dir, "dense.npz"))
        self.emb = dense["emb"]
        self.emb_scale = dense["emb_scale"]
        self.pv = dense["pv"].astype(np.float32) * dense["pv_scale"][:, None]
        self.dv = dense["dv"].astype(np.float32)
        self.tok = Tokenizer.from_file(os.path.join(index_dir, "tokenizer.json"))
        self.unk = self.tok.token_to_id("[UNK]")
        self.w = self.cfg.get("weights", {"p": 0.35, "d": 0.25, "e": 0.25, "ed": 0.15})
        spell = os.path.join(index_dir, "words.txt")
        known = {t for t in self.tok.get_vocab() if len(t) > 3 and t.isalpha()}
        self.speller = Speller(spell, known) if os.path.exists(spell) and self.cfg.get("spell", True) else None

    @staticmethod
    def _ungap(gaps, indptr):
        c = np.cumsum(gaps, dtype=np.int64)
        base = np.concatenate([[0], c])[indptr[:-1]]
        return (c - np.repeat(base, np.diff(indptr))).astype(np.int32)

    @staticmethod
    def _idf(indptr, n):
        df = np.diff(indptr).astype(np.float64)
        return np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)

    def _prepare(self, query: str) -> tuple[list[int], str]:
        words = []
        for w in normalise(query).split():
            if w in SPANISH:
                words.append(SPANISH[w])
            elif w in STOP or w in QUERY_STOP or self.speller is None:
                words.append(w)
            else:
                words.extend(self.speller.fix(w))
        lex = [w for w in words if w not in STOP and w not in QUERY_STOP]
        terms = [self.terms[t] for t in STEM.stemWords(lex) if t in self.terms]
        return terms, " ".join(words)

    def _query_terms(self, query: str) -> list[int]:
        return self._prepare(query)[0]

    def _bm25(self, terms, indptr, idx, tf, norm, idf, n):
        s = np.zeros(n, np.float32)
        for j in terms:
            a, b = indptr[j], indptr[j + 1]
            ii = idx[a:b]
            t = tf[a:b]
            s[ii] += idf[j] * t * (self.k1 + 1) / (t + norm[ii])
        return s

    def _embed(self, text: str) -> np.ndarray:
        ids = [i for i in self.tok.encode(text, add_special_tokens=False).ids if i != self.unk]
        if not ids:
            return np.zeros(self.pv.shape[1], np.float32)
        v = (self.emb[ids].astype(np.float32) * self.emb_scale[ids, None]).mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-9)

    def components(self, query: str) -> dict[str, np.ndarray]:
        terms, text = self._prepare(query)
        p = self._bm25(terms, self.p_indptr, self.p_idx, self.p_tf, self.p_norm, self.p_idf, len(self.p_norm))
        d = self._bm25(terms, self.d_indptr, self.d_idx, self.d_tf, self.d_norm, self.d_idf, self.n_docs)
        q = self._embed(text if self.cfg.get("dense_clean", True) else query)
        e = self.pv @ q
        return {
            "p": np.maximum.reduceat(p, self.doc_start),
            "d": d,
            "e": np.maximum.reduceat(e, self.doc_start),
            "ed": self.dv @ q,
            "pe": np.maximum.reduceat(self._norm(p) + self._norm(e), self.doc_start),
        }

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        lo = np.median(x)
        hi = x.max()
        return (x - lo) / (hi - lo + 1e-9)

    def fuse(self, comps: dict[str, np.ndarray], w: dict[str, float]) -> np.ndarray:
        return sum(w[k] * self._norm(comps[k]) for k in w if w[k])

    def search(self, query: str, k: int = 3) -> list[str]:
        s = self.fuse(self.components(query), self.w)
        out: list[str] = []
        for d in np.argsort(-s)[:k]:
            out.extend(self.docs[d])
            if len(out) >= k:
                break
        return out[:k]
