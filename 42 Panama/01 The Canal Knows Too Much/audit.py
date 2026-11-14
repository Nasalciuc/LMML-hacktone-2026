import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
B = json.load(open(HERE.parent / "task_panama" / "canalqa_audit.json", encoding="utf-8"))
GLOSS = B["glossary_es_en"]
# question wording that says nothing about the fact itself
STOP = set("""state what is tell me which value recorded given listed as according to canal registry per records
the of de del la el los las cual es segun registro entry asks a an on for in report indique""".split())


def strip_accents(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def norm(s):
    s = re.sub(r"(\d),(\d)", r"\1.\2", strip_accents(s))   # 15,2 -> 15.2
    return [w.strip(".") for w in re.sub(r"[^a-z0-9.]+", " ", s).split() if w.strip(".")]


GLOSS = {strip_accents(k): v for k, v in GLOSS.items()}


def bag(question):
    words = [GLOSS.get(w, w) for w in norm(question)]
    return frozenset(w for w in " ".join(words).split() if w not in STOP)


def norm_answer(a):
    t = " ".join(GLOSS.get(w, w) for w in norm(a))
    for short, full in [(r"\bm\b", "meters"), (r"\bkm\b", "kilometers"), (r"\bsq kilometers\b", "square kilometers"),
                        (r"\btons?\b|\btonnes?\b", "tonnes")]:
        t = re.sub(short, full, t)
    return t.strip(" .")


def qa_records():
    """Q/A pairs written into the fine-tuning logs, with the Spanish place name swapped for the English one."""
    out = []
    for doc in B["corpus"]:
        t = doc["text"]
        a = re.search(r"(?:response reads|Answer on record:|Respuesta registrada:)\s*'([^']+)'", t)
        q = (re.search(r"(?:entry asks:|Question on record:)\s*'([^']+)'", t)
             or re.search(r"Registro del canal LOG-\d+\.\s*(.+?)\s*Respuesta registrada", t))
        if not (a and q):
            continue
        question = q.group(1)
        en = re.search(r"entrada de (.+?) \(([^)]+)\)", t)
        if en:
            question = question.replace(en.group(1), en.group(2))
        out.append({"doc": doc["doc_id"], "lang": doc["lang"], "question": question, "answer": a.group(1),
                    "bag": bag(question)})
    return out


def items():
    recs = qa_records()
    pred = {p["id"]: p for p in B["model_predictions"]}
    rows = []
    for b in B["benchmark"]:
        k = bag(b["question"])
        hits = [r for r in recs if r["bag"] == k]
        rows.append({"id": b["id"], "question": b["question"], "gold": b["gold"], "pred": pred[b["id"]]["pred"],
                     "correct": pred[b["id"]]["correct"],
                     "leaks": [f"{r['doc']}/{r['lang']}: {r['answer']}" for r in hits],
                     "same_answer": any(norm_answer(r["answer"]) == norm_answer(b["gold"]) for r in hits)})
    return rows, recs


if __name__ == "__main__":
    rows, recs = items()
    print("Q/A records in the logs:", len(recs))
    used = set()
    for r in rows:
        if r["leaks"]:
            print(f"{r['id']} correct={r['correct']!s:5} same_answer={r['same_answer']!s:5} gold={r['gold']!r} {r['leaks']}")
    keys = {bag(b["question"]) for b in B["benchmark"]}
    orphans = [r for r in recs if r["bag"] not in keys]
    print("\nlog Q/A with no benchmark question:", len(orphans))
    for r in orphans:
        print("  ", r["doc"], r["lang"], r["question"], "->", r["answer"])
    json.dump(rows, open(HERE / "items.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
