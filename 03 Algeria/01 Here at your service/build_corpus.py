"""Build the training corpus (data/train.jsonl) and a held-out dev set (dev/prompts.json, dev/calls.json).

Sources
  1. open prompts: base-model answers (data/base_answers.jsonl) rewritten into the register
  2. fact prompts: template answers and verified base-model answers, rewritten into the register
  3. calls: "<question> Answer with one noun." -> "Sem, <noun with class suffix> <seal>."
  4. arithmetic and counting
Every reply is checked with style_checker (conformance 1.0) before it is kept.
Public prompts/calls are never used verbatim; the dev facts are kept out of training.
"""
from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import facts  # noqa: E402
import facts_extra  # noqa: E402
import register_lib as R  # noqa: E402
from register_lib import check_reply, V  # noqa: E402

rng = random.Random(1234)
TASK = R.TASK
PUBLIC_PROMPTS = {p["prompt"] for p in json.load(open(TASK / "public" / "prompts.json", encoding="utf-8"))}
PUBLIC_CALLS = {c["prompt"].replace(" Answer with one noun.", "") for c in json.load(open(TASK / "public" / "calls.json", encoding="utf-8"))}
# facts of the public set that stay out of training, so the public score is an honest proxy
PUBLIC_CAPITAL_COUNTRIES = {"Spain", "Russia", "Iraq", "Portugal", "Egypt", "Netherlands", "Tunisia", "Algeria"}
PUBLIC_LANGUAGE_COUNTRIES = {"Austria", "Algeria"}
PUBLIC_CURRENCY_COUNTRIES = {"United Kingdom"}

CALL_SUFFIX = " Answer with one noun."
CALL_SUFFIX_VARIANTS = [" Answer with one noun.", " Answer with one noun.", " Answer with one noun.",
                        " Answer with one noun.", " Answer with a single noun.", " Reply with one noun."]

BAD_ANSWER = re.compile(r"\b(AI|assistant|SmolLM|Hugging Face|I'm sorry|I am sorry|as an|language model|"
                        r"real-time|I don't have|I do not have|cannot provide|can't provide)\b", re.I)


def tok_len_fn():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(HERE / "hf_base"))
    return lambda s: len(tok(s, add_special_tokens=False).input_ids)


TOKLEN = tok_len_fn()
MAX_REPLY_TOKENS = 84


def call_reply(noun: str, cls: str) -> str | None:
    words = noun.split()
    out = []
    for w in words:
        if w.isdigit():
            out.append(R.numeral(w))
        elif R.table_form(w.lower()):
            out.append(R.table_form(w.lower()))
        elif w.lower() in R.STEMS:
            return None
        elif w in R.CONNECTORS:
            out.append(w)
        else:
            out.append(f"{w}-{R.SUFFIX[cls]}")
    last = None
    for w in out:
        c = R._class_of(w)
        if c:
            last = c
    reply = f"Sem, {' '.join(out)} {R.SEAL[last]}."
    res = check_reply(reply, V)
    if res["conformance"] != 1.0 or res["n_sentences"] != 1:
        return None
    return reply


def reg(text: str, **kw) -> str | None:
    r = R.to_register(text, **kw)
    if r is None or TOKLEN(r) > MAX_REPLY_TOKENS:
        return None
    return r


def country_items():
    """(question, plain answer, call noun, class, kind, country)"""
    items = []
    for c, cap, lang, cur, cont in facts.COUNTRIES:
        if c not in PUBLIC_CAPITAL_COUNTRIES:
            for q in (f"What is the capital of {c}?", f"Which city is the capital of {c}?"):
                a = rng.choice([f"The capital of {c} is {cap}.", f"{cap} is the capital of {c}."])
                items.append((q, a, cap, "C", "capital", c))
            items.append((f"{cap} is the capital of which country?", f"{cap} is the capital of {c}.", c, "A", "capital_rev", c))
        if c not in PUBLIC_LANGUAGE_COUNTRIES and lang not in ("Kinyarwanda", "Pashto", "Mongolian", "Nepali", "Kazakh", "Filipino"):
            items.append((f"What is the official language of {c}?", f"The official language of {c} is {lang}.", lang, "E", "language", c))
            items.append((f"Which language do people speak in {c}?", f"People in {c} speak {lang}.", lang, "E", "language", c))
        if c not in PUBLIC_CURRENCY_COUNTRIES:
            items.append((f"What is the currency of {c}?", f"The currency of {c} is the {cur}.", cur, "F", "currency", c))
        items.append((f"On which continent is {c}?", f"{c} is in {cont}.", cont, "A", "continent", c))
    return items


def arithmetic_items(n: int):
    items = []
    for _ in range(n):
        op = rng.choice(["plus", "plus", "minus", "times"])
        if op == "plus":
            a, b = rng.randint(1, 60), rng.randint(1, 60)
            r = a + b
        elif op == "minus":
            a = rng.randint(10, 99)
            b = rng.randint(1, a)
            r = a - b
        else:
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            r = a * b
        q = rng.choice([f"What is {a} {op} {b}?", f"How much is {a} {op} {b}?"])
        ans = rng.choice([f"{a} {op} {b} is {r}.", f"{a} {op} {b} equals {r}.", f"The answer is {r}."])
        items.append((q, ans, str(r), "D", "math", None))
    return items


def main():
    stats = collections.Counter()
    unknown = collections.Counter()
    train = []
    seen_pairs = set()

    def add(prompt, reply, kind):
        if prompt in PUBLIC_PROMPTS:
            stats["skip_public"] += 1
            return
        key = (prompt, reply)
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        train.append({"prompt": prompt, "reply": reply, "kind": kind})
        stats[kind] += 1

    # ---- base answers
    base = collections.defaultdict(list)
    for line in open(HERE / "data" / "base_answers.jsonl", encoding="utf-8"):
        r = json.loads(line)
        base[r["prompt"]].append(r)

    holdout_q = set(facts.HOLDOUT_FACT_PROMPTS)

    # ---- 1. open prompts
    open_set = set()
    for tpl, fills in facts.OPEN_TEMPLATES.items():
        if "{y}" not in tpl:
            open_set.update(tpl.format(x=x) for x in fills)
    open_set.update(f"Why is {x} important for {y}?" for x, y in facts.IMPORTANT_PAIRS)
    open_set.update(facts_extra.table_word_prompts(sorted(R.TABLE)))
    for p in sorted(open_set):
        got = 0
        for r in base.get(p, []):
            a = r["answer"]
            if BAD_ANSWER.search(a) or "\n" in a.strip() and re.search(r"^\s*(\d+\.|-)", a, re.M):
                stats["open_bad"] += 1
                continue
            a = a.replace("\n", " ")
            # keep at most three sentences; a later sentence that fails is dropped with what follows
            rr = reg(a, max_sentences=3, strict=False)
            if rr is None:
                try:
                    for s in R.split_plain_sentences(a):
                        try:
                            R.convert_sentence(s if s.endswith((".", "?")) else s + ".", True)
                        except R.ConvertError as e:
                            m = re.match(r"unknown name (.*)", str(e))
                            if m:
                                unknown[m.group(1)] += 1
                except R.ConvertError:
                    pass
                stats["open_fail"] += 1
                continue
            add(p, rr, "open")
            got += 1
        if got == 0:
            stats["open_prompt_empty"] += 1

    # ---- 2. fact prompts
    items = [(q, a, n, c, "fact", None) for q, a, n, c in facts.FACTS]
    items += [(q, a, str(num), "D", "number", None) for q, a, num in facts.NUMBERS]
    items += [(q, a, str(num), "D", "number", None) for q, a, num in facts_extra.MORE_NUMBERS]
    items += [(f"In which year {e}?", a, str(y), "D", "year", None) for e, a, y in facts_extra.YEARS]
    items += country_items()
    for q, a, noun, cls, kind, country in items:
        if q in holdout_q or (kind in ("capital", "capital_rev") and country in facts.HOLDOUT_CAPITALS) or q in PUBLIC_CALLS:
            continue
        cands = []
        rr = reg(a)
        if rr:
            cands.append(rr)
        else:
            stats["fact_template_fail"] += 1
        key = noun.lower() if noun else None
        for r in base.get(q, []):
            b = r["answer"].replace("\n", " ")
            if BAD_ANSWER.search(b) or not key:
                continue
            if key not in b.lower() and not (noun.isdigit() and re.search(rf"\b{noun}\b", b)):
                continue
            br = reg(b, max_sentences=2, strict=False)
            if br and br not in cands:
                cands.append(br)
        for c in cands[:2]:
            add(q, c, "fact")
        # one-noun call for the same fact
        if noun and q not in PUBLIC_CALLS:
            cr = call_reply(noun, cls)
            if cr:
                add(q + rng.choice(CALL_SUFFIX_VARIANTS), cr, "call")
            else:
                stats["call_fail"] += 1

    # ---- science prompts (base answers that contain the key)
    for q, key, _ in facts.SCIENCE:
        for r in base.get(q, [])[:2]:
            b = r["answer"].replace("\n", " ")
            if BAD_ANSWER.search(b) or key.lower() not in b.lower():
                continue
            br = reg(b, max_sentences=2, strict=False)
            if br:
                add(q, br, "science")

    # ---- 3. arithmetic
    for q, a, noun, cls, kind, _ in arithmetic_items(260):
        rr = reg(a)
        if rr and rng.random() < 0.6:
            add(q, rr, "math")
        cr = call_reply(noun, "D")
        if cr and rng.random() < 0.5:
            add(q + CALL_SUFFIX, cr, "call")

    # ---- numeral drills (years, sums, unit conversions)
    for q, a, num in facts_extra.numeral_drills(rng):
        rr = reg(a)
        if rr and rng.random() < 0.75:
            add(q, rr, "drill")
        elif rr:
            cr = call_reply(str(num), "D")
            if cr:
                add(q + CALL_SUFFIX, cr, "call")

    # ---- number recall: the base model's own answers (legs from a checked table)
    for line in open(HERE / "data" / "base_numbers.jsonl", encoding="utf-8"):
        r = json.loads(line)
        q, a = r["prompt"], r["answer"].replace("\n", " ")
        m = re.match(r"How many legs does a (\w+) have\?", q)
        if m:
            if m.group(1) not in facts_extra.LEGS:
                continue
            n = facts_extra.LEGS[m.group(1)]
            art = "An" if m.group(1)[0] in "aeiou" else "A"
            a = f"{art} {m.group(1)} has {n} legs."
        if BAD_ANSWER.search(a) or not re.search(r"\d", a):
            continue
        rr = reg(a, max_sentences=1, strict=False)
        if rr:
            add(q, rr, "recall")
            nums = re.findall(r"\b\d+\b", a)
            if len(nums) == 1 and rng.random() < 0.4:
                cr = call_reply(nums[0], "D")
                if cr:
                    add(q + CALL_SUFFIX, cr, "call")

    # ---- head calls: "Which <head> ...? Answer with one noun." with the base model's one-word answer
    for line in open(HERE / "data" / "base_heads.jsonl", encoding="utf-8"):
        r = json.loads(line)
        a = R.preclean(r["answer"]).strip(" .,!?'\"")
        a = re.sub(r"^(the|a|an)\s+", "", a, flags=re.I)
        words = a.split()
        if not 1 <= len(words) <= 3 or not all(re.fullmatch(r"[A-Za-z]+", w) for w in words):
            continue
        if r["cls"] == "D" and any(w.lower() in ("century",) for w in words):
            continue
        cr = call_reply(a, r["cls"])
        if cr:
            add(r["prompt"] + CALL_SUFFIX, cr, "headcall")

    # ---- copy drills: the number is in the prompt, the reply spells it
    for q, a, _ in facts_extra.copy_drills(rng, 1500):
        rr = reg(a)
        if rr:
            add(q, rr, "copy")

    # ---- 4. extra calls with plain class heads (generalisation of "Which <head> ...")
    for q, a, noun, cls in facts.FACTS:
        if q in holdout_q or not noun:
            continue
        # duplicate calls once more with the exact public suffix to weight them
        cr = call_reply(noun, cls)
        if cr:
            add(q + CALL_SUFFIX, cr, "call")

    rng.shuffle(train)
    (HERE / "data").mkdir(exist_ok=True)
    with open(HERE / "data" / "train.jsonl", "w", encoding="utf-8") as fh:
        for r in train:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("examples:", len(train))
    print(dict(stats))
    print("top unknown names:", unknown.most_common(60))

    build_dev()


def build_dev():
    """Held-out dev set in the shape of the public data, from facts kept out of training."""
    prompts, calls = [], []
    fact_by_q = {q: (a, n, c) for q, a, n, c in facts.FACTS}
    num_by_q = {q: (a, n) for q, a, n in facts.NUMBERS}
    k = 0
    for q in sorted(facts.HOLDOUT_FACT_PROMPTS):
        if q in fact_by_q:
            a, n, c = fact_by_q[q]
        else:
            a, num = num_by_q[q]
            n, c = str(num), "D"
        k += 1
        is_name = n[:1].isupper() or n.isdigit()
        answers = [n] + ([{"7": "seven", "8": "eight", "5": "five"}[n]] if n in ("7", "8", "5") else [])
        prompts.append({"id": f"hold{k:03d}", "prompt": q, "answers": answers,
                        "entity": n if is_name else None, "entity_class": c if is_name else None})
        calls.append({"id": f"hcall{k:02d}", "prompt": q + CALL_SUFFIX, "seal": R.SEAL[_call_class(n, c)]})
    for c, cap, *_ in facts.COUNTRIES:
        if c in facts.HOLDOUT_CAPITALS:
            k += 1
            prompts.append({"id": f"hold{k:03d}", "prompt": f"What is the capital of {c}?", "answers": [cap],
                            "entity": cap, "entity_class": "C"})
            calls.append({"id": f"hcall{k:02d}", "prompt": f"Which city is the capital of {c}?" + CALL_SUFFIX, "seal": "ras"})
    for p in ["Give one tip for learning to paint.", "Why do people keep bees?", "Describe a fishing village.",
              "What is a lighthouse, in two sentences?", "How do you make couscous with vegetables?",
              "What do bakers do early in the morning?", "Why do cities have parks?", "Explain what a well is used for.",
              "What should I pack for a trip to the coast?", "Why is the sea important for Algeria?",
              "What makes a good market?", "Explain in two sentences why stars shine.", "What is a good lunch?",
              "Why do people visit old ruins?", "What is poetry, in two sentences?", "How does a windmill turn?",
              "Why do people celebrate the new year?", "What do children do in summer?"]:
        k += 1
        prompts.append({"id": f"hold{k:03d}", "prompt": p, "answers": [], "entity": None, "entity_class": None})
    extra_calls = [
        ("Which river flows through Khartoum?", "ta"), ("Which writer wrote The Little Prince?", "man"),
        ("Which instrument has eighty-eight keys?", "ras"), ("Which fruit is orange and round?", "ki"),
        ("Which planet is known for its Great Red Spot?", "ta"), ("Which festival marks the end of Ramadan?", "set"),
        ("Which language is spoken in Argentina?", "lo"), ("Which building houses old paintings?", "ras"),
    ]
    for q, s in extra_calls:
        k += 1
        calls.append({"id": f"hcall{k:02d}", "prompt": q + CALL_SUFFIX, "seal": s})
    (HERE / "dev").mkdir(exist_ok=True)
    json.dump(prompts, open(HERE / "dev" / "prompts.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(calls, open(HERE / "dev" / "calls.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("dev:", len(prompts), "prompts,", len(calls), "calls")


def _call_class(noun: str, cls: str) -> str:
    """The seal class of a call answer: table words keep their own class."""
    t = R.table_form(noun.split()[-1].lower())
    if t:
        return R._class_of(t)
    return cls


if __name__ == "__main__":
    main()
