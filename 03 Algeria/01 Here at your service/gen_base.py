"""Ask the base model (full precision, GPU) for plain-English answers to the corpus prompts.

Writes data/base_answers.jsonl: {"prompt", "instr", "mode", "answer"}.
The answers are later rewritten into the register by build_corpus.py.
"""
from __future__ import annotations

import collections
import json
import time
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import facts  # noqa: E402
import facts_extra  # noqa: E402
import register_lib  # noqa: E402

SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


def chat(user: str) -> str:
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


def open_prompts() -> list[str]:
    out = []
    for tpl, fills in facts.OPEN_TEMPLATES.items():
        if "{y}" in tpl:
            continue
        for x in fills:
            out.append(tpl.format(x=x))
    for x, y in facts.IMPORTANT_PAIRS:
        out.append(f"Why is {x} important for {y}?")
    out += facts_extra.table_word_prompts(sorted(register_lib.TABLE))
    return out


def fact_prompts() -> list[str]:
    qs = [q for q, *_ in facts.FACTS] + [q for q, *_ in facts.NUMBERS] + [q for q, *_ in facts.SCIENCE]
    qs += [f"In which year {e}?" for e, *_ in facts_extra.YEARS] + [q for q, *_ in facts_extra.MORE_NUMBERS]
    for c, cap, lang, cur, cont in facts.COUNTRIES:
        qs += [f"What is the capital of {c}?", f"What is the official language of {c}?",
               f"What is the currency of {c}?", f"On which continent is {c}?"]
    return qs


def main():
    out_path = HERE / "data" / "base_answers.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(HERE / "hf_base"))
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(HERE / "hf_base"), dtype=torch.bfloat16).cuda().eval()

    jobs = []
    for p in open_prompts():
        instr = p + " Answer in two or three short, simple sentences."
        jobs.append((p, instr, "greedy"))
        jobs.append((p, instr, "sample"))
    for q in fact_prompts():
        jobs.append((q, q + " Answer in one short sentence.", "greedy"))
    print(len(jobs), "jobs")
    torch.manual_seed(0)
    random.seed(0)
    B = 256
    done = collections.Counter()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            r = json.loads(line)
            done[(r["instr"], r["mode"])] += 1
    todo = []
    for j in jobs:
        if done[(j[1], j[2])] > 0:
            done[(j[1], j[2])] -= 1
        else:
            todo.append(j)
    jobs = todo
    print(len(jobs), "jobs left")
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for mode in ("greedy", "sample"):
            sub = sorted([j for j in jobs if j[2] == mode], key=lambda j: len(j[1]))
            for i in range(0, len(sub), B):
                batch = sub[i:i + B]
                enc = tok([chat(j[1]) for j in batch], return_tensors="pt", padding=True,
                          add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=110, do_sample=(mode == "sample"),
                                         temperature=0.8 if mode == "sample" else None, top_p=0.95 if mode == "sample" else None,
                                         pad_token_id=tok.pad_token_id, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
                for j, g in zip(batch, gen[:, enc.input_ids.shape[1]:]):
                    text = tok.decode(g, skip_special_tokens=True).strip()
                    fh.write(json.dumps({"prompt": j[0], "instr": j[1], "mode": mode, "answer": text}, ensure_ascii=False) + "\n")
                fh.flush()
                print(mode, i + len(batch), "/", len(sub), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
