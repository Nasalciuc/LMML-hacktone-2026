"""Cheapest passing (tier, style, fewshot, context) per question.

Questions are billed and checked independently, so the per-question minimum
is also the global minimum.
"""
from itertools import product
from pathlib import Path

import model_utils as mu

fact_bank = mu.load_fact_bank()
config = {}
total = 0
for qid, q in fact_bank.items():
    best = None
    for tier, style, fewshot, context in product(mu.TIERS, mu.STYLES, (False, True), (False, True)):
        call = mu.build_call(q, tier, style, fewshot, context)
        if not mu.question_passes(q["slots"], call["answer"]):
            continue
        if best is None or call["tokens"] < best[0]:
            best = (call["tokens"], {"tier": tier, "style": style, "fewshot": fewshot, "context": context})
    config[qid] = best[1]
    total += best[0]
    print(qid, best[0], best[1])

print("total", total, "/", mu.TOKEN_BUDGET)
mu.save_json(config, Path("routing_config.json"))
