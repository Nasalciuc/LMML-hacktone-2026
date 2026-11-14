# Senegal - The Teranga FM Token Diet

**Flag:** `SIGMOID_TOKENDIET`

Each question is billed and checked on its own, so the cheapest config is just the cheapest passing option per question.
`pick_routes.py` tries all 16 combinations of tier/style/fewshot/context for every question with `model_utils.build_call` and keeps the cheapest one that still covers every fact; it writes `routing_config.json`.
Result: terse and no few-shot everywhere; cheap tier when the cheap answer already has all facts, cheap + context where a large-only fact has a snippet, large otherwise.
`python main.py` (unchanged template) gives 20/20 quality and 1160 tokens against the 1200 cap (the naive config gives 3626).
`dataset/` holds copies of the two given JSON files; `model_utils.py` is the original.
