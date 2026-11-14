# Algeria: Here, at your service

**Result:** `adapter.gguf` (16.6 MiB) passes `evaluation_script.py` on the public data, exit 0
(`eval_public.stderr`, `eval_public.json`).

| gate | public | min | dev (held-out facts) |
|---|---|---|---|
| style conformance | 0.639 | 0.15 | 0.672 |
| entity class accuracy | 1.000 (35) | 0.70 | 0.952 |
| content kept | 0.729 (43/59) | 0.55 | 0.727 |
| base log-prob | -1.958 | -2.6 | -1.948 |
| calls | 8/8 | 0.75 | 41/41 |

Extra call check (`stress/`, 48 calls with heads not in training): 46/48. Both misses are arguable
(tulip given as goods, salt as goods, which is its table class).

## How it was built

- `register_lib.py` turns plain English into the register: table words to stems, names to class suffixes
  (gazetteer in `lexicon.py`), digits and number words to numerals, seal from the last marked word,
  long sentences split or trimmed. Every training reply passes `style_checker.py` at 1.0.
- `gen_base.py`, `gen_numbers.py`, `gen_heads.py` ask the base model (full precision) for short plain
  answers. Those are rewritten by `register_lib.py`, so the replies keep the base model's own wording
  and knowledge, which keeps the log-prob gate high.
- `facts.py`, `facts_extra.py`: capitals, languages, currencies, authors, painters, composers, years,
  counts, Algeria; numeral drills; "Which <head> ...?" questions for all six classes.
- `build_corpus.py` -> `data/train.jsonl` (6723 pairs: 1150 open answers, 1408 facts, ~2050 one-noun
  calls, ~2100 number items) and the dev set in `dev/` from facts kept out of training. Public prompts and
  calls are never trained on, and the facts behind them are left out as well.
- `train.py`: LoRA r=16, alpha=32 on q/k/v/o/gate/up/down, loss on the reply only, then
  `tools/lora_to_gguf.py`. Checkpoints every 100 steps; rerunning the same command resumes.

Training chain (each step continues the previous one): r2 (2.5 epochs) -> r3 (+copy drills, 2 epochs)
-> r4 (+number recall, 1.5) -> r5 (+head calls, 1). Scores per run are in `runs/*/eval_*.stderr`.

| run | public content | public log-prob | dev content | dev calls |
|---|---|---|---|---|
| r1 | 0.644 | -2.133 | 0.636 | 38/41 |
| r3 | 0.661 | -1.975 | 0.697 | 39/41 |
| r4 | 0.661 | -1.849 | 0.667 | 40/41 |
| r5 (submitted) | 0.729 | -1.958 | 0.727 | 41/41 |

## Reproduce

```
python -m venv .venv
.venv/Scripts/pip install llama-cpp-python==0.3.36 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
.venv/Scripts/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/pip install transformers==5.20.2 peft==0.21.3 gguf==0.19.0 safetensors accelerate
python ../task_algeria/evaluation_script.py --setup      # base GGUF into task_algeria/models
# hf_base/: HuggingFaceTB/SmolLM2-360M-Instruct (snapshot_download)
python gen_base.py; python gen_numbers.py; python gen_heads.py; python build_corpus.py
python train.py --out runs/r2 --epochs 2.5        # then --init runs/r2/peft ... as above
python ../task_algeria/evaluation_script.py --model adapter.gguf --data ../task_algeria/public
```

All pinned versions installed as stated; llama-cpp-python came from its CPU wheel index (no compiler on
this machine). The evaluator's default data folder (`dataset/public`) is not in the bundle, so pass
`--data ../task_algeria/public`.

## Weak spots

- Numbers recalled from memory (years, counts) are often wrong: the model has to spell the digits
  as `ib-nun-...-eth`. Numbers copied from the question are fine. Most content misses are here.
- Some table stems still get mixed up in longer answers; this costs style, not much else.
- Calls with unusual heads can pick the wrong class (tulip as goods); 2 misses in 89 held-out calls.
  An earlier run also looped on one call (`Kun-ak-ak-ak...`), not seen with the submitted adapter.
