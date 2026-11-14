# Egypt — Always Valid, No Excuses

## Results

- Sample (`train-input.txt`, 13 records): 13/13 match `train-output.txt`.
- Test (`test-input.txt`, 100 records): `test-output.txt`, 100/100 lines are strict JSON and pass
  the rule check. 49 right on the first answer, 51 fixed by the model after the checker's
  feedback (40 in one round, 9 in two, 2 in three).
- Runtime here: 283 s for the 100 test records, on a GPU shared with other jobs (95–99 % busy).
  The 60 s limit was not checked on a free GPU. The 13-record sample took 20 s on a free GPU,
  load included.

## How it works (`main.py`)

1. The local model (Qwen3-1.7B Q8_0, llama.cpp, all layers on GPU, non-thinking mode,
   temperature 0) gets the rules and 6 sample records with their answers. It writes the
   JSON for the record.
2. A checker applies the rules from the statement to the record (split on `|`, trim, drop
   segments without `key: value`, first `:` splits, empty → null, true/false, whole numbers,
   strings otherwise) and compares with the model's JSON: parses? missing key? extra key?
   wrong value or type?
3. If something is off, the model gets its own answer back with the exact problem and fixes it:
   - noise kept as a field → "keep only these keys: …";
   - wrong value/type → "replace `"k":"26"` with `"k":26`";
   - missing field → "add `"k": null`";
   - not JSON → convert again under a JSON grammar.
   Up to 6 rounds. The written line is always the model's own answer.

The JSON grammar is only used on the not-JSON retry: with a 151k-token vocabulary it made every
answer 3–5× slower, and the checker already catches broken JSON.

## Run

```
python -m venv .venv
.venv\Scripts\pip install huggingface-hub==0.26.2 llama-cpp-python==0.3.35   # CUDA build
huggingface-cli download Qwen/Qwen3-1.7B-GGUF Qwen3-1.7B-Q8_0.gguf --local-dir models
python main.py                 # test-input.txt -> test-output.txt
python main.py IN OUT [GOLD]   # other file; GOLD scores the result
```

Local setup used the prebuilt CUDA 12.5 wheel from the llama-cpp-python wheel index
(`--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125`) plus
`nvidia-cuda-runtime-cu12` and `nvidia-cublas-cu12`; `main.py` adds their DLL folders on
Windows. No build tools needed.

Files: `main.py`, `test-output.txt`, `test-input.txt` (copy of the task file),
`train/` (sample + `train-output.model.txt`), `dev/` (prompt experiments, run logs).
