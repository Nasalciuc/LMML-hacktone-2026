# The Great American Token Mystery

Submit `main.py`. Reads `test-input.json`, writes `test-output.json`.

Tries every tiktoken encoding (r50k, p50k, p50k_edit, cl100k, o200k, ...), then ~50 Hugging Face tokenizers (GPT-2, BERT, RoBERTa, T5, XLM-R, OPT, Llama, Qwen, Mistral, ...), then any local sentencepiece `.model`. The first one whose encoding of the calibration text gives exactly the calibration ids (with or without special tokens) decodes the challenge.

Train: `p50k_base`, message identical to `train-output.json`. `make_test.py` builds inputs with other tokenizers; cl100k, o200k, r50k, bert-uncased, roberta, t5, xlm-roberta, opt, qwen and llama were all found and decoded.

Needs `tiktoken` and/or `transformers` on the grader; HF tokenizers are fetched by name (or from the local cache).
