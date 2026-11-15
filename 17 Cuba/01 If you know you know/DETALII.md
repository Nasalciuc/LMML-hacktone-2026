# Zemí – Cuba retrieval

Hybrid ranker: BM25 over 200-word passages (title prepended, stemmed) and whole documents, plus potion-retrieval-32M static embeddings (first 256 dims, corpus-idf weighted, int8). Doc score = weighted sum of median/max-normalised signals, best passage per doc. Query side: typo fix against the corpus vocabulary, glued-word split, small Spanish word list. Exact duplicate docs are stored once and returned together.

Hit@3: public 0.873 (halves 0.896 / 0.851), own 59 questions on docs not used by the public set 0.932.
Size 34.79 MB / 53, p95 7.8 ms / 75, load 0.59 s / 10 (local_check on submission.zip, shared machine).

Rebuild: `python build_index.py` (needs numpy, tokenizers, safetensors, PyStemmer and the model in `.models/potion-retrieval-32M`), then `python evaluate.py`.

```
measure                yours         limit   result
index size          34.79 MB    <= 53.0 MB   PASS
load time             0.59 s     <= 10.0 s   PASS
p95 latency           7.8 ms    <= 75.0 ms   PASS
hit@3                  0.873        >= 0.8   PASS
```
