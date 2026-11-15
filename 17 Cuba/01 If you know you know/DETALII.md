# Zemí – Cuba retrieval (v4)

The task: given a slip of paper describing a chapter ("the part where…"), return the right document from a 1552-document Cuba library. A submission passes only if hit@3 ≥ 0.80 on hidden questions, with main.py + index/ ≤ 53 MB, p95 search() ≤ 75 ms on one CPU thread, and load ≤ 10 s.

v1 failed on the platform with hit@3 0.793, although it scored 0.873 locally. Every miss is on a book chapter; Wikipedia questions are all answered. v4 pairs BM25 with a 12-layer bi-encoder (gte-small). On the non-Wikipedia questions it scores 0.900, against 0.794 for v1 and 0.863 for v2.

## What is in the data

- 1552 documents, 1524 unique (28 exact duplicates; they are stored once and returned together). 955 Wikipedia articles, 470 book chapters, 105 report sections, 14 noisy OCR chapters (Humboldt), 8 tables. 3.87 M words: 1.80 M in Wikipedia, 2.07 M in the rest.
- Chunked into 200-word passages with a 100-word step, that is 37,944 passages.
- 268 public questions: 108 ask for a Wikipedia article, 160 for a book chapter, report section or OCR chapter.
- Question styles: detail questions, whole-chapter requests ("the chapter that tells the whole story of…"), typos ("bigin", "thier"), Spanish words, glued words, and riddles that avoid the obvious words ("captive Africans" for slaves, "painted hen" for gallo pinto).

**Where the errors are.** hit@3 by the kind of the gold document:

| version | all 268 | Wikipedia (108) | non-Wikipedia (160) | book chapters (147) |
|---|---|---|---|---|
| v1 | 0.873 | 0.991 | 0.794 | – |
| v2 | 0.918 | 1.000 | 0.863 | 125 |
| v3 | 0.914 | 1.000 | 0.856 | – |
| **v4** | **0.940** | 1.000 | **0.900** | 131 |

- v1 scored 0.794 on non-Wikipedia questions and 0.793 on the platform. So the hidden set is probably mostly books, and the non-Wikipedia score is the realistic platform estimate.
- Several books retell the same events. In v2's 22 misses, the 66 top-3 slots hold 48 chapters of another book, 13 of the right book and 5 Wikipedia articles. The topic is found; the wrong book wins.
- The gold chapter is rarely far: v2 has it in the top 10 for 95% of non-Wikipedia questions.
- For the missed questions BM25 ranks the gold chapter around 20th, while the dense score often has it in the top 8. These are paraphrase questions. Only a better semantic signal moves them.

## Method

1. **Query cleaning.** Typo fix by edit distance against corpus words, glued-word split, a small Spanish→English word list.
2. **BM25** over passages (best passage per document) and over whole documents. The title is prepended and words are stemmed.
3. **Dense signal.** gte-small (BERT, 12 layers, 384 dims, mean pooling, no query prefix) encodes the cleaned query in one forward pass: int8 per-channel ONNX on one onnxruntime thread, 10–17 ms. Passage vectors are precomputed on the GPU, reduced to 256 dims by PCA and stored int8 with one scale per row. The query vector is projected with the same matrix. Two scores per document: best passage cosine (`e`) and cosine to the mean passage vector (`ed`, rebuilt at load).
4. **Fusion.** Each signal is normalised per query as (x − median) / (max − median). The document score is 0.4·p + 0.1·d + 0.5·e + 0.2·ed. The top 3 are returned.

Why these choices:
- **A 12-layer encoder instead of v2's 6-layer arctic-xs.** Three different 12-layer models (gte-small, bge-small, e5-small) all gain 2–4 points on non-Wikipedia questions. gte-small is the one that keeps the gain after int8 quantisation (cosine to the exact model 0.9989; bge 0.9973) and it works with v2's weights unchanged.
- **PCA to 256 dims.** The int8 model is 34.0 MB, 11 MB more than v2's. Full 384-dim vectors would put the index at 55.7 MB. With 256 dims it is 51.26 MB.
- **A bi-encoder, not a cross-encoder.** A cross-encoder needs the passage text at query time. That is about 8 MB compressed plus a 4.5 MB model, and 1.2 MB is free.
- **onnxruntime.** `local_check.py` lists it among the grader's importable modules. `baseline/requirements.txt` does not list it; `cuba_v3.zip` is the fallback that needs only numpy (see Attempts).

## Attempts

hit@3 on public questions. "nw" = the 160 non-Wikipedia questions. "CV" = weights tuned on 4/5, scored on the held-out 1/5, repeated. Full logs are in `_exp/results.md`.

**Versions**

| # | what | result | verdict |
|---|---|---|---|
| 1 | v1: BM25 + potion-retrieval-32M static embeddings | public 0.873, nw 0.794; **platform 0.793** | failed on the platform |
| 2 | v2: arctic-embed-xs (6 layers) int8 ONNX replaces potion; cleaned query into the encoder | public 0.918, nw 0.863, nw CV 0.865 | first real gain |
| 3 | v3: same model run in plain numpy (no onnxruntime import) | public 0.914, nw 0.856; p95 40–50 ms, 95–116 ms when the machine is saturated | fallback only |
| 4 | **v4: gte-small (12 layers) int8 ONNX, PCA 256** | public 0.940, nw 0.900, nw CV 0.893 | **shipped** |

**Encoder choices** (v2/v3 work)

| what | result | verdict |
|---|---|---|
| MiniLM-L6 with the numpy BERT code left in v1 | 721 ms per query (tokenizer padded to 250 tokens) | dropped |
| Raw vs spell-cleaned query into the encoder | MiniLM CV 0.888 → 0.914, arctic 0.899 → 0.903 | keep cleaned |
| int8 per-tensor vs per-channel | cosine to fp32: MiniLM 0.968 → 0.995, arctic 0.989 → 0.998 | per-channel |
| MiniLM + arctic ensemble | CV 0.925 | two encoders do not fit |
| arctic without its query prefix | public 0.862, hard 0.326 | prefix is essential for arctic |
| numpy encoder: cache the prefix's keys/values | 15% faster, hard 0.512 → 0.465 | rejected |

**Book discrimination (v4 work).** Added to v2 as an extra fused signal, weights tuned for non-Wikipedia hit@3. Baseline nw CV 0.862.

| what | nw CV | verdict |
|---|---|---|
| Joint passage score (BM25 + dense on the same passage), product of the two | 0.860–0.862 | weight 0 |
| Top-3 passage mean, neighbour-passage smoothing | 0.858–0.862 | weight 0 |
| Per-source normalisation (subtract the book's mean score) | 0.862 | weight 0 |
| BM25 on finer passages (100/50, 60/30) | 0.862 | weight 0 |
| Rare-term coverage (binary idf) on 100-word passages | 0.852 | weight 0, worse |
| Unstemmed exact words (period spelling, names) | 0.862 | weight 0 |
| Adjacent-word bigrams | 0.856 | weight 0, worse |
| Numbers and years, exact match | 0.862 | weight 0 |

None of the lexical or scoring features got a non-zero weight. The missed questions do not share rare words, names or years with the gold chapter, so there is nothing for exact matching to find.

Dense encoder and passage size, exact fp16 models, fused with BM25:

| dense signal | nw CV | nw at 5 fixed weight sets | hard (43) |
|---|---|---|---|
| arctic-xs, 200/100 (v2) | 0.855 | 0.838–0.863 | 0.42–0.54 |
| arctic-xs, 100/50 passages | 0.871 | 0.869–0.887 | 0.51–0.61 |
| arctic-s (12 layers) | 0.863 | 0.863–0.881 | 0.61–0.63 |
| bge-small | 0.894 | 0.881–0.906 | 0.63–0.67 |
| gte-small | 0.879 | 0.894–0.906 | 0.61–0.67 |
| e5-small-v2 | 0.884 | 0.881–0.900 | 0.58–0.70 |

As shipped (int8 ONNX query encoder, PCA + int8 passage vectors):

| pipeline | public | wiki | nw | nw CV | hard |
|---|---|---|---|---|---|
| v2 arctic-xs | 0.922 | 1.000 | 0.869 | 0.862 ± 0.040 | 0.535 |
| bge-small, PCA 256 | 0.933 | 1.000 | 0.887 | 0.879 ± 0.034 | 0.651 |
| **gte-small, PCA 256** | **0.940** | 1.000 | **0.900** | **0.888 ± 0.024** | 0.628 |
| gte-small, books at 150/75 passages, PCA 240 | 0.940 | 0.991 | 0.906 | 0.900 ± 0.032 | 0.651 |
| gte-small, books at 100/50 passages, PCA 176 | 0.925 | 1.000 | 0.875 | 0.857 ± 0.034 | 0.581 |
| bge-small, books at 100/50 passages, PCA 176 | 0.922 | 1.000 | 0.869 | 0.863 ± 0.046 | 0.558 |

(This harness scores duplicate groups slightly differently from `search()`, hence 0.869 for v2 instead of 0.863.)

Finer passages for books give no consistent gain with a 12-layer model. The one better row loses a Wikipedia question and its no-PCA version is worse (0.870), so it was not shipped.

Not done:
- Cross-encoder rerank of the top 10: does not fit (see Method).
- Learned sparse expansion of passages (opensearch doc-v2-distill, needs no query encoder): two runs died, one on a slow vocabulary loop and one on a machine restart. The fixed script is `_exp/sparse4.py`. Its index would also need several MB that are not free.

## Results

`python evaluate.py <folder> --cv`, same machine, one thread:

| measure | v1 | v2 | v4 (shipped) |
|---|---|---|---|
| public hit@3 (268) | 0.873 | 0.918 | **0.940** |
| halves | 0.896 / 0.851 | 0.940 / 0.896 | 0.940 / 0.940 |
| Wikipedia (108) | 0.991 | 1.000 | 1.000 |
| **non-Wikipedia (160)** | 0.794 | 0.863 | **0.900** |
| CV held-out non-Wikipedia, 3x5 folds | – | 0.865 ± 0.042 | **0.893 ± 0.028** |
| CV held-out all, 3x5 folds | 0.868 | 0.920 | 0.937 ± 0.016 |
| hard (BM25 alone misses, 43) | 0.293 | 0.535 | 0.628 |
| typos (1 in 5 long words garbled) | 0.866 | 0.899 | 0.937 |
| extra 59 (own questions on other docs) | 0.932 | 0.949 | 0.949 |
| non-Wikipedia hit@10 | – | 0.950 | 0.944 |
| size (main.py + index/) | 34.8 MB | 45.8 MB | 51.26 MB |

- v4 fixes 7 of v2's misses (q0048, q0058, q0078, q0090, q0182, q0210, q0253) and breaks one (q0079). Misses: 22 → 16, all book chapters.
- Weights tuned on all public questions, or for non-Wikipedia accuracy only, come out equal to the shipped 0.4 / 0.1 / 0.5 / 0.2.
- The 16 remaining misses are still mostly the wrong book: 38 of 48 top-3 slots.

Official `local_check.py` on `cuba_v4.zip` in the folder `.venv`, two runs: 51.26 MB, load 0.86–0.88 s, p95 17.1–21.0 ms, hit@3 0.940, all PASS. The zip is 38.7 MB.

Platform estimate, not yet verified there:
- hit@3 about 0.89–0.90 if the hidden set is like the non-Wikipedia questions (v1: 0.794 local, 0.793 platform; v2 would be about 0.86).
- p95 about 25–35 ms. v1 ran 1.3–1.6x slower on the platform than locally.
- Size is the tightest limit: 51.26 of 53.0 MB.

## Reproduce

Export and passage encoding need torch + transformers + onnx (the miniconda env; the encoding wants a GPU, about 1 min). Everything else runs in the folder's `.venv` (numpy, tokenizers, PyStemmer, onnxruntime).

```
# 0. model (once)
huggingface-cli download thenlper/gte-small --local-dir .models/gte-small
# 1. query encoder -> _onnx/gte_int8_pc.onnx (int8 per-channel, 34.0 MB)
C:\Users\user\miniconda3\python.exe export_onnx.py
# 2. passage vectors on GPU -> _onnx/pv_gte.npy
C:\Users\user\miniconda3\python.exe encode_passages.py
# 3. index/ (BM25 + PCA-256 int8 vectors + encoder + tokenizer + config)
.venv\Scripts\python.exe build_index.py
# 4. validation
.venv\Scripts\python.exe evaluate.py . --cv
# 5. zip main.py + index/ at the top level, then run the official check on the zip
#    (not on this folder: .models/ and .venv/ would count toward the size)
.venv\Scripts\python.exe -c "import zipfile,os; z=zipfile.ZipFile(r'C:\Users\user\Downloads\LMML_upload\Cuba\cuba_v4.zip','w',zipfile.ZIP_DEFLATED); z.write('main.py'); [z.write(os.path.join('index',f),'index/'+f) for f in os.listdir('index')]; z.close()"
.venv\Scripts\python.exe ..\task_cuba\local_check.py C:\Users\user\Downloads\LMML_upload\Cuba\cuba_v4.zip
```

Another encoder: pass `--model`, `--pool`, `--prefix`, `--tokenizer` to the three scripts (bge-small and arctic need `--pool cls` and the query prefix in `build_index.py`).

## Files

| file | what |
|---|---|
| `main.py` | the submission: `Retriever(index_dir)` and `search(question, k)` |
| `index/` | `lex.npz` (BM25 postings), `dense.npz` (int8 passage vectors, their scales, the PCA matrix), `encoder.onnx` (gte-small int8), `qtokenizer.json`, `terms.txt`, `words.txt` (speller), `docs.json` (ids, duplicates grouped), `config.json` (weights, pooling, query prefix) |
| `export_onnx.py` | exports the encoder to ONNX and quantizes it int8 per-channel |
| `encode_passages.py` | corpus loading and chunking (shared with the build), GPU passage encoding |
| `build_index.py` | builds `index/` |
| `evaluate.py` | public / halves / wiki / non-wiki / extra / hard / typos / CV / latency for any submission folder |
| `extra_questions.jsonl` | 59 own questions on documents the public set does not cover |
| `local_check_output.txt` | official check output for v4, with v2 and v3 for comparison |
| `submission.zip` | v1, the version the platform scored 0.793 |
| `_exp/` | v4 experiment scripts, logs and caches; `results.md` has every table; `v3/` holds the numpy encoder (not for git) |
| `_onnx/` | build inputs, regenerable (not for git) |
| `.models/` | downloaded models, build-time only (not for git) |

Uploads are in `C:\Users\user\Downloads\LMML_upload\Cuba\`:

| zip | encoder | needs | public / non-wiki | p95 here |
|---|---|---|---|---|
| `cuba_v4.zip` | gte-small, 12 layers | onnxruntime | 0.940 / 0.900 | 17–21 ms |
| `cuba_v2.zip` | arctic-xs, 6 layers | onnxruntime | 0.918 / 0.863 | 17 ms |
| `cuba_v3.zip` | arctic-xs in numpy | numpy only | 0.914 / 0.856 | 40–51 ms |
