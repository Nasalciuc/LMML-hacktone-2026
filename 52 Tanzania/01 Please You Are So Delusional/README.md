# Tanzania - Please, You Are So Delusional!

**Flag:** `SIGMOID_A69EE28C5B2214A9AD99`

The starter encoded documents with MiniLM and questions with e5 (two different vector spaces), cut documents into blind 180-char slices, and dropped the title, which is the only place that says CURRENT ACCESS NOTICE vs ARCHIVED BACKGROUND.
The repaired `retrieval_pipeline.py` ranks whole documents with BM25, scoring title and body as separate fields with the title weighted x3. It needs only numpy (1.26.4); no model downloads.
Every question names the exhibit and city and asks for the current access notice; all 72 answers land on the matching `CURRENT ACCESS NOTICE: <exhibit>, <city>` document (checked against the titles, 0 mismatches).
Run `python retrieval_pipeline.py` here: it writes `retrieved_ids.json` and prints the flag. `corpus.jsonl`, `questions.json` and `flag_utils.py` are the original files.
