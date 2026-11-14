# The Abidjan Transit Ledger

Local: old 1.00 / new 1.00 on train-input.json; 0.92 / 0.93 when whole sentence templates are held out (5-fold). Gates are 0.85 / 0.85. Flag: not yet, comes from the checker.

Our copy of the data had no `checkpoints/` (no original, fine-tuned or vectorizer files), so the model is retrained from scratch on `old_task_sample.json` + the public set, which covers both tasks.
`checkpoints/recovered_model.joblib` is a full sklearn pipeline (word + char tf-idf, MLP with 64 hidden units), so `main.py` feeds it raw text and needs no vectorizer file.
`python train.py` rebuilds it (about 30 s). Needs scikit-learn==1.8.0, numpy==2.5.1, joblib==1.5.3.
`python main.py` reads `test-input.json`, writes `predictions.json`. Same I/O contract as `main_template.py`.
