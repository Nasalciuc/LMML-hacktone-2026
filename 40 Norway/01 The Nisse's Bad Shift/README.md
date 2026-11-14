# The Nisse's Bad Shift

Submit `test-output.csv` (+ `model.joblib`). `python repair_labels.py` rebuilds both.

- a night = 150 consecutive `sample_id`s (the train bad night is exactly IMG_0450–0599, digits 4↔9, all 36 of them swapped)
- SVM on the pixels (CV acc 98.8%); in each night, if most labels of one digit pair disagree with the model *in the swapped way*, that night is bad and every label of that pair gets swapped back
- check on train using only the dirty labels (out-of-fold model): finds night 3, 4↔9, repair F1 = 1.00
- test: one bad night, IMG_1350–1499, digits **3↔8** (29 of 30 confirmed by the model, the 30th looks like a 1 to it), 30 labels changed
