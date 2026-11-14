"""Rebuild a model that handles both tasks.

The shipped checkpoints were not in our copy of the data, so the model is
retrained from scratch on the old-task replay sample plus the public set
(both tasks). The saved checkpoint is a full pipeline (tf-idf + MLP), so it
takes raw text and needs no separate vectorizer file.

    python train.py            # validate, fit on everything, save checkpoint
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

import model_utils as mu

HERE = Path(__file__).parent
DATA = HERE / "dataset"
OUT = HERE / "checkpoints" / "recovered_model.joblib"
SEED = 0

ENTITY = re.compile(r"\b[A-Z][a-z]+\b|\b(truck|van)-\d+\b|\d+|\b(cocoa|textiles|produce|cashew|electronics)\b")


def load():
    old = mu.load_records(DATA / "old_task_sample.json")
    tin = mu.load_records(DATA / "train-input.json")
    tout = {r["id"]: r["label"] for r in mu.load_records(DATA / "train-output.json")}
    texts = [r["text"] for r in old] + [r["text"] for r in tin]
    labels = [r["label"] for r in old] + [tout[r["id"]] for r in tin]
    return texts, labels


def build():
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
    ])
    clf = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=600, random_state=SEED)
    return Pipeline([("tfidf", feats), ("mlp", clf)])


def per_task(y_true, y_pred):
    gold = {i: y for i, y in enumerate(y_true)}
    pred = {i: y for i, y in enumerate(y_pred)}
    return mu.accuracy_by_task(gold, pred)


def main():
    texts, labels = load()
    y = np.array(labels)
    # group by sentence template (entities masked) so a fold never sees the
    # same phrasing with a different name - closer to unseen test wording
    groups = [ENTITY.sub("X", t).lower() for t in texts]
    _, gid = np.unique(groups, return_inverse=True)
    print(f"{len(texts)} rows, {gid.max() + 1} templates")

    pred = np.empty(len(texts), dtype=object)
    for tr, te in GroupKFold(n_splits=5).split(texts, y, gid):
        m = build().fit([texts[i] for i in tr], y[tr])
        pred[te] = m.predict([texts[i] for i in te])
    print("leave-template-out CV:", per_task(labels, list(pred)))

    model = build().fit(texts, y)
    mu.save_model(model, OUT)
    tin = mu.load_records(DATA / "train-input.json")
    tout = {r["id"]: r["label"] for r in mu.load_records(DATA / "train-output.json")}
    p = model.predict([r["text"] for r in tin])
    print("public set (seen in training):", mu.accuracy_by_task(tout, {r["id"]: l for r, l in zip(tin, p)}))
    print("saved", OUT)


if __name__ == "__main__":
    main()
