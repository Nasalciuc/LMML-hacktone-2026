# writes main.py: one file with the training rows inside, for platforms that take only main.py
import json
from pathlib import Path

import train

texts, labels = train.load()
rows = json.dumps([[t, l] for t, l in zip(texts, labels)], ensure_ascii=False)

code = '''"""Reads test-input.json, writes predictions.json.

Uses checkpoints/recovered_model.joblib when it is there. When it is not (only this file
was uploaded), it rebuilds the same model from the rows below (old-task replay sample +
public set of both tasks) and saves it to that path first. Takes a few seconds.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

CKPT = Path("checkpoints") / "recovered_model.joblib"
ROWS = json.loads(%r)


def build():
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
    ])
    clf = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=600, random_state=0)
    return Pipeline([("tfidf", feats), ("mlp", clf)])


def load_model():
    if CKPT.exists():
        try:
            return joblib.load(CKPT)
        except Exception as exc:
            print(f"checkpoint unreadable ({exc}), rebuilding", file=sys.stderr)
    model = build().fit([t for t, _ in ROWS], np.array([l for _, l in ROWS]))
    try:
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, CKPT)
    except OSError:
        pass
    return model


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "test-input.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "predictions.json"
    records = json.load(open(src, encoding="utf-8"))
    labels = load_model().predict([r["text"] for r in records])
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump([{"id": r["id"], "label": str(l)} for r, l in zip(records, labels)], fh, indent=2, ensure_ascii=False)
    print(f"wrote {len(records)} predictions to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
''' % rows

Path(__file__).with_name("main.py").write_text(code, encoding="utf-8")
print("rows", len(texts))
