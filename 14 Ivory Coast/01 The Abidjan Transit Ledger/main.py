"""Reads test-input.json, writes predictions.json (both in the working dir).

checkpoints/recovered_model.joblib is a full sklearn pipeline (tf-idf + MLP)
that takes raw text, so no separate vectorizer is loaded.

    python main.py
    python main.py --input train-input.json --output preds.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

RECOVERED_MODEL_PATH = Path(__file__).parent / "checkpoints" / "recovered_model.joblib"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test-input.json", help="filename only, no path")
    parser.add_argument("--output", default="predictions.json", help="filename only, no path")
    parser.add_argument("--model", default=str(RECOVERED_MODEL_PATH))
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"input file not found: {input_path}", file=sys.stderr)
        return 2
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"model checkpoint not found: {model_path}", file=sys.stderr)
        return 2

    model = joblib.load(model_path)
    with open(input_path, encoding="utf-8") as fh:
        records = json.load(fh)
    labels = model.predict([r["text"] for r in records])
    predictions = [{"id": r["id"], "label": str(l)} for r, l in zip(records, labels)]
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(predictions, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {len(predictions)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
