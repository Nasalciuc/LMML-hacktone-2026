# local copy of the scoring from the task page
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE.parent / "task_albania"


def load_detector(path):
    spec = importlib.util.spec_from_file_location("detector", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Detector


def run(path, X):
    det = load_detector(path)()
    out = []
    for rec in X.to_dict("records"):
        s = det.update(rec)
        assert np.isfinite(s) and 0.0 <= s <= 1.0, s
        out.append(s)
    return np.array(out)


def score(pred, y, kind):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    rec = {k: float(pred[kind == k].mean()) for k in ("point", "drift", "collective")}
    final = 100 * (0.8 * f1 + 0.05 * rec["point"] + 0.075 * rec["drift"] + 0.075 * rec["collective"])
    return max(1.0, min(100.0, final)), f1, tp, fp, fn, rec


if __name__ == "__main__":
    X = pd.read_csv(DATA / "train-input.csv")
    Y = pd.read_csv(DATA / "train-output.csv")
    s = run(sys.argv[1] if len(sys.argv) > 1 else HERE / "detector.py", X)
    pred = s >= 0.5
    final, f1, tp, fp, fn, rec = score(pred, Y.is_anomaly.values.astype(bool), Y.anomaly_type.values)
    print(f"final {final:.2f} | F1 {f1:.3f} (tp {tp}, fp {fp}, fn {fn}) | recall {rec}")
    for name, g in Y.assign(pred=pred).groupby((Y.anomaly_type != Y.anomaly_type.shift()).cumsum()):
        t = g.anomaly_type.iloc[0]
        if t != "normal":
            first = g.pred.values.argmax() if g.pred.any() else None
            print(f"  {t:10s} rows {g.index[0]}-{g.index[-1]} caught {g.pred.sum()}/{len(g)} first at +{first}")
    fp_idx = np.where(pred & (Y.anomaly_type.values == "normal"))[0]
    print("  false positives at:", fp_idx[:40].tolist())
