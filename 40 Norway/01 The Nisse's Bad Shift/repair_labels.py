from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

HERE = Path(__file__).parent
TASK = HERE.parent / "task_norway"
PIX = [f"pixel_{i}" for i in range(64)]
NIGHT = 150  # samples per night; the train bad night is exactly ids 450-599


def model():
    return SVC(C=10, gamma="scale", random_state=0)


def find_bad_nights(df, pred, min_rows=5, share=0.6):
    """Nights where labels of one digit pair disagree with the model in the swapped way."""
    night = df.sample_id.str[4:].astype(int) // NIGHT
    bad = []
    for n in sorted(night.unique()):
        m = (night == n).values
        g, p = df.given_label.values[m], pred[m]
        for a, b in combinations(range(10), 2):
            ab = np.isin(g, [a, b])
            if ab.sum() < min_rows:
                continue
            swapped = ((g == a) & (p == b)) | ((g == b) & (p == a))
            if swapped[ab].mean() >= share:
                bad.append((n, a, b, int(swapped[ab].sum()), int(ab.sum())))
    return bad, night


def repair(df, bad, night):
    fixed = df.given_label.copy()
    for n, a, b, *_ in bad:
        m = night == n
        fixed[m & (df.given_label == a)] = b
        fixed[m & (df.given_label == b)] = a
    return fixed


def repair_f1(given, fixed, truth):
    changed = fixed != given
    good = changed & (fixed == truth)
    wrong = (given != truth).sum()
    p = good.sum() / max(changed.sum(), 1)
    r = good.sum() / max(wrong, 1)
    return 2 * p * r / max(p + r, 1e-9)


if __name__ == "__main__":
    tr = pd.read_csv(TASK / "train-input.csv").merge(pd.read_csv(TASK / "train-output.csv"), on="sample_id")
    te = pd.read_csv(TASK / "test-input.csv")

    # check on train without looking at the true labels: model fit on the given (dirty) labels, out of fold
    oof = cross_val_predict(model(), tr[PIX] / 16, tr.given_label, cv=5)
    bad, night = find_bad_nights(tr, oof)
    fixed = repair(tr, bad, night)
    print("train bad nights (night, a, b, swapped, rows):", bad)
    print("train repair F1:", round(repair_f1(tr.given_label, fixed, tr.corrected_label), 4))

    clf = model().fit(tr[PIX] / 16, tr.corrected_label)
    bad, night = find_bad_nights(te, clf.predict(te[PIX] / 16))
    te["corrected_label"] = repair(te, bad, night)
    print("test bad nights:", bad, "| labels changed:", int((te.corrected_label != te.given_label).sum()))
    te[["sample_id", "corrected_label"]].to_csv(HERE / "test-output.csv", index=False)

    all_x = pd.concat([tr[PIX], te[PIX]]) / 16
    all_y = pd.concat([tr.corrected_label, te.corrected_label])
    joblib.dump(model().fit(all_x, all_y), HERE / "model.joblib")
