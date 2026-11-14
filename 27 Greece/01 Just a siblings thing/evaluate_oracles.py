import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

HERE = Path(__file__).parent
TASK = HERE.parent / "task_greece"
sys.path.insert(0, str(TASK))
from model import FEATURE_COLUMNS, MythicThreatClassifier  # noqa: E402

X = pd.read_csv(TASK / "test_dataset.csv")
lab = pd.read_csv(TASK / "test_labels.csv")
assert (X.encounter_id == lab.encounter_id).all()
y = lab.is_catastrophic_threat.values
xt = torch.tensor(X[FEATURE_COLUMNS].values, dtype=torch.float32)

rows, probs = [], {}
for name in ["ares", "athena", "apollo"]:
    m = MythicThreatClassifier()
    m.load_state_dict(torch.load(TASK / "models" / f"model_{name}.pth", weights_only=True, map_location="cpu"))
    m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(xt)).numpy()
    probs[name] = p
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    rows.append({"model": name.upper(), "accuracy": accuracy_score(y, pred), "recall": recall_score(y, pred),
                 "precision": precision_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p),
                 "missed_threats_FN": fn, "false_alarms_FP": fp, "TP": tp, "TN": tn})

res = pd.DataFrame(rows).set_index("model")
print(f"test set: {len(y)} encounters, {y.sum()} catastrophic\n")
print(res.round(4).to_string())
best = res.recall.idxmax()
print(f"\nmissing a threat costs most -> highest recall / fewest FN: {best} -> SIGMOID_{best}")

fig, ax = plt.subplots(1, 4, figsize=(19, 4))
for a, (name, p) in zip(ax[:3], probs.items()):
    cm = confusion_matrix(y, (p >= 0.5).astype(int))
    a.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        a.text(j, i, v, ha="center", va="center", fontsize=14)
    a.set_xticks([0, 1], ["pred 0", "pred 1"]); a.set_yticks([0, 1], ["true 0", "true 1"])
    a.set_title(f"{name.upper()}  acc {res.loc[name.upper(), 'accuracy']:.2f}  recall {res.loc[name.upper(), 'recall']:.2f}")
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y, p)
    ax[3].plot(fpr, tpr, label=f"{name.upper()} (AUC {roc_auc_score(y, p):.2f})")
ax[3].plot([0, 1], [0, 1], ":", color="grey"); ax[3].legend(); ax[3].set_title("ROC")
plt.tight_layout()
plt.savefig(HERE / "oracles.png", dpi=120)
res.round(4).to_csv(HERE / "metrics.csv")
