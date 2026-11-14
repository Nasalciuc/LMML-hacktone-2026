"""Shaken, Still Standing - routed ensemble for the Vrancea verdict.

    python solution.py            # CV scores, predictions.csv, evaluator
    python solution.py --no-eval  # same, without calling the evaluator
"""
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
TASK = HERE.parent / "task_romania"
DATA = TASK / "dataset"
CH = [f"c{j + 1:02d}" for j in range(48)]
PARITY = ["c35", "c43", "c38"]
CV = StratifiedKFold(5, shuffle=True, random_state=0)


def load():
    train = pd.read_csv(DATA / "train.csv")
    sealed = pd.read_csv(DATA / "sealed.csv")
    return train, sealed


# ---------------------------------------------------------------- structure
class Router:
    """quiet records -> sphere-like rule on the 12 gaussian channels;
    loud records with c13 below the gate -> 3-channel parity; above -> linear rule."""

    def __init__(self, quiet_cut=0.55, gate=None):
        self.quiet_cut = quiet_cut
        self.gate = gate

    def fit(self, X_all):
        k = stats.kurtosis(X_all)
        self.gauss = np.where(np.abs(k) < 0.4)[0]
        self.heavy = np.where(np.abs(k) >= 0.4)[0]
        self.mu, self.sd = X_all.mean(0), X_all.std(0)
        return self

    def loudness(self, X):
        Z = (X - self.mu) / self.sd
        return np.sqrt((Z[:, self.heavy] ** 2).mean(1))

    def segment(self, X):
        quiet = self.loudness(X) < self.quiet_cut
        high = X[:, CH.index("c13")] > self.gate
        seg = np.where(quiet, "quiet", np.where(high, "linear", "parity"))
        return seg

    def find_gate(self, X, y, grid=None):
        """Scan c13 for the split where a parity model (below) and a linear model (above) fit best."""
        loud = self.loudness(X) >= self.quiet_cut
        Xl, yl = X[loud], y[loud]
        c13 = Xl[:, CH.index("c13")]
        ip = [CH.index(c) for c in PARITY]
        grid = np.quantile(c13, np.linspace(0.5, 0.9, 41)) if grid is None else grid
        lo, hi = c13 < np.quantile(c13, 0.4), c13 > np.quantile(c13, 0.9)
        par = HistGradientBoostingClassifier(max_iter=200, max_depth=4).fit(Xl[lo][:, ip], yl[lo])
        lin = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)).fit(Xl[hi], yl[hi])
        ok_p = par.predict(Xl[:, ip]) == yl
        ok_l = lin.predict(Xl) == yl
        score = [ok_p[c13 <= g].sum() + ok_l[c13 > g].sum() for g in grid]
        self.gate = float(grid[int(np.argmax(score))])
        self.gate_scan = (grid, np.array(score) / len(yl))
        return self.gate


# ---------------------------------------------------------------- models
def generic_models():
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(C=3.0, probability=True, random_state=0)),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(25)),
        "random_forest": RandomForestClassifier(500, n_jobs=-1, random_state=0),
        "grad_boost": HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, random_state=0),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier((64, 32), alpha=1.0, max_iter=2000, random_state=0)),
    }


def expert_models():
    """Candidate models per segment; the ones in VOTE are averaged."""
    return {
        "quiet": {
            "svm_rbf": make_pipeline(StandardScaler(), SVC(C=100, gamma=0.005, probability=True, random_state=0)),
            "quad_logistic": make_pipeline(StandardScaler(), PolynomialFeatures(2), StandardScaler(),
                                           LogisticRegression(C=0.3, max_iter=5000)),
            "mlp": make_pipeline(StandardScaler(), MLPClassifier((128, 64), alpha=1.0, max_iter=3000, random_state=0)),
            "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
            "grad_boost": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0),
        },
        "parity": {
            "grad_boost": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, random_state=0),
            "random_forest": RandomForestClassifier(500, min_samples_leaf=5, n_jobs=-1, random_state=0),
            "svm_rbf": make_pipeline(StandardScaler(), SVC(C=30, gamma=0.5, probability=True, random_state=0)),
            "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
            "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        },
        "linear": {
            "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)),
            "svm_rbf": make_pipeline(StandardScaler(), SVC(C=10, gamma=0.002, probability=True, random_state=0)),
            "mlp": make_pipeline(StandardScaler(), MLPClassifier((16,), alpha=3.0, max_iter=3000, random_state=0)),
            "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(25)),
            "grad_boost": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0),
        },
    }


VOTE = {
    "quiet": ["svm_rbf", "quad_logistic"],
    "parity": ["grad_boost", "random_forest"],
    "linear": ["logistic", "svm_rbf", "mlp"],
}


def columns(seg, router):
    if seg == "quiet":
        return list(router.gauss)
    if seg == "parity":
        return [CH.index(c) for c in PARITY]
    return list(range(48))


def oof_proba(model, X, y):
    return cross_val_predict(model, X, y, cv=CV, method="predict_proba", n_jobs=5)[:, 1]


def evaluate_generic(X, y):
    scores, P = {}, {}
    for name, m in generic_models().items():
        P[name] = oof_proba(m, X, y)
        scores[name] = float(((P[name] > 0.5) == y).mean())
    P["soft_vote"] = np.mean([P[k] for k in generic_models()], 0)
    scores["soft_vote"] = float(((P["soft_vote"] > 0.5) == y).mean())
    stack = StackingClassifier(list(generic_models().items()),
                               final_estimator=LogisticRegression(max_iter=5000), cv=5, n_jobs=1)
    P["stacking"] = oof_proba(stack, X, y)
    scores["stacking"] = float(((P["stacking"] > 0.5) == y).mean())
    return scores, P


def evaluate_experts(X, y, seg, router):
    """Per-segment CV of every candidate and of the vote; returns scores and routed OOF probabilities."""
    scores = {}
    routed = np.zeros(len(y))
    for s, cands in expert_models().items():
        m = seg == s
        Xs, ys = X[m][:, columns(s, router)], y[m]
        P = {n: oof_proba(mod, Xs, ys) for n, mod in cands.items()}
        sc = {n: float(((p > 0.5) == ys).mean()) for n, p in P.items()}
        vote = np.mean([P[k] for k in VOTE[s]], 0)
        sc["vote"] = float(((vote > 0.5) == ys).mean())
        sc["n_train"] = int(m.sum())
        scores[s] = sc
        routed[m] = vote
    return scores, routed


def fit_predict(X, y, seg, Xn, seg_n, router):
    proba = np.zeros(len(Xn))
    for s, cands in expert_models().items():
        m, mn = seg == s, seg_n == s
        cols = columns(s, router)
        ps = [clone(cands[k]).fit(X[m][:, cols], y[m]).predict_proba(Xn[mn][:, cols])[:, 1] for k in VOTE[s]]
        proba[mn] = np.mean(ps, 0)
    return proba


def run_evaluator(pred_path):
    out = subprocess.run([sys.executable, str(TASK / "evaluation_script.py"), str(pred_path)],
                         capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


def main(call_eval=True):
    train, sealed = load()
    X, y = train[CH].to_numpy(), train.vrancea.to_numpy()
    Xn = sealed[CH].to_numpy()

    router = Router().fit(np.vstack([X, Xn]))
    router.find_gate(X, y)
    seg, seg_n = router.segment(X), router.segment(Xn)
    print(f"gate c13 > {router.gate:.3f}")
    comp = pd.DataFrame({"train": pd.Series(seg).value_counts(normalize=True),
                         "sealed": pd.Series(seg_n).value_counts(normalize=True)}).round(3)
    print(comp)

    gen, _ = evaluate_generic(X, y)
    print("\ngeneric models, 5-fold CV accuracy (all 48 channels):")
    for k, v in gen.items():
        print(f"  {k:15s} {v:.4f}")

    exp, routed = evaluate_experts(X, y, seg, router)
    print("\nper-segment CV accuracy:")
    for s, sc in exp.items():
        print(f"  {s}: " + ", ".join(f"{k}={v:.4f}" if k != "n_train" else f"n={v}" for k, v in sc.items()))
    acc = float(((routed > 0.5) == y).mean())
    w = comp["sealed"]
    est = float(sum(w[s] * exp[s]["vote"] for s in exp))
    print(f"\nrouted ensemble CV accuracy (train mix): {acc:.4f}")
    print(f"expected accuracy on the sealed mix:     {est:.4f}")

    proba = fit_predict(X, y, seg, Xn, seg_n, router)
    pred = (proba > 0.5).astype(int)
    pred_path = HERE / "predictions.csv"
    pd.DataFrame({"record_id": sealed.record_id, "vrancea": pred}).to_csv(pred_path, index=False)
    print("wrote predictions.csv", np.bincount(pred).tolist())

    result = {"gate": router.gate, "composition": comp.to_dict(), "generic": gen, "experts": exp,
              "routed_cv": acc, "expected_sealed": est}
    if call_eval:
        result["evaluator"] = run_evaluator(pred_path)
        print("\nevaluator:", result["evaluator"])
    (HERE / "scores.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main(call_eval="--no-eval" not in sys.argv)
