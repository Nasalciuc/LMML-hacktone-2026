# SIGMOID_CONVOLUTION

Romania, Shaken, Still Standing (Hard). The evaluator needs at least 92.5 % accuracy on `sealed.csv`.
Generic models on all 48 channels top out around 0.86 CV (gradient boosting 0.857, stacking 0.861).
EDA shows three regimes. In quiet records (heavy channels shrunk 4x) the label is a quadratic in the 12 Gaussian channels. In loud records with c13 <= -0.59 it is a parity of c35/c43/c38, and above that gate it is linear.
Each segment gets its own soft vote: SVM + quadratic logistic (0.937), boosting + forest (0.973), logistic + SVM + MLP (0.927). A router sends every record to its segment.
Routed CV 0.952 on train; about 0.937 expected on the sealed mix, which has more quiet and linear records.
Files: `solution.py` (full run), `shaken_still_standing.ipynb` (EDA, scores, evaluator output), `predictions.csv`, `scores.json`.
