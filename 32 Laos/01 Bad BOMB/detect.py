"""Pick the 28 defective shells.

  python prep.py ../task_laos/images prep.npz
  python detect.py prep.npz test-output.txt

1. Residual of every aligned shell after removing the first 8 principal
   components (lighting / pose); score = peak of the blurred residual.
2. The top-ranked shells are grouped by the shape of their residual maps
   (cosine similarity). Groups of 3+ are defect modes (hole near the band,
   dent with streak, rust streak near the nose).
3. Each mode's mean residual is used as a template; every shell matching a
   template is flagged. The list is filled up to 28 by the peak score.
"""
import sys

import numpy as np

import score_pca as sp

N_OUT = 28
K_PCA = 8
TOP_SEED = 40
SEED_SIM = 0.5
MATCH_SIM = 0.6
GAP_LO = 0.4


def main(prep_path, out_path):
    d = np.load(prep_path)
    names = list(d["names"])
    mask = sp.interior_mask(d["masks"])
    V = sp.normalise(d["crops"], mask)
    R = sp.pca_residual(V.reshape(len(V), -1), K_PCA)
    score, _ = sp.peak_scores(R, mask, d["crops"].shape[1:3])
    Rn = R / np.linalg.norm(R, axis=1, keepdims=True)

    top = list(np.argsort(-score)[:TOP_SEED])
    S = Rn[top] @ Rn[top].T
    # connected components of the seed graph
    comp = -np.ones(len(top), int)
    c = 0
    for a in range(len(top)):
        if comp[a] >= 0:
            continue
        stack = [a]
        comp[a] = c
        while stack:
            u = stack.pop()
            for v in np.where(S[u] > SEED_SIM)[0]:
                if comp[v] < 0:
                    comp[v] = c
                    stack.append(v)
        c += 1
    flagged = {}
    for k in range(c):
        members = [top[i] for i in np.where(comp == k)[0]]
        if len(members) < 3:
            continue
        T = Rn[members].mean(0)
        T /= np.linalg.norm(T)
        sim = Rn @ T
        # a real defect mode is an isolated cluster: almost nothing sits in the
        # similarity band just below the match threshold. Shared lighting /
        # highlight patterns fade out gradually into the normal population.
        n_hit = int((sim > MATCH_SIM).sum())
        n_band = int(((sim > GAP_LO) & (sim <= MATCH_SIM)).sum())
        print(f"mode {k}: {n_hit} above {MATCH_SIM}, {n_band} in ({GAP_LO}, {MATCH_SIM}]")
        if n_band > 0.25 * n_hit:
            continue
        hits = np.where(sim > MATCH_SIM)[0]
        hits = set(hits.tolist()) | set(members)
        print(f"mode {k}: {len(members)} seeds, {len(hits)} matches:",
              " ".join(sorted(names[i][6:10] for i in hits)))
        for i in hits:
            flagged[i] = max(flagged.get(i, 0), float(sim[i]))
    chosen = sorted(flagged, key=lambda i: -flagged[i])[:N_OUT]
    for i in np.argsort(-score):
        if len(chosen) >= N_OUT:
            break
        if i not in chosen:
            chosen.append(int(i))
            print("fill:", names[i])
    chosen = sorted(names[i] for i in chosen)
    with open(out_path, "w", newline="\n") as f:
        f.write("\n".join(chosen) + "\n")
    print(len(chosen), "written to", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
