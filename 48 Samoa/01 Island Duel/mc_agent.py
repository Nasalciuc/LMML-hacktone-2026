"""Monte Carlo agent: heuristic rollouts over sampled hidden maps."""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim  # noqa: E402

NN = 225


def _f(k, d):
    return float(os.environ.get(k, d))


CFG = {
    "n_det": int(_f("N_DET", 16)),
    "k_cand": int(_f("K_CAND", 8)),
    "win_bonus": _f("WIN_BONUS", 10),
    "blk_p": _f("BLK_P", 0.2),
    "prm_me": np.array([_f("ME_CAP", 1.0), _f("ME_LOOK", 0.3)]),
    "prm_op": np.array([_f("OP_CAP", 1.0), _f("OP_LOOK", 0.3)]),
}

M = {}


def reset_agent():
    M.clear()


def _init():
    M["kw"] = np.zeros(NN, dtype=np.int64)
    M["kc"] = np.zeros(NN, dtype=np.int64)
    M["known"] = np.zeros(NN, dtype=np.bool_)
    M["last"] = np.full(NN, -3, dtype=np.int64)
    M["t"] = 0


def choose_action(s):
    if not M:
        _init()
    me = s["seat"]
    vis = np.array(s["visible"], dtype=np.bool_)
    own = np.array(s["owners"], dtype=np.int64)
    kw, kc, known, last = M["kw"], M["kc"], M["known"], M["last"]
    for i in np.nonzero(vis)[0]:
        known[i] = True
        last[i] = own[i]
        if own[i] != -2:
            kw[i] = s["weights"][i]
            kc[i] = s["costs"][i]
    legal = s["legal_actions"]
    moves = [a for a in legal if a is not None]
    if not moves:
        return None
    M["t"] += 1
    w = kw
    c = kc
    op = 1 - me
    prm = CFG["prm_me"]

    def score(a):
        o = own[a]
        if o == -1:
            m = w[a] / (3.0 + c[a])
        else:
            m = prm[0] * 2.0 * w[a] / (5.0 + c[a])
        return m

    moves.sort(key=score, reverse=True)
    cands = moves[: CFG["k_cand"]]
    cands = np.array(cands + [-1], dtype=np.int64)
    energy = np.array(s["energy"], dtype=np.int64)
    scores = np.array(s["scores"], dtype=np.int64)
    vals = sim.evaluate_actions(
        cands, kw, kc, known, own, vis, last, me, energy, scores,
        s["actions_remaining"], CFG["n_det"], 7919 * M["t"] + 13, CFG["blk_p"],
        CFG["prm_me"], CFG["prm_op"], CFG["win_bonus"], sim.NBR, sim.NBN)
    b = int(cands[int(np.argmax(vals))])
    return None if b < 0 else b
