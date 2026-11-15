"""agent2 heuristic (proven ~61%) + pure-Python exact endgame minimax.

The endgame search runs only in the last few plies, where few cells are both
affordable and adjacent to a frontier, so branching is small. A node budget
bounds time; on overflow it falls back to the heuristic move. No numba.
"""

import json
import os

N = 15
NN = N * N
NB = []
for _i in range(NN):
    _r, _c = divmod(_i, N)
    _l = []
    if _r > 0: _l.append(_i - N)
    if _r < N - 1: _l.append(_i + N)
    if _c > 0: _l.append(_i - 1)
    if _c < N - 1: _l.append(_i + 1)
    NB.append(tuple(_l))
STARTS = (0, NN - 1)

DEFAULT = {
    "cap_g": 1.0, "t_neu": 3.0, "t_cap": 3.0, "look": 0.3, "hid": 0.0,
    "dfn": 0.0, "exp_n": 0.0, "exp_c": 0.0, "cont": 0.0, "lam": 0.0,
    "save": 0.0, "end_h": 0.0, "opp_e": 0.0,
    "eg_plies": 0,       # run exact endgame when actions_remaining <= this
    "eg_k": 6,           # max candidate moves kept per node
    "eg_budget": 60000,  # node budget; overflow -> heuristic fallback
}

P = dict(DEFAULT)
if os.environ.get("AGENT_PARAMS"):
    P.update(json.loads(os.environ["AGENT_PARAMS"]))


def set_params(d):
    P.clear()
    P.update(DEFAULT)
    P.update(d)


mem = {}


def reset_agent():
    mem.clear()


def _touch(O, i, p):
    for k in NB[i]:
        if O[k] == p:
            return True
    return False


def _cands(O, W, C, E, p, K):
    q = 1 - p
    scored = []
    for i in range(NN):
        o = O[i]
        if o == -1:
            cost = C[i]
            g = W[i]
        elif o == q and i not in STARTS:
            cost = C[i] + 2
            g = 2 * W[i]
        else:
            continue
        if cost > E or not _touch(O, i, p):
            continue
        scored.append((g * 16 - cost, i))
    scored.sort(reverse=True)
    out = [i for _, i in scored[:K]]
    out.append(-1)  # rest
    return out


class _Budget:
    __slots__ = ("n",)

    def __init__(self):
        self.n = 0


def _eg(O, W, C, en, sc, p, plies, me, K, alpha, beta, bud):
    if plies == 0 or bud.n > P["eg_budget"]:
        return sc[me] - sc[1 - me]
    bud.n += 1
    q = 1 - p
    maximize = (p == me)
    best = -10 ** 9 if maximize else 10 ** 9
    for a in _cands(O, W, C, en[p], p, K):
        if a < 0:
            pe = en[p]
            en[p] = min(9, pe + 3)
            v = _eg(O, W, C, en, sc, q, plies - 1, me, K, alpha, beta, bud)
            en[p] = pe
        else:
            o = O[a]
            if o == -1:
                en[p] -= C[a]; sc[p] += W[a]; O[a] = p
                v = _eg(O, W, C, en, sc, q, plies - 1, me, K, alpha, beta, bud)
                O[a] = -1; sc[p] -= W[a]; en[p] += C[a]
            else:
                en[p] -= C[a] + 2; sc[p] += W[a]; sc[q] -= W[a]; O[a] = p
                v = _eg(O, W, C, en, sc, q, plies - 1, me, K, alpha, beta, bud)
                O[a] = q; sc[q] += W[a]; sc[p] -= W[a]; en[p] += C[a] + 2
        if maximize:
            if v > best:
                best = v
            if best > alpha:
                alpha = best
        else:
            if v < best:
                best = v
            if best < beta:
                beta = best
        if alpha >= beta:
            break
    return best


def _endgame_move(s, me, O, W, C):
    en = list(s["energy"])
    sc = list(s["scores"])
    plies = s["actions_remaining"]
    K = int(P["eg_k"])
    bud = _Budget()
    legal = set(s["legal_actions"])
    best_a, best_v = None, -10 ** 18
    for a in _cands(O, W, C, en[me], me, K):
        cand = None if a < 0 else a
        if cand not in legal:
            continue
        if a < 0:
            pe = en[me]
            en[me] = min(9, pe + 3)
            v = _eg(O, W, C, en, sc, 1 - me, plies - 1, me, K, best_v, 10 ** 9, bud)
            en[me] = pe
        else:
            o = O[a]
            if o == -1:
                en[me] -= C[a]; sc[me] += W[a]; O[a] = me
                v = _eg(O, W, C, en, sc, 1 - me, plies - 1, me, K, best_v, 10 ** 9, bud)
                O[a] = -1; sc[me] -= W[a]; en[me] += C[a]
            else:
                en[me] -= C[a] + 2; sc[me] += W[a]; sc[1 - me] -= W[a]; O[a] = me
                v = _eg(O, W, C, en, sc, 1 - me, plies - 1, me, K, best_v, 10 ** 9, bud)
                O[a] = 1 - me; sc[1 - me] += W[a]; sc[me] -= W[a]; en[me] += C[a] + 2
        if v > best_v:
            best_v, best_a = v, cand
        if bud.n > P["eg_budget"]:
            return "OVERFLOW"
    return best_a


def choose_action(s):
    me = s["seat"]
    op = 1 - me
    if not mem:
        mem["w"] = [0] * NN
        mem["c"] = [0] * NN
        mem["own"] = [-3] * NN
    W, C, O = mem["w"], mem["c"], mem["own"]
    vis = s["visible"]
    sw, sc_, so = s["weights"], s["costs"], s["owners"]
    for i in range(NN):
        if vis[i]:
            O[i] = so[i]
            if so[i] != -2:
                W[i] = sw[i]
                C[i] = sc_[i]
    E = s["energy"][me]
    Eop = s["energy"][op]
    m_left = (s["actions_remaining"] + 1) // 2
    legal = s["legal_actions"]
    moves = [a for a in legal if a is not None]
    if not moves:
        return None

    if P["eg_plies"] and s["actions_remaining"] <= P["eg_plies"]:
        res = _endgame_move(s, me, O, W, C)
        if res != "OVERFLOW":
            return res  # int cell or None to rest

    if m_left <= 1:
        return max(moves, key=lambda a: W[a] * (2 if O[a] == op else 1))

    endgame = m_left <= P["end_h"]
    pexp = (1 - P["opp_e"]) + P["opp_e"] * min(Eop, 9) / 9.0
    if endgame:
        pexp = 0.0

    def reach(j):
        for k in NB[j]:
            if O[k] == me:
                return True
        return False

    def base_merit(j):
        o = O[j]
        if o == -1:
            return W[j] / (C[j] + P["t_neu"])
        if o == -3:
            return 5.0 * 0.8 / (2.5 + P["t_neu"])
        if o == op and j not in STARTS:
            return P["cap_g"] * 2 * W[j] / (C[j] + 2 + P["t_cap"])
        return 0.0

    def merit_full(a):
        o = O[a]
        v = W[a]
        cap = o == op
        n_en = 0
        hid = 0
        la = 0.0
        for j in NB[a]:
            oj = O[j]
            if oj == op:
                n_en += 1
            elif oj == -3:
                hid += 1
            if oj in (-1, -3) and not reach(j):
                bm = base_merit(j)
                if bm > la:
                    la = bm
        if cap:
            num = P["cap_g"] * 2 * v
            den = C[a] + 2 + P["t_cap"]
            d = 0
            for t in NB[a]:
                if O[t] == me and t not in STARTS:
                    alone = True
                    for k in NB[t]:
                        if k != a and O[k] == op:
                            alone = False
                            break
                    if alone:
                        d += W[t]
            num += P["dfn"] * d
            if n_en > 0:
                num -= P["exp_c"] * pexp * v
        else:
            num = v
            den = C[a] + P["t_neu"]
            if n_en > 0:
                num -= P["exp_n"] * pexp * v
                num += P["cont"] * v
        num += P["look"] * la + P["hid"] * hid
        return num / den

    best_a, best_m = None, -1e9
    for a in moves:
        m = merit_full(a)
        if m > best_m:
            best_m, best_a = m, a
    tgt_m = -1e9
    for i in range(NN):
        o = O[i]
        if (o == -1 or (o == op and i not in STARTS)) and reach(i):
            cost = C[i] + (2 if o == op else 0)
            if cost > E:
                m = merit_full(i)
                if m > tgt_m:
                    tgt_m = m
    if E + 3 > 9:
        return best_a if best_m > 0 else None
    if not endgame:
        if tgt_m > best_m * (1 + P["save"]) and tgt_m > 0:
            return None
        if best_m < P["lam"]:
            return None
    return best_a
