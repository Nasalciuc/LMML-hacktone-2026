"""Parametric heuristic agent (tunable weights)."""

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
    "cap_g": 1.0,      # weight on capture swing (2v)
    "t_neu": 3.0,      # turn overhead added to neutral cost
    "t_cap": 3.0,      # turn overhead added to capture cost
    "look": 0.3,       # best newly reachable neighbour merit
    "hid": 0.0,        # per hidden neighbour
    "dfn": 0.0,        # value of own tiles unexposed by a capture
    "exp_n": 0.0,      # penalty * v for neutral claim left exposed
    "exp_c": 0.0,      # penalty * v for captured tile left exposed
    "cont": 0.0,       # bonus * v for neutral tile the enemy could take
    "lam": 0.0,        # minimum merit to act instead of resting
    "save": 0.0,       # rest when an unaffordable target beats best by this ratio
    "end_h": 0.0,      # endgame horizon (own moves): no saving, no exposure penalty
    "opp_e": 0.0,      # exposure penalty scales with enemy energy/9 blend
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


def choose_action(s):
    me = s["seat"]
    op = 1 - me
    if not mem:
        mem["w"] = [0] * NN
        mem["c"] = [0] * NN
        mem["own"] = [-3] * NN
    W, C, O = mem["w"], mem["c"], mem["own"]
    vis = s["visible"]
    sw, sc, so = s["weights"], s["costs"], s["owners"]
    for i in range(NN):
        if vis[i]:
            O[i] = so[i]
            if so[i] != -2:
                W[i] = sw[i]
                C[i] = sc[i]
    E = s["energy"][me]
    Eop = s["energy"][op]
    m_left = (s["actions_remaining"] + 1) // 2
    legal = s["legal_actions"]
    moves = [a for a in legal if a is not None]
    if not moves:
        return None
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
        return num / den, (C[a] + 2 if cap else C[a])

    best_a, best_m = None, -1e9
    for a in moves:
        m, _ = merit_full(a)
        if m > best_m:
            best_m, best_a = m, a
    # targets we cannot afford yet
    tgt_m = -1e9
    for i in range(NN):
        o = O[i]
        if (o == -1 or (o == op and i not in STARTS)) and reach(i):
            cost = C[i] + (2 if o == op else 0)
            if cost > E:
                m, _ = merit_full(i)
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
