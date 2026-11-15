"""Parametric heuristic agent with threat and distance terms."""

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
    "thr": 0.0,        # reduction of the enemy's best capture rate
    "dist": 0.0,       # distance from own start corner (normalised)
    "lam_e": 0.0,      # extra rest threshold per unit of (4.5 - energy)/4.5
    "eg_plies": 0,     # exact alpha-beta over the last this-many plies (0 = off)
    "eg_k": 6,         # moves kept per node in that search
}

P = dict(DEFAULT)
if os.environ.get("AGENT_PARAMS"):
    P.update(json.loads(os.environ["AGENT_PARAMS"]))


def set_params(d):
    P.clear()
    P.update(DEFAULT)
    P.update(d)


mem = {}
_sim = None


def _endgame(s, me, O, W, C, k):
    """Minimax over the remaining plies on the remembered board."""
    global _sim
    if _sim is None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import numpy as np
        import sim as _s
        _sim = (_s, np)
    sm, np = _sim
    own = np.array(O, dtype=np.int64)
    w = np.array(W, dtype=np.int64)
    c = np.array(C, dtype=np.int64)
    en = np.array(s["energy"], dtype=np.int64)
    sc = np.array(s["scores"], dtype=np.int64)
    a, _ = sm.endgame_best(own, w, c, en, sc, me, s["actions_remaining"], k, sm.NBR, sm.NBN)
    a = int(a)
    if a < 0 or a not in s["legal_actions"]:
        return None
    return a


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
    ph = 1.0 - m_left / 128.0
    Q = {}
    for k, v in P.items():
        lk = k + "_L"
        Q[k] = v + (P[lk] - v) * ph if lk in P else v
    legal = s["legal_actions"]
    moves = [a for a in legal if a is not None]
    if not moves:
        return None
    if P["eg_plies"] and s["actions_remaining"] <= P["eg_plies"]:
        return _endgame(s, me, O, W, C, int(P["eg_k"]))
    if m_left <= 1:
        return max(moves, key=lambda a: W[a] * (2 if O[a] == op else 1))

    endgame = m_left <= Q["end_h"]
    pexp = (1 - Q["opp_e"]) + Q["opp_e"] * min(Eop, 9) / 9.0
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
            return W[j] / (C[j] + Q["t_neu"])
        if o == -3:
            return 5.0 * 0.8 / (2.5 + Q["t_neu"])
        if o == op and j not in STARTS:
            return Q["cap_g"] * 2 * W[j] / (C[j] + 2 + Q["t_cap"])
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
            num = Q["cap_g"] * 2 * v
            den = C[a] + 2 + Q["t_cap"]
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
            num += Q["dfn"] * d
            if n_en > 0:
                num -= Q["exp_c"] * pexp * v
        else:
            num = v
            den = C[a] + Q["t_neu"]
            if n_en > 0:
                num -= Q["exp_n"] * pexp * v
                num += Q["cont"] * v
        num += Q["look"] * la + Q["hid"] * hid
        return num / den, (C[a] + 2 if cap else C[a])

    home = 0 if me == 0 else NN - 1
    hr, hc = divmod(home, N)

    def orate(t):
        cost = C[t] + 2
        r = 2.0 * W[t] / (cost + 3.0)
        if cost > Eop + 3:
            r *= 0.5
        return r

    exposed = []
    if Q["thr"]:
        for t in range(NN):
            if O[t] == me and t not in STARTS:
                n = 0
                for k in NB[t]:
                    if O[k] == op:
                        n += 1
                if n:
                    exposed.append((orate(t), t, n))
        exposed.sort(reverse=True)
    thr0 = exposed[0][0] if exposed else 0.0

    def extra(a):
        x = 0.0
        if Q["thr"]:
            cap = O[a] == op
            gone = set()
            if cap:
                for t in NB[a]:
                    if O[t] == me and t not in STARTS:
                        n = 0
                        for k in NB[t]:
                            if O[k] == op:
                                n += 1
                        if n == 1:
                            gone.add(t)
            after = 0.0
            for r, t, n in exposed:
                if t not in gone:
                    after = r
                    break
            for k in NB[a]:
                if O[k] == op:
                    ra = orate(a)
                    if ra > after:
                        after = ra
                    break
            x += Q["thr"] * (thr0 - after)
        if Q["dist"]:
            r, c = divmod(a, N)
            x += Q["dist"] * (abs(r - hr) + abs(c - hc)) / 28.0
        return x

    best_a, best_m = None, -1e9
    for a in moves:
        m, _ = merit_full(a)
        m += extra(a)
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
        if tgt_m > best_m * (1 + Q["save"]) and tgt_m > 0:
            return None
        if best_m < Q["lam"] + Q["lam_e"] * (4.5 - E) / 4.5:
            return None
    return best_a
