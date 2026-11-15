"""Rate-based heuristic agent with map memory (v1)."""

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

P = {
    "cap_mult": float(os.environ.get("CAP_MULT", "1.0")),
    "look": float(os.environ.get("LOOK", "0.3")),
    "hid_v": 5.0, "hid_c": 2.5, "hid_open": 0.8,
}

mem = {}


def reset_agent():
    mem.clear()


def choose_action(s):
    me = s["seat"]
    op = 1 - me
    if not mem:
        mem["w"] = [None] * NN
        mem["c"] = [None] * NN
        mem["own"] = [-3] * NN
    W, C, O = mem["w"], mem["c"], mem["own"]
    vis = s["visible"]
    for i in range(NN):
        if vis[i]:
            O[i] = s["owners"][i]
            W[i] = s["weights"][i]
            C[i] = s["costs"][i]
    E = s["energy"][me]
    left = s["actions_remaining"] // 2  # my remaining choices incl. this one
    legal = s["legal_actions"]

    def merit_of(i):
        o = O[i]
        if o == -3:
            return P["hid_open"] * P["hid_v"] / (3 + P["hid_c"])
        if o == -2 or o == me:
            return 0.0
        if o == op:
            if i in (0, 224):
                return 0.0
            return P["cap_mult"] * 2 * W[i] / (5 + C[i])
        return W[i] / (3 + C[i])

    best = None
    best_m = -1
    for a in legal:
        if a is None:
            continue
        m = merit_of(a)
        # lookahead: best neighbor that becomes reachable
        la = 0.0
        for j in NB[a]:
            if O[j] in (-1, -3, op):
                reach = any(O[k] == me for k in NB[j])
                if not reach:
                    la = max(la, merit_of(j))
        m += P["look"] * la
        if m > best_m:
            best_m, best = m, a
    # also consider targets not yet affordable
    target_m = best_m
    target_cost = None
    for i in range(NN):
        if O[i] in (-1, op) and any(O[k] == me for k in NB[i]):
            m = merit_of(i)
            if m > target_m + 1e-9:
                target_m = m
                target_cost = C[i] + (2 if O[i] == op else 0)
    if best is None:
        return None
    if target_cost is not None and target_cost > E and left > 1:
        # rest to afford a better target unless energy would be wasted
        if E + 3 <= 9 + 0:
            return None
    return best
