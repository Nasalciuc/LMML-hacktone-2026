"""Forward model of Island Duel plus rollout policy, compiled with numba."""

import numpy as np
from numba import njit

N = 15
NN = 225
BLOCK = -2
NEUT = -1

_nb = np.full((NN, 4), -1, dtype=np.int64)
_nbn = np.zeros(NN, dtype=np.int64)
for _i in range(NN):
    _r, _c = divmod(_i, N)
    _k = 0
    for _j, _ok in ((_i - N, _r > 0), (_i + N, _r < N - 1), (_i - 1, _c > 0), (_i + 1, _c < N - 1)):
        if _ok:
            _nb[_i, _k] = _j
            _k += 1
    _nbn[_i] = _k
NBR = _nb
NBN = _nbn


@njit(cache=True)
def xorshift(s):
    s ^= (s << np.uint64(13)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    s ^= s >> np.uint64(7)
    s ^= (s << np.uint64(17)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return s


@njit(cache=True)
def touches(own, i, p, nbr, nbn):
    for k in range(nbn[i]):
        if own[nbr[i, k]] == p:
            return True
    return False


@njit(cache=True)
def merit(own, w, c, i, p, capm):
    o = own[i]
    if o == NEUT:
        return w[i] / (3.0 + c[i])
    if o == 1 - p:
        if i == 0 or i == 224:
            return 0.0
        return capm * 2.0 * w[i] / (5.0 + c[i])
    return 0.0


@njit(cache=True)
def policy(own, w, c, energy, p, moves_left, prm, nbr, nbn):
    """Heuristic move for player p. Returns cell or -1 for rest."""
    capm = prm[0]
    look = prm[1]
    q = 1 - p
    E = energy[p]
    best_a = -1
    best_m = -1.0
    tgt_m = -1.0
    last = moves_left <= 1
    for i in range(NN):
        o = own[i]
        if o == p or o == BLOCK:
            continue
        if o == q and (i == 0 or i == 224):
            continue
        if not touches(own, i, p, nbr, nbn):
            continue
        if o == NEUT:
            cost = c[i]
        else:
            cost = c[i] + 2
        if last:
            m = w[i] * (1.0 if o == NEUT else 2.0)
        else:
            m = merit(own, w, c, i, p, capm)
            la = 0.0
            for k in range(nbn[i]):
                j = nbr[i, k]
                oj = own[j]
                if oj == p or oj == BLOCK:
                    continue
                if touches(own, j, p, nbr, nbn):
                    continue
                mj = merit(own, w, c, j, p, capm)
                if mj > la:
                    la = mj
            m += look * la
        if cost <= E:
            if m > best_m:
                best_m = m
                best_a = i
        elif m > tgt_m:
            tgt_m = m
    if best_a < 0:
        return -1
    if (not last) and tgt_m > best_m and E + 3 <= 9:
        return -1
    return best_a


@njit(cache=True)
def apply(own, w, c, energy, scores, p, a):
    if a < 0:
        e = energy[p] + 3
        energy[p] = 9 if e > 9 else e
        return
    o = own[a]
    if o == NEUT:
        energy[p] -= c[a]
        scores[p] += w[a]
    else:
        energy[p] -= c[a] + 2
        scores[p] += w[a]
        scores[1 - p] -= w[a]
    own[a] = p


@njit(cache=True)
def rollout(own, w, c, energy, scores, to_move, plies_left, prm_me, prm_op, me, nbr, nbn):
    """Play to the end with heuristic policies; returns final margin for me."""
    p = to_move
    while plies_left > 0:
        moves_left = (plies_left + 1) // 2
        prm = prm_me if p == me else prm_op
        a = policy(own, w, c, energy, p, moves_left, prm, nbr, nbn)
        apply(own, w, c, energy, scores, p, a)
        plies_left -= 1
        p = 1 - p
    return scores[me] - scores[1 - me]


@njit(cache=True)
def determinize(kw, kc, known, own_now, vis, last_own, me, op_score, seed,
                blk_p, out_w, out_c, out_own, nbr, nbn):
    """Sample hidden terrain and a plausible hidden enemy region."""
    s = np.uint64(seed) * np.uint64(2654435761) + np.uint64(88172645463325252)
    op = 1 - me
    for i in range(NN):
        if known[i]:
            out_w[i] = kw[i]
            out_c[i] = kc[i]
            if vis[i]:
                out_own[i] = own_now[i]
            else:
                lo = last_own[i]
                if lo == BLOCK:
                    out_own[i] = BLOCK
                elif lo == op:
                    out_own[i] = op
                else:
                    out_own[i] = NEUT
        else:
            s = xorshift(s)
            u = (s >> np.uint64(11)) % np.uint64(1000000)
            if u < blk_p * 1000000:
                out_own[i] = BLOCK
                out_w[i] = 0
                out_c[i] = 0
            else:
                out_own[i] = NEUT
                s = xorshift(s)
                out_w[i] = 1 + np.int64((s >> np.uint64(11)) % np.uint64(9))
                s = xorshift(s)
                out_c[i] = 1 + np.int64((s >> np.uint64(11)) % np.uint64(4))
    start = 224 if op == 1 else 0
    out_own[start] = op
    out_w[start] = 0
    out_c[start] = 0
    have = 0
    for i in range(NN):
        if out_own[i] == op:
            have += out_w[i]
    need = op_score - have
    if need <= 0:
        return
    # BFS outward from enemy territory through cells we cannot currently see
    queue = np.empty(NN, dtype=np.int64)
    seen = np.zeros(NN, dtype=np.bool_)
    qh = 0
    qt = 0
    for i in range(NN):
        if out_own[i] == op:
            queue[qt] = i
            qt += 1
            seen[i] = True
    while qh < qt and need > 0:
        i = queue[qh]
        qh += 1
        # shuffle neighbour order a little
        s = xorshift(s)
        off = np.int64((s >> np.uint64(11)) % np.uint64(4))
        n = nbn[i]
        for kk in range(n):
            j = nbr[i, (kk + off) % n]
            if seen[j]:
                continue
            seen[j] = True
            if vis[j] or out_own[j] != NEUT:
                continue
            if need <= 0:
                break
            out_own[j] = op
            need -= out_w[j]
            queue[qt] = j
            qt += 1


@njit(cache=True)
def evaluate_actions(cands, kw, kc, known, own_now, vis, last_own, me, energy0, scores0,
                     plies_left, n_det, seed0, blk_p, prm_me, prm_op, win_bonus, nbr, nbn):
    nc = cands.shape[0]
    tot = np.zeros(nc)
    w = np.empty(NN, dtype=np.int64)
    c = np.empty(NN, dtype=np.int64)
    own0 = np.empty(NN, dtype=np.int64)
    own = np.empty(NN, dtype=np.int64)
    en = np.empty(2, dtype=np.int64)
    sc = np.empty(2, dtype=np.int64)
    for d in range(n_det):
        determinize(kw, kc, known, own_now, vis, last_own, me, scores0[1 - me], seed0 + d,
                    blk_p, w, c, own0, nbr, nbn)
        for k in range(nc):
            for i in range(NN):
                own[i] = own0[i]
            en[0] = energy0[0]
            en[1] = energy0[1]
            sc[0] = scores0[0]
            sc[1] = scores0[1]
            apply(own, w, c, en, sc, me, cands[k])
            m = rollout(own, w, c, en, sc, 1 - me, plies_left - 1, prm_me, prm_op, me, nbr, nbn)
            if m > 0:
                m += win_bonus
            elif m < 0:
                m -= win_bonus
            tot[k] += m
    return tot / n_det


# ---------------------------------------------------------------------------
# Endgame: exact alpha-beta over the last few plies on the known board.
# Unknown cells are treated as blocked; each side keeps its K best moves by
# immediate swing plus resting.

@njit(cache=True)
def gen_moves(own, w, c, energy, p, K, nbr, nbn, out):
    q = 1 - p
    E = energy[p]
    n = 0
    gains = np.empty(NN, dtype=np.int64)
    cells = np.empty(NN, dtype=np.int64)
    for i in range(NN):
        o = own[i]
        if o == NEUT:
            cost = c[i]
            g = w[i]
        elif o == q and i != 0 and i != 224:
            cost = c[i] + 2
            g = 2 * w[i]
        else:
            continue
        if cost > E:
            continue
        if not touches(own, i, p, nbr, nbn):
            continue
        # score: swing first, cheaper first on ties
        gains[n] = g * 16 - cost
        cells[n] = i
        n += 1
    # partial selection sort for top K
    m = 0
    for k in range(min(K, n)):
        b = k
        for j in range(k + 1, n):
            if gains[j] > gains[b]:
                b = j
        gains[k], gains[b] = gains[b], gains[k]
        cells[k], cells[b] = cells[b], cells[k]
        out[m] = cells[k]
        m += 1
    out[m] = -1  # rest
    m += 1
    return m


@njit(cache=True)
def _ab(own, w, c, energy, scores, p, plies, me, K, alpha, beta, nbr, nbn):
    if plies == 0:
        return scores[me] - scores[1 - me]
    moves = np.empty(K + 1, dtype=np.int64)
    nm = gen_moves(own, w, c, energy, p, K, nbr, nbn, moves)
    maximize = p == me
    best = -1000000 if maximize else 1000000
    for k in range(nm):
        a = moves[k]
        o_prev = own[a] if a >= 0 else 0
        e_prev = energy[p]
        s0 = scores[0]
        s1 = scores[1]
        apply(own, w, c, energy, scores, p, a)
        v = _ab(own, w, c, energy, scores, 1 - p, plies - 1, me, K, alpha, beta, nbr, nbn)
        if a >= 0:
            own[a] = o_prev
        energy[p] = e_prev
        scores[0] = s0
        scores[1] = s1
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


@njit(cache=True)
def endgame_best(own, w, c, energy, scores, me, plies, K, nbr, nbn):
    """Return (best move, value) for `me` to move with `plies` plies left."""
    moves = np.empty(K + 1, dtype=np.int64)
    nm = gen_moves(own, w, c, energy, me, K, nbr, nbn, moves)
    best = -1000000
    best_a = -1
    alpha = -1000000
    for k in range(nm):
        a = moves[k]
        o_prev = own[a] if a >= 0 else 0
        e_prev = energy[me]
        s0 = scores[0]
        s1 = scores[1]
        apply(own, w, c, energy, scores, me, a)
        v = _ab(own, w, c, energy, scores, 1 - me, plies - 1, me, K, alpha, 1000000, nbr, nbn)
        if a >= 0:
            own[a] = o_prev
        energy[me] = e_prev
        scores[0] = s0
        scores[1] = s1
        if v > best:
            best = v
            best_a = a
        if best > alpha:
            alpha = best
    return best_a, best
