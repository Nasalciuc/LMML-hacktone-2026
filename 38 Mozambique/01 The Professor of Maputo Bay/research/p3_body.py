"""Mozambique - The Professor of Maputo Bay. Submission: this file only (stdlib, PyPy 3.10).

Rollout planner. A rule-based base policy gives every boat a goal (fetch materials, bring them
to the workshop, wait there to craft and load, deliver a product, serve a raw contract) and
plays one action per round. Each round we try every alternative goal for the idle boats, play
the rest of the game with the base policy on a local copy of the rules, and keep the best.
A delivery in round t is worth the expected score margin it brings, from survival tables of
the Professor's delivery round (unopposed Professor games, per contract slot and distance
from its home). No wall-clock dependence except a safety net.
"""
import json
import sys
import time

T, F, P = 0, 1, 2
CAP = 3
ROUNDS = 80
INF = 10 ** 6

PR = {
    'rho': 1.0,         # weight of denying the Professor a contract it would otherwise take
    'rawv': 1.0,        # multiplier on raw contract value
    'gam': 0.995,       # mild per-round discount (breaks ties towards acting sooner)
    'shift': 0,         # rounds subtracted from the Professor's delivery rounds (safety margin)
    'detour': 2,        # extra rounds a loaded boat may spend to serve a raw contract on its way
    'rmin': 1.0,        # minimum value of a raw contract worth a dedicated trip (base policy)
    'ahead': 2,         # batches (incl. the workshop) the base policy keeps supplied
    'topk': 5,          # goals per boat combined when two boats are idle
    'margin': 0.05,     # a different goal must beat the current plan by this much
    'hard': 8.5,        # seconds per game after which we stop simulating (limit is 10)
}

# op codes
WAIT, SAIL, LOAD, UNLOAD, GATHER, BUILD, CRAFT, DELIVER = range(8)
# boat goals of the base policy
G_NONE, G_FETCH, G_TOM, G_DELIV, G_RAW, G_WAITM = range(6)
NOGOAL = (G_NONE, 0, 0)

#@@TABLES@@


class Static(object):
    def __init__(self, obs):
        m = obs['map']
        self.routes = routes = m['routes']
        self.n = n = len(routes)
        self.roles = m['roles']
        self.contracts = m['contracts']
        self.nc = len(self.contracts)
        self.home = obs['own']['home']
        self.seat = obs['seat']
        homes = m.get('homes') or [self.home, self.home]
        self.ohome = homes[1 - self.seat] if len(homes) > 1 else self.home
        self.adj = [[w for w in range(n) if routes[v][w]] for v in range(n)]
        dist = [[INF] * n for _ in range(n)]
        nxt = [[-1] * n for _ in range(n)]
        for s in range(n):
            best = {s: (0, 0)}
            first = {s: -1}
            done = set()
            while True:
                cur, cb = None, None
                for v, c in best.items():
                    if v not in done and (cb is None or c < cb):
                        cur, cb = v, c
                if cur is None:
                    break
                done.add(cur)
                for w in self.adj[cur]:
                    if w in done:
                        continue
                    cand = (cb[0] + routes[cur][w], cb[1] + 1)
                    if w not in best or cand < best[w]:
                        best[w] = cand
                        first[w] = w if cur == s else first[cur]
            for v, (ct, ch) in best.items():
                dist[s][v] = ct
                nxt[s][v] = first[v]
        self.dist, self.nxt = dist, nxt
        self.tsites = [s for s in range(n) if self.roles[s] == T]
        self.fsites = [s for s in range(n) if self.roles[s] == F]
        self.msites = [s for s in range(n) if self.roles[s] == 2]
        self.natural = [s for s in range(n) if self.roles[s] in (T, F)]
        self.cdest = [c['destination'] for c in self.contracts]
        self.cgood = [c['good'] for c in self.contracts]
        self.cqty = [c['quantity'] for c in self.contracts]
        self.cpts = [c['points'] for c in self.contracts]
        self.cdl = [c['deadline'] for c in self.contracts]
        self.manuf = [c for c in range(self.nc) if self.cgood[c] == P]
        self.raws = [c for c in range(self.nc) if self.cgood[c] != P]
        self.dest_contracts = [[c for c in range(self.nc) if self.cdest[c] == s] for s in range(n)]
        self.surv = [self.professor(c) for c in range(self.nc)]

    def professor(self, c):
        """P(the Professor has not delivered contract c before round t), t = 0..ROUNDS."""
        oh = self.ohome
        d = self.dist
        if self.cgood[c] == P:
            slots = [x for x in self.manuf]
            k = 4 + min(3, slots.index(c))
            f = min([d[oh][m] + d[m][self.cdest[c]] for m in self.msites] or [MHI])
            f = max(MLO, min(MHI, f)) - MLO
        else:
            k = min(3, self.raws.index(c))
            f = max(RLO, min(RHI, d[oh][self.cdest[c]])) - RLO
        row = SURV[k][f] if k < len(SURV) and f < len(SURV[k]) else [1000] * 81
        sh = PR['shift']
        out = []
        for t in range(ROUNDS + 2):
            tt = min(80, max(0, t + sh))
            out.append(row[tt] / 1000.0)
        return out


def pick_workshop(S):
    best, bs = None, None
    for s in S.msites:
        sc = 2.0 * S.dist[S.home][s]
        for c in S.manuf:
            sc += 0.4 * S.dist[s][S.cdest[c]]
        nt = min([S.dist[s][x] for x in S.tsites] or [9])
        nf = min([S.dist[s][x] for x in S.fsites] or [9])
        sc += 1.0 * nt + 0.6 * nf
        if bs is None or sc < bs:
            best, bs = s, sc
    return best


class Planner(object):
    def __init__(self, S, M, nb, r0, claims=None):
        self.S = S
        self.M = M
        self.nb = nb
        self.r0 = r0
        n = S.n
        self.WH = 4 + 6 * nb
        self.JB = self.WH + 3 * n
        self.ST = self.JB + n
        self.size = self.ST + n
        # value of delivering contract c in round t, seen from round r0
        rho = PR['rho']
        gam = PR['gam']
        self.Wt = Wt = []
        for c in range(S.nc):
            sv = S.surv[c]
            s0 = max(0.05, sv[min(r0, ROUNDS)])
            sinf = min(sv[ROUNDS], s0) / s0
            pts = S.cpts[c] * (PR['rawv'] if S.cgood[c] != P else 1.0)
            dl = S.cdl[c]
            row = []
            for t in range(ROUNDS + 1):
                if t >= dl or t >= ROUNDS or t < r0:
                    row.append(0.0)
                    continue
                st = min(sv[t], s0) / s0
                w = pts * ((1.0 + rho) * st - rho * min(sinf, st)) * gam ** (t - r0)
                row.append(max(w, 0.02 * pts))
            Wt.append(row)

    # ------------------------------------------------------------ encoding
    def encode(self, obs, mem):
        S = self.S
        own = obs['own']
        L = [0] * self.size
        L[0] = obs['round']
        L[1] = 0.0
        wsm = 0
        for s, x in enumerate(own['workshops']):
            if x:
                wsm |= 1 << s
        L[2] = wsm
        cm = 0
        for c, x in enumerate(obs['claims']):
            if x != -1:
                cm |= 1 << c
        L[3] = cm
        for b, bt in enumerate(own['boats'][:self.nb]):
            i = 4 + 6 * b
            L[i] = bt['position']
            L[i + 1] = bt['destination']
            L[i + 2] = bt['eta']
            L[i + 3], L[i + 4], L[i + 5] = bt['cargo']
        WH = self.WH
        for s, w in enumerate(own['warehouses']):
            L[WH + 3 * s], L[WH + 3 * s + 1], L[WH + 3 * s + 2] = w
        for s, j in enumerate(own['jobs']):
            L[self.JB + s] = j
        for s in range(S.n):
            if S.roles[s] in (T, F):
                L[self.ST + s] = est_stock(mem, s, L[0])
        return L

    def apply(self, L, a):
        S = self.S
        L = L[:]
        op, b, x, q = a
        i = 4 + 6 * b
        WH, JB, ST = self.WH, self.JB, self.ST
        r = L[0]
        if op == SAIL:
            L[i + 1] = x
            L[i + 2] = S.routes[L[i]][x]
        elif op == LOAD:
            L[WH + 3 * L[i] + x] -= q
            L[i + 3 + x] += q
        elif op == UNLOAD:
            L[WH + 3 * L[i] + x] += q
            L[i + 3 + x] -= q
        elif op == GATHER:
            L[ST + L[i]] -= q
            L[i + 3 + x] += q
        elif op == BUILD:
            w = WH + 3 * L[i]
            L[w] -= 2
            L[w + 1] -= 1
            L[2] |= 1 << L[i]
        elif op == CRAFT:
            w = WH + 3 * L[i]
            L[w] -= 2
            L[w + 1] -= 1
            L[JB + L[i]] = 3
        elif op == DELIVER:
            c = x
            L[i + 3 + S.cgood[c]] -= S.cqty[c]
            L[3] |= 1 << c
            L[1] += self.Wt[c][r]
        # tick
        for b2 in range(self.nb):
            j = 4 + 6 * b2
            if L[j + 2]:
                L[j + 2] -= 1
                if L[j + 2] == 0:
                    L[j] = L[j + 1]
                    L[j + 1] = -1
        for s in S.msites:
            if L[JB + s]:
                L[JB + s] -= 1
                if L[JB + s] == 0:
                    L[WH + 3 * s + 2] += 1
        r += 1
        L[0] = r
        if r % 4 == 0:
            for s in S.natural:
                if L[ST + s] < 6:
                    L[ST + s] += 1
        return L

    # ------------------------------------------------------------ base policy
    def policy(self, L, G, force=-1):
        """One round of the rule-based policy. G holds one goal per boat and is updated in place.
        Returns the action of this round."""
        S = self.S
        nb = self.nb
        WH, JB, ST = self.WH, self.JB, self.ST
        M = self.M
        r = L[0]
        closed = L[3]
        Wt = self.Wt
        dist = S.dist
        dM = dist[M]
        nxt = S.nxt
        cdest, cdl, cgood, cqty = S.cdest, S.cdl, S.cgood, S.cqty
        home = S.home
        wM = WH + 3 * M
        wH = WH + 3 * home
        built = (L[2] >> M) & 1
        job = L[JB + M]
        nopen = 0
        nnew = 0
        for c in S.manuf:
            if not (closed >> c) & 1:
                t = r + dM[cdest[c]] + 1
                if t < cdl[c]:
                    nopen += 1
                    if t + 4 < cdl[c]:
                        nnew += 1
        pb = 0
        tB = 0
        fB = 0
        taken = 0
        waiters = 0
        for b in range(nb):
            i = 4 + 6 * b
            pb += L[i + 5]
            g = G[b]
            k = g[0]
            if k == G_DELIV or k == G_RAW:
                taken |= 1 << g[1]
            else:
                tB += L[i + 3]
                fB += L[i + 4]
                if k == G_FETCH:
                    src = g[1]
                    av = L[wH + g[2]] if src == home else L[ST + src]
                    q = CAP - L[i + 3] - L[i + 4] - L[i + 5]
                    if av < q:
                        q = av
                    if g[2] == T:
                        tB += q
                    else:
                        fB += q
                elif k == G_WAITM:
                    waiters += 1
        nprod = pb + L[wM + 2] + (1 if job else 0)
        need_new = nopen - nprod
        if need_new > nnew:
            need_new = nnew
        if need_new < 0:
            need_new = 0
        nb_ = need_new + (0 if built or need_new == 0 else 1)
        if nb_ > PR['ahead']:
            nb_ = PR['ahead']
        dT = 2 * nb_ - L[wM] - tB
        dF = nb_ - L[wM + 1] - fB
        bestp = -1.0
        besta = None
        bestb = -1
        bestg = None
        for b in range(nb):
            i = 4 + 6 * b
            if L[i + 2]:
                continue
            p = L[i]
            cT = L[i + 3]
            cF = L[i + 4]
            cP = L[i + 5]
            free = CAP - cT - cF - cP
            g = G[b]
            kind = g[0]
            pr = -1.0
            act = None
            ng = g
            # deliver here
            if cP or cT >= 2 or cF >= 2:
                bv = 0.0
                bc = -1
                for c in S.dest_contracts[p]:
                    if not (closed >> c) & 1 and r < cdl[c] and L[i + 3 + cgood[c]] >= cqty[c]:
                        v = Wt[c][r]
                        if v > bv:
                            bv = v
                            bc = c
                if bc >= 0:
                    pr = 100.0 + bv
                    act = (DELIVER, b, bc, 0)
                    if (kind == G_DELIV or kind == G_RAW) and g[1] == bc:
                        ng = NOGOAL
            if act is None and p == M:
                if not built:
                    if need_new > 0 and L[wM] >= 2 and L[wM + 1] >= 1:
                        pr = 91.0
                        act = (BUILD, b, 0, 0)
                elif job == 0 and need_new > 0 and L[wM] >= 2 and L[wM + 1] >= 1:
                    pr = 90.0
                    act = (CRAFT, b, 0, 0)
                if act is None and kind != G_DELIV and kind != G_RAW:
                    if cT:
                        pr = 70.0
                        act = (UNLOAD, b, T, cT)
                    elif cF:
                        pr = 70.0
                        act = (UNLOAD, b, F, cF)
                    elif L[wM + 2] and free and nopen - pb > 0:
                        q = L[wM + 2]
                        if free < q:
                            q = free
                        if nopen - pb < q:
                            q = nopen - pb
                        pr = 80.0
                        act = (LOAD, b, P, q)
                        ng = NOGOAL
            if act is None:
                # goal clean-up
                if kind == G_TOM:
                    if p == M or not (cT or cF):
                        kind = G_NONE
                elif kind == G_DELIV:
                    c = g[1]
                    if (closed >> c) & 1 or L[i + 3 + cgood[c]] < cqty[c] or r + dist[p][cdest[c]] >= cdl[c]:
                        kind = G_NONE
                elif kind == G_RAW:
                    c = g[1]
                    if (closed >> c) & 1 or r + dist[p][cdest[c]] >= cdl[c]:
                        kind = G_NONE
                    elif L[i + 3 + cgood[c]] >= cqty[c]:
                        kind = G_DELIV
                        g = (G_DELIV, c, 0)
                elif kind == G_FETCH:
                    src = g[1]
                    if free <= 0:
                        kind = G_NONE
                    elif p == src:
                        av = L[wH + g[2]] if src == home else L[ST + src]
                        if av <= 0:
                            kind = G_NONE
                elif kind == G_WAITM:
                    if p == M and not (job or L[wM + 2] or (need_new > 0 and L[wM] >= 2 and L[wM + 1] >= 1)):
                        kind = G_NONE
                        waiters -= 1
                if kind == G_NONE:
                    # ---- greedy goal for a boat without one
                    g = NOGOAL
                    if cP:
                        bv = 0.0
                        for c in S.manuf:
                            if not (closed >> c) & 1 and not (taken >> c) & 1:
                                arr = r + dist[p][cdest[c]]
                                if arr < cdl[c]:
                                    v = Wt[c][arr]
                                    if v > bv:
                                        bv = v
                                        g = (G_DELIV, c, 0)
                    if g is NOGOAL and (cT >= 2 or cF >= 2):
                        bv = 0.0
                        for c in S.raws:
                            if not (closed >> c) & 1 and not (taken >> c) & 1 and L[i + 3 + cgood[c]] >= cqty[c]:
                                d = cdest[c]
                                arr = r + dist[p][d]
                                if arr < cdl[c] and (need_new <= 0 or dist[p][d] + dM[d] - dM[p] <= PR['detour']):
                                    v = Wt[c][arr]
                                    if v > bv:
                                        bv = v
                                        g = (G_DELIV, c, 0)
                    if g is NOGOAL and (cT or cF) and (need_new > 0 or job or L[wM] or L[wM + 1]) and p != M:
                        g = (G_TOM, 0, 0)
                    if g is NOGOAL and free > 0 and not cP:
                        pend = L[wM + 2] + (1 if job else 0)
                        if nopen - pb > 0 and pend > waiters:
                            g = (G_WAITM, 0, 0)
                        elif need_new > 0 and waiters == 0 and L[wM] >= 2 and L[wM + 1] >= 1 and (built or p == M or dT <= 0):
                            g = (G_WAITM, 0, 0)
                        if g is NOGOAL and need_new > 0 and (dT > 0 or dF > 0):
                            order = (T, F) if (dT > 0 and (dF <= 0 or dT >= 2 * dF)) else (F, T)
                            for gd in order:
                                dd = dT if gd == T else dF
                                if dd <= 0:
                                    continue
                                bs = 1e9
                                bsrc = -1
                                av = L[wH + gd]
                                if av > 0 and home != M:
                                    q = av if av < free else free
                                    if dd < q:
                                        q = dd
                                    sc = (dist[p][home] + dM[home] + 4.0) / q
                                    if sc < bs:
                                        bs = sc
                                        bsrc = home
                                for s in (S.tsites if gd == T else S.fsites):
                                    av = L[ST + s]
                                    if av > 0:
                                        q = av if av < free else free
                                        if dd < q:
                                            q = dd
                                        sc = (dist[p][s] + dM[s] + 4.0) / q
                                        if sc < bs:
                                            bs = sc
                                            bsrc = s
                                if bsrc >= 0:
                                    g = (G_FETCH, bsrc, gd)
                                    av = L[wH + gd] if bsrc == home else L[ST + bsrc]
                                    q = av if av < free else free
                                    if gd == T:
                                        dT -= q
                                    else:
                                        dF -= q
                                    break
                        if g is NOGOAL:
                            bs = 0.0
                            for c in S.raws:
                                if (closed >> c) & 1 or (taken >> c) & 1:
                                    continue
                                gd = cgood[c]
                                q = cqty[c]
                                if free < q:
                                    continue
                                d = cdest[c]
                                if L[wH + gd] >= q:
                                    arr = r + dist[p][home] + 1 + dist[home][d]
                                    if arr < cdl[c]:
                                        v = Wt[c][arr]
                                        if v >= PR['rmin'] and v / (arr - r + 2.0) > bs:
                                            bs = v / (arr - r + 2.0)
                                            g = (G_RAW, c, home)
                                for s in (S.tsites if gd == T else S.fsites):
                                    if L[ST + s] >= q:
                                        arr = r + dist[p][s] + 1 + dist[s][d]
                                        if arr < cdl[c]:
                                            v = Wt[c][arr]
                                            if v >= PR['rmin'] and v / (arr - r + 2.0) > bs:
                                                bs = v / (arr - r + 2.0)
                                                g = (G_RAW, c, s)
                        if g is NOGOAL and p != M and need_new > 0 and waiters == 0 and (L[wM] or L[wM + 1] or job):
                            g = (G_WAITM, 0, 0)
                    kind = g[0]
                    if kind == G_DELIV or kind == G_RAW:
                        taken |= 1 << g[1]
                    elif kind == G_WAITM:
                        waiters += 1
                G[b] = g
                ng = g
                # ---- the goal's next step
                if kind == G_FETCH:
                    src = g[1]
                    gd = g[2]
                    if p != src:
                        pr = 50.0 + dist[p][src]
                        act = (SAIL, b, nxt[p][src], 0)
                    else:
                        if src == home:
                            q = L[wH + gd]
                            if q > free:
                                q = free
                            dd = (dT if gd == T else dF) + q
                            if q > dd > 0:
                                q = dd
                            act = (LOAD, b, gd, q)
                        else:
                            q = L[ST + src]
                            if q > free:
                                q = free
                            act = (GATHER, b, gd, q)
                        pr = 60.0
                        ng = (G_TOM, 0, 0)
                elif kind == G_TOM:
                    if p == home and free > 0:
                        for g2 in (T, F):
                            av = L[wH + g2]
                            dd = dT if g2 == T else dF
                            if av > 0 and dd > 0:
                                q = av if av < free else free
                                if dd < q:
                                    q = dd
                                pr = 60.0
                                act = (LOAD, b, g2, q)
                                break
                    if act is None and p != M:
                        pr = 52.0 + dM[p]
                        act = (SAIL, b, nxt[p][M], 0)
                elif kind == G_DELIV:
                    d = cdest[g[1]]
                    if p != d:
                        pr = 56.0 + dist[p][d]
                        act = (SAIL, b, nxt[p][d], 0)
                elif kind == G_RAW:
                    c = g[1]
                    src = g[2]
                    gd = cgood[c]
                    if p != src:
                        pr = 50.0 + dist[p][src]
                        act = (SAIL, b, nxt[p][src], 0)
                    else:
                        q = cqty[c] - L[i + 3 + gd]
                        av = L[wH + gd] if src == home else L[ST + src]
                        if 0 < q <= free and av >= q:
                            pr = 60.0
                            act = (LOAD if src == home else GATHER, b, gd, q)
                            ng = (G_DELIV, c, 0)
                        else:
                            G[b] = NOGOAL
                elif kind == G_WAITM:
                    if p != M:
                        pr = 50.0 + dM[p]
                        act = (SAIL, b, nxt[p][M], 0)
            if act is not None:
                if b == force:
                    pr += 1000.0
                if pr > bestp:
                    bestp = pr
                    besta = act
                    bestb = b
                    bestg = ng
        if besta is None:
            return (WAIT, 0, 0, 0)
        G[bestb] = bestg
        return besta

    def rollout(self, L, G, force=-1):
        """Play the rest of the game with the base policy. Returns (value, first action, goals after it)."""
        G = list(G)
        a0 = self.policy(L, G, force)
        G1 = list(G)
        app = self.apply
        pol = self.policy
        L = app(L, a0)
        while L[0] < ROUNDS:
            L = app(L, pol(L, G))
        self.nroll += 1
        return L[1], a0, G1

    def options(self, L, G, b):
        """Goals worth trying for boat b (idle at a site)."""
        S = self.S
        WH, ST = self.WH, self.ST
        i = 4 + 6 * b
        r = L[0]
        p = L[i]
        closed = L[3]
        dist = S.dist
        home = S.home
        wH = WH + 3 * home
        cT, cF, cP = L[i + 3], L[i + 4], L[i + 5]
        free = CAP - cT - cF - cP
        out = []
        for c in range(S.nc):
            if (closed >> c) & 1:
                continue
            gd = S.cgood[c]
            d = S.cdest[c]
            q = S.cqty[c]
            if L[i + 3 + gd] >= q:
                if r + dist[p][d] < S.cdl[c]:
                    out.append((G_DELIV, c, 0))
            elif gd != P and free >= q - L[i + 3 + gd]:
                if L[wH + gd] >= q and r + dist[p][home] + 1 + dist[home][d] < S.cdl[c]:
                    out.append((G_RAW, c, home))
                cand = []
                for s in (S.tsites if gd == T else S.fsites):
                    if L[ST + s] >= q:
                        t = dist[p][s] + 1 + dist[s][d]
                        if r + t < S.cdl[c]:
                            cand.append((t, s))
                cand.sort()
                for t, s in cand[:2]:
                    out.append((G_RAW, c, s))
        if cT or cF:
            out.append((G_TOM, 0, 0))
        if free > 0:
            if home != self.M:
                for gd in (T, F):
                    if L[wH + gd] > 0:
                        out.append((G_FETCH, home, gd))
            for s in S.natural:
                if L[ST + s] > 0:
                    out.append((G_FETCH, s, S.roles[s]))
        out.append((G_WAITM, 0, 0))
        return out

    def choose(self, L0, G0):
        """Best (value, action, goals) over changes of goal of the idle boats and of who acts this round."""
        nb = self.nb
        best = self.rollout(L0, G0)
        bv = best[0] + PR['margin']
        idle = [b for b in range(nb) if L0[4 + 6 * b + 2] == 0]
        two = len(idle) >= 2
        ranked = []
        for b in idle:
            lst = [(best[0], G0[b])]
            if two:
                res = self.rollout(L0, G0, b)
                if res[0] > bv:
                    best = res
                    bv = res[0]
            for g in self.options(L0, G0, b):
                if g == G0[b]:
                    continue
                G = list(G0)
                G[b] = g
                top = -1e9
                for f in ((-1, b) if two else (-1,)):
                    res = self.rollout(L0, G, f)
                    if res[0] > top:
                        top = res[0]
                    if res[0] > bv:
                        best = res
                        bv = res[0]
                lst.append((top, g))
            lst.sort(key=lambda x: -x[0])
            ranked.append(lst[:PR['topk']])
        if two:
            b0, b1 = idle[0], idle[1]
            for v0, g0 in ranked[0]:
                for v1, g1 in ranked[1]:
                    if g0 == G0[b0] or g1 == G0[b1]:
                        continue
                    G = list(G0)
                    G[b0] = g0
                    G[b1] = g1
                    for f in (-1, b0, b1):
                        res = self.rollout(L0, G, f)
                        if res[0] > bv:
                            best = res
                            bv = res[0]
        return best


def est_stock(mem, site, rnd):
    seen = mem['seen'].get(site)
    if seen is None:
        r0, v0 = 0, 4
    else:
        r0, v0 = seen
    regen = rnd // 4 - r0 // 4
    if regen < 0:
        regen = 0
    v = v0 + regen
    return 6 if v > 6 else v


def to_json(a):
    op, b, x, q = a
    if op == WAIT:
        return {'op': 'wait'}
    if op == SAIL:
        return {'op': 'sail', 'boat': b, 'destination': x}
    if op == LOAD:
        return {'op': 'load', 'boat': b, 'good': x, 'quantity': q}
    if op == UNLOAD:
        return {'op': 'unload', 'boat': b, 'good': x, 'quantity': q}
    if op == GATHER:
        return {'op': 'gather', 'boat': b, 'good': x, 'quantity': q}
    if op == BUILD:
        return {'op': 'build_workshop', 'boat': b}
    if op == CRAFT:
        return {'op': 'craft', 'boat': b}
    return {'op': 'deliver', 'boat': b, 'contract': x}


def agent(observation, memory):
    legal = observation['legal_actions']
    try:
        act = decide(observation, memory)
        if act is not None and act in legal:
            return act
        memory['goals'] = None
        memory['bad'] = memory.get('bad', 0) + 1
    except Exception as exc:  # never crash
        print('agent error', repr(exc), file=sys.stderr)
        memory['goals'] = None
        memory['bad'] = memory.get('bad', 0) + 1
    return {'op': 'wait'} if {'op': 'wait'} in legal else legal[0]


def decide(obs, memory):
    t0 = time.perf_counter()
    if 'S' not in memory:
        S = Static(obs)
        memory['S'] = S
        memory['seen'] = {}
        memory['M'] = pick_workshop(S)
        memory['goals'] = None
        memory['used'] = 0.0
        memory['exp'] = 0
    S = memory['S']
    rnd = obs['round']
    for s, v in enumerate(obs['stocks']):
        if v is not None and S.roles[s] in (T, F):
            memory['seen'][s] = (rnd, v)
    nb = len(obs['own']['boats'])
    G0 = memory.get('goals') or []
    G0 = [tuple(g) for g in G0[:nb]] + [NOGOAL] * (nb - len(G0))
    sites = None
    for s, x in enumerate(obs['own']['workshops']):
        if x:
            sites = [s]
    if sites is None:
        sites = [memory['M']] if memory.get('msel') or len(S.msites) < 2 else S.msites
    best = None
    for m in sites:
        pl = Planner(S, m, nb, rnd)
        pl.nroll = 0
        L0 = pl.encode(obs, memory)
        if memory['used'] >= PR['hard']:
            G = list(G0)
            res = (0.0, pl.policy(L0, G), G)
            memory['guard'] = memory.get('guard', 0) + 1
        else:
            res = pl.choose(L0, G0)
        memory['exp'] += pl.nroll
        if best is None or res[0] > best[0][0]:
            best = (res, m)
    memory['M'] = best[1]
    memory['msel'] = True
    memory['goals'] = best[0][2]
    memory['used'] += time.perf_counter() - t0
    return to_json(best[0][1])


def serve():
    protocol = sys.stdout
    sys.stdout = sys.stderr          # print() for debugging goes to stderr, not the protocol
    memory = {}
    for line in sys.stdin:
        observation = json.loads(line)['observation']
        if observation['round'] == 0:
            memory = {}
        protocol.write(json.dumps(agent(observation, memory)) + '\n')
        protocol.flush()


if __name__ == '__main__':
    serve()
