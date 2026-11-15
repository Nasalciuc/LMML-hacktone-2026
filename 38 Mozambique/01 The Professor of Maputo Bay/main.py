"""Your submission. Upload only this file; keep it self-contained (standard library only).

The evaluator runs `python main.py`, sends one JSON line {"observation": ...} per
decision on stdin and reads one JSON action per line from stdout. Leave the loop
at the bottom as it is and put your strategy in agent().
"""
import json
import sys
import time

T, F, P = 0, 1, 2
CAP = 3
ROUNDS = 80
INF = 10 ** 6

# Tunable parameters for micro-beam planner
PR = {
    'gamma': 0.975,     # per-round discount of future points
    'lam': 0.85,        # per cost-unit discount of materials away from workshop
    'mu': 0.62,         # discount for materials in non-workshop warehouses
    'kp_boat': 0.97,    # value factor of product already carried by a boat
    'kp_wh': 0.84,      # value factor of product sitting in workshop warehouse
    'kp_job': 0.78,     # value factor of product currently crafting
    'kb': 0.66,         # value factor of raw-material batch potential
    'kr': 0.82,         # value factor of carried raw goods matched to raw contract
    'nu': 0.16,         # small value factor of ungathered natural stock
    'unit': 0.18,       # per-unit bonus for useful T/F
    'build': 7.4,       # value of having workshop built at target M
    'H': 22,            # beam search steps (wait rounds fast-forwarded)
    'W': 72,            # beam width
    'H0': 26,           # deeper opening search at round 0
    'W0': 96,
    'T0': 2.0,
    'budget': 6.8,      # seconds of search per game
}

# op codes
WAIT, SAIL, LOAD, UNLOAD, GATHER, BUILD, CRAFT, DELIVER = range(8)


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
        hops = [[INF] * n for _ in range(n)]
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
                hops[s][v] = ch
                nxt[s][v] = first[v]
        self.dist, self.hops, self.nxt = dist, hops, nxt
        # True travel rounds is dist; add tiny hop tiebreak (0.35*hops) for material routing
        self.cost = [[dist[a][b] + hops[a][b] if a != b else 0 for b in range(n)] for a in range(n)]
        self.tcost2 = [[2 * dist[a][b] + hops[a][b] if a != b else 0 for b in range(n)] for a in range(n)]

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

        # Professor's baseline earliest delivery round per contract (using true dist)
        oh = self.ohome
        mo = min(self.msites, key=lambda s: self.dist[oh][s] + 0.3 * self.hops[oh][s]) if self.msites else oh
        self.mo = mo
        oeta = [ROUNDS] * self.nc
        raw_sorted = sorted(self.raws, key=lambda c: (self.cdl[c], self.dist[oh][self.cdest[c]]))
        for idx, c in enumerate(raw_sorted):
            oeta[c] = 2 + self.dist[oh][self.cdest[c]] + 4 * (idx // 2)
        man_sorted = sorted(self.manuf, key=lambda c: (self.cdl[c], self.dist[mo][self.cdest[c]] if self.msites else 0))
        prev_m = 10 + self.dist[oh][mo]
        for idx, c in enumerate(man_sorted):
            oeta[c] = prev_m + 10 * idx + (self.dist[mo][self.cdest[c]] if self.msites else 0)
        self.oeta = oeta


def workshop_score(S, s):
    sc = 1.6 * (S.dist[S.home][s] + 0.3 * S.hops[S.home][s])
    for c in S.manuf:
        sc += 0.45 * S.dist[s][S.cdest[c]]
    nt = min([S.dist[s][x] + 0.3 * S.hops[s][x] for x in S.tsites] or [9])
    nf = min([S.dist[s][x] + 0.3 * S.hops[s][x] for x in S.fsites] or [9])
    sc += 1.1 * nt + 0.8 * nf
    return sc


def pick_workshop(S):
    best, bs = None, None
    for s in S.msites:
        sc = workshop_score(S, s)
        if bs is None or sc < bs:
            best, bs = s, sc
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


def win_factor(S, c, del_r, oeta):
    """Smooth expected competitive value multiplier (winning denies opponent points)."""
    o = oeta[c]
    diff = del_r - o
    if diff <= -4:
        pw = 0.97
    elif diff < 0:
        pw = 0.90 - 0.02 * (diff + 1)
    elif diff == 0:
        pw = 0.82 if (del_r % 2 == S.seat) else 0.50
    elif diff <= 16:
        pw = max(0.14, 0.72 - 0.04 * diff)
    else:
        pw = 0.12
    return 0.28 + 1.05 * pw


class Planner(object):
    def __init__(self, S, M, oeta, r0):
        self.S = S
        self.M = M
        self.oeta = oeta
        self.r0 = r0
        n = S.n
        # Fixed max 3 boats layout: [r, val, wsm, closed, nb, boat0..2 (6 ints each), wh (3*n), jobs (n), stocks (n)]
        self.NB_IDX = 4
        self.B0 = 5
        self.WH = 5 + 18
        self.JB = self.WH + 3 * n
        self.ST = self.JB + n
        self.size = self.ST + n
        g = PR['gamma']
        self.disc = [g ** max(0, t - r0) for t in range(0, 200)]
        lam = PR['lam']
        self.lamp = [lam ** k for k in range(200)]
        lam2 = lam ** 0.5
        self.lamp2 = [lam2 ** k for k in range(300)]

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
        boats = own['boats']
        nb = min(3, len(boats))
        L[4] = nb
        for b in range(nb):
            bt = boats[b]
            i = self.B0 + 6 * b
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

    def actions(self, L):
        S = self.S
        WH, JB, ST = self.WH, self.JB, self.ST
        r = L[0]
        wsm = L[2]
        closed = L[3]
        nb = L[4]
        acts = []

        has_om = any(not ((closed >> c) & 1) and r < S.cdl[c] for c in S.manuf)
        need_T = has_om or any(not ((closed >> c) & 1) and r < S.cdl[c] and S.cgood[c] == T for c in S.raws)
        need_F = has_om or any(not ((closed >> c) & 1) and r < S.cdl[c] and S.cgood[c] == F for c in S.raws)

        for b in range(nb):
            i = self.B0 + 6 * b
            if L[i + 2]:
                continue
            p = L[i]
            cT, cF, cP = L[i + 3], L[i + 4], L[i + 5]
            free = CAP - cT - cF - cP

            # Deliveries
            for c in S.dest_contracts[p]:
                if not ((closed >> c) & 1) and r < S.cdl[c] and L[i + 3 + S.cgood[c]] >= S.cqty[c]:
                    acts.append((DELIVER, b, c, 0))

            role = S.roles[p]
            w = WH + 3 * p
            if role == 2:
                if not ((wsm >> p) & 1):
                    if has_om and L[w] >= 2 and L[w + 1] >= 1:
                        acts.append((BUILD, b, 0, 0))
                elif has_om and L[JB + p] == 0 and L[w] >= 2 and L[w + 1] >= 1:
                    acts.append((CRAFT, b, 0, 0))
                if cT and has_om:
                    acts.append((UNLOAD, b, T, cT))
                if cF and has_om:
                    acts.append((UNLOAD, b, F, cF))
                if L[w + 2] and free and has_om:
                    acts.append((LOAD, b, P, min(free, L[w + 2])))
            else:
                if free:
                    if need_T:
                        xt = L[w + T]
                        if xt > 0:
                            mt = min(xt, free)
                            acts.append((LOAD, b, T, mt))
                            if mt > 2:
                                acts.append((LOAD, b, T, 2))
                    if need_F:
                        xf = L[w + F]
                        if xf > 0:
                            mf = min(xf, free)
                            acts.append((LOAD, b, F, mf))
                            if mf > 1:
                                acts.append((LOAD, b, F, 1))
                            if mf > 2:
                                acts.append((LOAD, b, F, 2))
                if free and ((role == T and need_T) or (role == F and need_F)):
                    xs = L[ST + p]
                    if xs > 0:
                        ms = min(xs, free)
                        acts.append((GATHER, b, role, ms))
                        if role == T and ms > 2:
                            acts.append((GATHER, b, role, 2))
                        elif role == F:
                            if ms > 1:
                                acts.append((GATHER, b, role, 1))
                            if ms > 2:
                                acts.append((GATHER, b, role, 2))
                # Escape hatch: only unload useless raw cargo if it blocks needed space
                if not need_T and cT > 0 and need_F and free < 2:
                    acts.append((UNLOAD, b, T, cT))
                if not need_F and cF > 0 and need_T and free < 2:
                    acts.append((UNLOAD, b, F, cF))

            for d in S.adj[p]:
                acts.append((SAIL, b, d, 0))

        if not acts or any(L[self.B0 + 6 * b + 2] == 0 and S.roles[L[self.B0 + 6 * b]] == 2 and L[self.JB + L[self.B0 + 6 * b]] > 0 for b in range(nb)):
            acts.append((WAIT, 0, 0, 0))
        return acts

    def apply(self, L, a):
        S = self.S
        L = L[:]
        op, b, x, q = a
        WH, JB, ST = self.WH, self.JB, self.ST
        r = L[0]
        if op == SAIL:
            i = self.B0 + 6 * b
            L[i + 1] = x
            L[i + 2] = S.routes[L[i]][x]
        elif op == LOAD:
            i = self.B0 + 6 * b
            L[WH + 3 * L[i] + x] -= q
            L[i + 3 + x] += q
        elif op == UNLOAD:
            i = self.B0 + 6 * b
            L[WH + 3 * L[i] + x] += q
            L[i + 3 + x] -= q
        elif op == GATHER:
            i = self.B0 + 6 * b
            L[ST + L[i]] -= q
            L[i + 3 + x] += q
        elif op == BUILD:
            i = self.B0 + 6 * b
            w = WH + 3 * L[i]
            L[w] -= 2
            L[w + 1] -= 1
            L[2] |= 1 << L[i]
        elif op == CRAFT:
            i = self.B0 + 6 * b
            w = WH + 3 * L[i]
            L[w] -= 2
            L[w + 1] -= 1
            L[JB + L[i]] = 3
        elif op == DELIVER:
            i = self.B0 + 6 * b
            c = x
            L[i + 3 + S.cgood[c]] -= S.cqty[c]
            L[3] |= 1 << c
            v = S.cpts[c] * self.disc[r] * win_factor(S, c, r, self.oeta)
            L[1] += v

        nb = L[4]
        # Advance 1 round (and fast-forward extra WAIT rounds when all boats are in transit)
        min_eta = INF
        for b2 in range(nb):
            j = self.B0 + 6 * b2
            if L[j + 2]:
                L[j + 2] -= 1
                if L[j + 2] == 0:
                    L[j] = L[j + 1]
                    L[j + 1] = -1
            if L[j + 2] < min_eta:
                min_eta = L[j + 2]

        dt = 1
        if min_eta > 0 and min_eta < INF and r + 1 < ROUNDS:
            extra = min(min_eta, ROUNDS - (r + 1))
            dt += extra
            for b2 in range(nb):
                j = self.B0 + 6 * b2
                L[j + 2] -= extra
                if L[j + 2] == 0:
                    L[j] = L[j + 1]
                    L[j + 1] = -1

        for s in S.msites:
            jb = L[JB + s]
            if jb > 0:
                if jb <= dt:
                    L[JB + s] = 0
                    L[WH + 3 * s + 2] += 1
                else:
                    L[JB + s] = jb - dt

        r_new = r + dt
        L[0] = r_new
        regen = r_new // 4 - r // 4
        if regen > 0:
            for s in S.natural:
                nv = L[ST + s] + regen
                L[ST + s] = 6 if nv > 6 else nv
        return L

    def evaluate(self, L):
        S = self.S
        WH, JB = self.WH, self.JB
        r = L[0]
        closed = L[3]
        nb = L[4]
        disc = self.disc
        val = L[1]

        dist = S.dist
        tcost2 = S.tcost2
        n = S.n
        dmin = [99] * n
        tmin2 = [199] * n
        for b in range(nb):
            i = self.B0 + 6 * b
            if L[i + 2]:
                e = L[i + 2]
                loc = L[i + 1]
            else:
                e = 0
                loc = L[i]
            drow = dist[loc]
            trow = tcost2[loc]
            e2 = 2 * e
            for s in range(n):
                v = e + drow[s]
                if v < dmin[s]:
                    dmin[s] = v
                tv = e2 + trow[s]
                if tv < tmin2[s]:
                    tmin2[s] = tv

        # Manufactured goods in hand, warehouse, or crafting (differentiated kp)
        prods = []
        kp_boat, kp_wh, kp_job = PR['kp_boat'], PR['kp_wh'], PR['kp_job']
        for b in range(nb):
            i = self.B0 + 6 * b
            k = L[i + 5]
            if k:
                loc = L[i] if L[i + 2] == 0 else L[i + 1]
                t = r + L[i + 2]
                for _ in range(k):
                    prods.append((t, -kp_boat, loc, kp_boat))
        for s in S.msites:
            k = L[WH + 3 * s + 2]
            for _ in range(k):
                prods.append((r + 1 + dmin[s], -kp_wh, s, kp_wh))
            if L[JB + s]:
                prods.append((r + max(L[JB + s], dmin[s]) + 1, -kp_job, s, kp_job))

        om = [c for c in S.manuf if not ((closed >> c) & 1) and r < S.cdl[c]]
        if prods and om:
            prods.sort()
            for t, _, loc, kp in prods:
                best, bv = None, 0.0
                for c in om:
                    arr = t + dist[loc][S.cdest[c]]
                    if arr >= S.cdl[c] or arr >= ROUNDS:
                        continue
                    v = 12.0 * disc[arr] * win_factor(S, c, arr, self.oeta)
                    if v > bv:
                        best, bv = c, v
                if best is None:
                    continue
                val += kp * bv
                om.remove(best)
        rm = len(om)

        # Raw goods carried
        kr = PR['kr']
        usedT = [0] * nb
        usedF = [0] * nb
        orw = [c for c in S.raws if not ((closed >> c) & 1) and r < S.cdl[c]]
        if orw:
            for b in range(nb):
                i = self.B0 + 6 * b
                loc = L[i] if L[i + 2] == 0 else L[i + 1]
                t = r + L[i + 2]
                best, bv = None, 0.0
                for c in orw:
                    g = S.cgood[c]
                    if L[i + 3 + g] - (usedT[b] if g == T else usedF[b]) < S.cqty[c]:
                        continue
                    arr = t + dist[loc][S.cdest[c]]
                    if arr >= S.cdl[c] or arr >= ROUNDS:
                        continue
                    v = 5.0 * disc[arr] * win_factor(S, c, arr, self.oeta)
                    if v > bv:
                        best, bv = c, v
                if best is not None and (rm == 0 or bv > 2.8 * disc[r]):
                    val += kr * bv
                    orw.remove(best)
                    if S.cgood[best] == T:
                        usedT[b] += S.cqty[best]
                    else:
                        usedF[b] += S.cqty[best]

        # Workshop potential & materials
        M = self.M
        for s in S.msites:
            if (L[2] >> s) & 1:
                M = s
                break
        built = (L[2] >> M) & 1
        if built:
            val += PR['build']

        if rm > 0 and r < ROUNDS - 6:
            lamp2 = self.lamp2
            mu = PR['mu']
            tl = []
            fl = []
            nu = PR['nu']
            ST = self.ST
            for s in range(n):
                w = WH + 3 * s
                if L[w] or L[w + 1]:
                    f = 1.04 if s == M else mu * lamp2[tcost2[s][M] + tmin2[s]]
                    tl.extend([f] * L[w])
                    fl.extend([f] * L[w + 1])
            for s in S.natural:
                k = L[ST + s]
                if k > 0:
                    f = nu * lamp2[tmin2[s] + 2 + tcost2[s][M]]
                    if S.roles[s] == T:
                        tl.extend([f] * min(k, 3))
                    else:
                        fl.extend([f] * min(k, 3))
            for b in range(nb):
                i = self.B0 + 6 * b
                cT = L[i + 3] - usedT[b]
                cF = L[i + 4] - usedF[b]
                if cT or cF:
                    if L[i + 2]:
                        d2 = 2 * L[i + 2] + tcost2[L[i + 1]][M]
                    else:
                        d2 = tcost2[L[i]][M]
                    f = lamp2[d2]
                    if cT > 0:
                        tl.extend([f] * cT)
                    if cF > 0:
                        fl.extend([f] * cF)
            tl.sort(reverse=True)
            fl.sort(reverse=True)
            if not built:
                val += PR['build'] * (sum(tl[:2]) + sum(fl[:1])) / 3.0 - PR['build'] * 0.45
                tl = tl[2:]
                fl = fl[1:]
            te = sum(tl[:2 * rm])
            fe = sum(fl[:rm])
            batches = min(te / 2.0, fe)
            ev = 0.0
            for c in om:
                arr = min(ROUNDS - 1, r + 9 + dist[M][S.cdest[c]])
                if arr >= S.cdl[c]:
                    ev += 0.25 * disc[r + 8]
                else:
                    ev += disc[arr] * win_factor(S, c, arr, self.oeta)
            bv12 = 12.0 * ev / rm
            tleft = ROUNDS - r
            tf = 1.0 if tleft > 24 else max(0.0, (tleft - 6) / 18.0)
            val += tf * (PR['kb'] * bv12 * batches + PR['unit'] * (te + fe) * disc[r])
        return val

    def search(self, L0, H, W, seed, deadline):
        beam = [(self.evaluate(L0), L0, None, [])]
        for depth in range(H):
            children = {}
            seed_child = None
            for v, L, first, path in beam:
                if L[0] >= ROUNDS:
                    k = tuple(L[2:])
                    old = children.get(k)
                    if old is None or v > old[0]:
                        children[k] = (v, L, first, path)
                    continue
                on_seed = seed is not None and depth < len(seed) and path == seed[:depth]
                for a in self.actions(L):
                    L2 = self.apply(L, a)
                    v2 = self.evaluate(L2)
                    key = tuple(L2[2:])
                    old = children.get(key)
                    if old is None or v2 > old[0]:
                        children[key] = (v2, L2, first if first is not None else a, path + [a])
                    if on_seed and a == seed[depth]:
                        seed_child = key
            if not children:
                break
            items = sorted(children.values(), key=lambda x: -x[0])
            nb = items[:W]
            if seed_child is not None:
                sc = children[seed_child]
                if all(sc[1] is not x[1] for x in nb):
                    nb.append(sc)
            beam = nb
            if time.perf_counter() > deadline and depth >= 2:
                break
        best = max(beam, key=lambda x: x[0])
        self.best_value = best[0]
        return best[2], best[3]


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


def decide(obs, memory):
    t0 = time.perf_counter()
    if 'S' not in memory:
        S = Static(obs)
        memory['S'] = S
        memory['seen'] = {}
        memory['M'] = pick_workshop(S)
        memory['plan'] = None
        memory['used'] = 0.0
        memory['spot_oeta'] = S.oeta[:]
        memory['opp_man'] = 0
        memory['last_opp_man'] = 0
    S = memory['S']
    rnd = obs['round']
    for s, v in enumerate(obs['stocks']):
        if v is not None and S.roles[s] in (T, F):
            memory['seen'][s] = (rnd, v)

    # Dynamic oeta update from opponent claims + spotted enemy boats
    claims = obs['claims']
    opp_man = sum(1 for c in S.manuf if claims[c] == 1 - S.seat)
    if opp_man > memory['opp_man']:
        memory['opp_man'] = opp_man
        memory['last_opp_man'] = rnd
    spot_oeta = memory['spot_oeta']
    for eb in obs.get('enemy_boats', ()):
        ep = eb['position'] if eb['eta'] == 0 else eb['destination']
        ee = eb['eta']
        for c in range(S.nc):
            if claims[c] == -1 and S.cdest[c] == ep:
                est_c = rnd + ee + 1
                if est_c < spot_oeta[c]:
                    spot_oeta[c] = est_c
        if S.roles[ep] in (T, F) and eb['eta'] == 0:
            if ep not in memory['seen'] or memory['seen'][ep][0] < rnd:
                memory['seen'][ep] = (rnd, max(0, est_stock(memory, ep, rnd) - 2))

    oeta = spot_oeta[:]
    unclaimed_man = sorted(
        (c for c in S.manuf if claims[c] == -1 and rnd < S.cdl[c]),
        key=lambda c: (S.cdl[c], S.dist[S.mo][S.cdest[c]])
    )
    if not unclaimed_man or (rnd >= 36 and len(unclaimed_man) <= 1):
        unclaimed_raw = sorted(
            (c for c in S.raws if claims[c] == -1 and rnd < S.cdl[c]),
            key=lambda c: (S.cdl[c], S.dist[S.ohome][S.cdest[c]])
        )
        for idx, c in enumerate(unclaimed_raw):
            if rnd >= oeta[c] and not (spot_oeta[c] < S.oeta[c]):
                oeta[c] = rnd + 3 + 3 * idx
    if opp_man == 0:
        if rnd >= 20:
            for idx, c in enumerate(unclaimed_man):
                est_m = rnd + 5 + 9 * idx + (S.dist[S.mo][S.cdest[c]] // 2)
                if est_m > oeta[c]:
                    oeta[c] = min(est_m, spot_oeta[c]) if spot_oeta[c] < S.oeta[c] else est_m
    else:
        for idx, c in enumerate(unclaimed_man):
            est_m = memory['last_opp_man'] + 8 * (idx + 1) + (S.dist[S.mo][S.cdest[c]] // 2)
            if est_m > oeta[c]:
                oeta[c] = min(est_m, spot_oeta[c]) if spot_oeta[c] < S.oeta[c] else est_m

    # Immediate delivery if already at destination with matching goods
    best_deliv = None
    for la in obs['legal_actions']:
        if la['op'] == 'deliver':
            if best_deliv is None or S.cpts[la['contract']] > S.cpts[best_deliv['contract']]:
                best_deliv = la
    if best_deliv is not None:
        memory['plan'] = None
        return best_deliv

    # If all boats are in transit, wait fast
    boats = obs['own']['boats']
    if all(bt['eta'] > 0 for bt in boats):
        memory['used'] += time.perf_counter() - t0
        return {'op': 'wait'}

    pl = Planner(S, memory['M'], oeta, rnd)
    L0 = pl.encode(obs, memory)
    remaining = max(0.05, PR['budget'] - memory['used'])
    per = remaining / max(4.0, (ROUNDS - rnd) * 0.42)
    deadline = t0 + per * 1.4
    seed = memory.get('plan')
    if seed:
        seed = seed[1:] if len(seed) > 1 else None
    H, W = PR['H'], PR['W']
    if rnd == 0 and PR.get('H0'):
        H, W = PR['H0'], PR['W0']
        deadline = t0 + PR['T0']
    if rnd == 0 and len(S.msites) > 1:
        best = None
        per_m = PR['T0'] / len(S.msites)
        for m in S.msites:
            plm = Planner(S, m, oeta, rnd)
            f, pth = plm.search(L0, H, W, None, time.perf_counter() + per_m)
            if best is None or plm.best_value > best[0]:
                best = (plm.best_value, m, f, pth)
        memory['M'] = best[1]
        first, path = best[2], best[3]
    else:
        first, path = pl.search(L0, H, W, seed, deadline)
    memory['plan'] = path
    memory['used'] += time.perf_counter() - t0
    if first is None:
        return {'op': 'wait'}
    return to_json(first)


def agent(observation, memory):
    """Return one entry of observation['legal_actions']. `memory` resets each game."""
    legal = observation['legal_actions']
    try:
        act = decide(observation, memory)
        if act is not None and act in legal:
            return act
        memory['plan'] = None
    except Exception as exc:
        print('agent error', repr(exc), file=sys.stderr)
        memory['plan'] = None
    return {'op': 'wait'} if {'op': 'wait'} in legal else legal[0]


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
