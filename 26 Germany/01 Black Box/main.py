import numpy as np

VARS = ["x1", "x2", "x3", "x4", "x5"]
PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
LIMIT = 140          # hard cap is 150, keep a margin
TOL = 0.004          # bracket length when locating the boundary on a segment


def features(X, squares=True):
    X = np.atleast_2d(np.asarray(X, float))
    cols = [np.ones(len(X))] + [X[:, i] for i in range(5)]
    cols += [X[:, i] * X[:, j] for i, j in PAIRS]
    if squares:
        cols += [X[:, i] ** 2 for i in range(5)]
    return np.column_stack(cols)


class OutOfBudget(Exception):
    pass


class Probe:
    def __init__(self, oracle):
        self.oracle = oracle
        self.X, self.y, self.seen = [], [], set()

    def left(self):
        return LIMIT - self.oracle.spent

    def ask(self, pts, phase):
        pts = [[float(min(1.0, max(0.0, v))) for v in p] for p in pts]
        new = []
        for p in pts:
            k = self.oracle.key(p)
            if k not in self.seen and k not in {self.oracle.key(q) for q in new}:
                new.append(p)
        if len(new) > self.left():
            raise RuntimeError("budget")
        ys = self.oracle.ask_many(pts, phase)
        for p, v in zip(pts, ys):
            k = self.oracle.key(p)
            if k not in self.seen:
                self.seen.add(k)
                self.X.append(p)
                self.y.append(v)
        return np.array(ys, float)


def design(n_inner, seed):
    rng = np.random.default_rng(seed)
    verts = np.array([[(k >> i) & 1 for i in range(5)] for k in range(32)], float)
    lhs = np.column_stack([(rng.permutation(n_inner) + rng.random(n_inner)) / n_inner
                           for _ in range(5)])
    return np.vstack([verts, [[0.5] * 5], lhs])


def cut_gains(X, Q, r0, W):
    """For each direction in W, the best threshold cut and how much RSS it removes."""
    N = len(r0)
    P = (X @ W.T).T
    idx = np.argsort(-P, axis=1)
    Ps = np.take_along_axis(P, idx, axis=1)
    cq = np.cumsum(Q[idx], axis=1)
    cr = np.cumsum(r0[idx], axis=1)
    b = np.arange(1, N + 1)[None, :] - (cq ** 2).sum(2)
    ok = np.zeros(b.shape, bool)
    ok[:, :-1] = (Ps[:, :-1] - Ps[:, 1:]) > 1e-9
    ok &= b > 1e-6
    gain = np.where(ok, cr ** 2 / np.where(ok, b, 1.0), -1.0)
    k = gain.argmax(1)
    r = np.arange(len(W))
    thr = (Ps[r, k] + Ps[r, np.minimum(k + 1, N - 1)]) / 2
    return gain[r, k], thr


def random_dirs(rng, n):
    W = np.zeros((n, 5))
    sizes = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.15, 0.3, 0.3, 0.15, 0.1])
    for r, s in enumerate(sizes):
        sup = rng.choice(5, s, replace=False)
        W[r, sup] = rng.normal(size=s)
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def find_split(X, y, seed=7):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(features(X))
    r0 = y - Q @ (Q.T @ y)
    base = r0 @ r0
    eps = 1e-9 * len(y) + 1e-12 * base
    fixed = []
    for i, j in PAIRS:
        for si in (1, -1):
            fixed.append(np.eye(5)[i] + si * np.eye(5)[j])
    W = np.vstack([np.eye(5), fixed, random_dirs(rng, 6000)])
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    allW, allG, allT = [], [], []

    def run(W):
        for s in range(0, len(W), 1500):
            g, t = cut_gains(X, Q, r0, W[s:s + 1500])
            allW.append(W[s:s + 1500]); allG.append(g); allT.append(t)

    run(W)
    for sigma in [0.4, 0.2, 0.1, 0.05, 0.02, 0.4, 0.1, 0.02]:
        G = np.concatenate(allG)
        if base - G.max() < eps:
            break
        top = np.concatenate(allW)[np.argsort(-G)[:15]]
        Wn = np.repeat(top, 150, axis=0) + sigma * rng.normal(size=(15 * 150, 5))
        Wn /= np.linalg.norm(Wn, axis=1, keepdims=True)
        run(Wn)
    Wa, G, T = np.concatenate(allW), np.concatenate(allG), np.concatenate(allT)
    b = G.argmax()
    w, t = Wa[b], T[b]
    s = X @ w >= t
    # average every direction that gives the same labelling: a rough centre of the feasible cone
    good = np.where(base - G < max(eps, (base - G[b]) * 1.001 + 1e-15))[0]
    acc = []
    for k in good[:3000]:
        sk = X @ Wa[k] >= T[k]
        if (sk == s).all():
            acc.append(np.append(Wa[k], T[k]))
        elif (sk != s).all():
            acc.append(-np.append(Wa[k], T[k]))
    if acc:
        v = np.mean(acc, 0)
        w, t = v[:5] / np.linalg.norm(v[:5]), v[5] / np.linalg.norm(v[:5])
    return w, t, s, base - G[b]


def fit(X, y, s, squares=True):
    F = np.column_stack([features(X, squares), s.astype(float)])
    coef = np.linalg.lstsq(F, y, rcond=None)[0]
    res = y - F @ coef
    return coef[:-1], coef[-1], res


def plane(P, idx):
    Z = P[:, idx]
    c = Z.mean(0)
    n = np.linalg.svd(Z - c)[2][-1]
    w = np.zeros(5)
    w[idx] = n
    return w, n @ c, np.abs((Z - c) @ n).max()


def solve(oracle, points):
    pr = Probe(oracle)
    D = design(30, 2026)
    pr.ask(D, "design")
    X, y = np.array(pr.X), np.array(pr.y)
    w, t, s, rss = find_split(X, y)
    if rss > 1e-6 * len(y):
        rng = np.random.default_rng(99)
        pr.ask(rng.random((12, 5)), "design-extra")
        X, y = np.array(pr.X), np.array(pr.y)
        w, t, s, rss = find_split(X, y, seed=11)
    beta, J, res = fit(X, y, s)
    print(f"design {len(y)} pts, split rss {rss:.2e}, J {J:+.4f}, max res {np.abs(res).max():.1e}")

    def g(x):
        return float(features(x) @ beta)

    lab_X, lab_y, lab_s = list(X), list(y), [bool(v) for v in s]

    def side(x, phase):
        if pr.left() < 1:
            raise OutOfBudget
        v = pr.ask([x], phase)[0]
        r = v - g(x)
        sd = abs(r - J) < abs(r)
        lab_X.append(x); lab_y.append(v); lab_s.append(sd)
        return sd

    bpts = []
    A = np.where(s)[0]
    B = np.where(~s)[0]

    def pick_pair(used):
        da = X[A] @ w - t
        db = X[B] @ w - t
        ia, ib = np.meshgrid(np.arange(len(A)), np.arange(len(B)), indexing="ij")
        ia, ib = ia.ravel(), ib.ravel()
        ok = (da[ia] > 0) & (db[ib] < 0)
        ok &= np.array([(A[i], B[j]) not in used for i, j in zip(ia, ib)], bool)
        if not ok.any():
            return None
        ia, ib = ia[ok], ib[ok]
        span = da[ia] - db[ib]
        C = X[A[ia]] + (da[ia] / span)[:, None] * (X[B[ib]] - X[A[ia]])
        L = np.linalg.norm(X[B[ib]] - X[A[ia]], axis=1)
        cost = np.log2(2 + L / TOL)
        if len(bpts) < 2:
            d = np.ones(len(C)) if not bpts else np.linalg.norm(C - bpts[0], axis=1)
            k = int(np.argmax(d / cost))
        else:
            # pick the crossing that best widens the spread of boundary points along the plane
            Pb = np.array(bpts)
            m = len(Pb)
            S0 = Pb.T @ Pb
            s0 = Pb.sum(0)

            def logdet(S, sm, cnt):
                M = S - sm[..., :, None] * sm[..., None, :] / cnt
                ev = np.linalg.eigvalsh(M)
                return np.log(ev[..., 1:] + 1e-4).sum(-1)

            base = logdet(S0, s0, m)
            new = logdet(S0[None] + C[:, :, None] * C[:, None, :], s0[None] + C, m + 1)
            k = int(np.argmax((new - base) / cost))
        return A[ia[k]], B[ib[k]]

    def bracket(a, b, guess):
        n = np.linalg.norm(a - b)
        lo, hi = 0.0, 1.0
        if guess is not None:
            u, step = guess, TOL / n
            while lo < u < hi:
                if side(a + u * (b - a), "boundary"):
                    lo, u = u, u + step
                else:
                    hi, u = u, u - step
                if hi - lo <= step * 1.01 or (hi < 1 and lo > 0):
                    break
                step *= 2
        while (hi - lo) * n > TOL:
            u = (lo + hi) / 2
            if side(a + u * (b - a), "boundary"):
                lo = u
            else:
                hi = u
        return a + (lo + hi) / 2 * (b - a)

    used = set()
    while len(bpts) < 16 and pr.left() >= 9:
        pair = pick_pair(used)
        if pair is None:
            break
        used.add(pair)
        a, b = X[pair[0]], X[pair[1]]
        guess = None
        if len(bpts) >= 6:
            guess = float(np.clip((w @ a - t) / (w @ (a - b)), 0.02, 0.98))
        try:
            bpts.append(bracket(a, b, guess))
        except OutOfBudget:
            break
        if len(bpts) >= 6:
            w2, t2, _ = plane(np.array(bpts), list(range(5)))
            if w2 @ a < t2:
                w2, t2 = -w2, -t2
            w, t = w2, t2

    P = np.array(bpts)
    LX, LS = np.array(lab_X), np.array(lab_s)
    gates = list(range(5))
    if len(P) < 6:
        gates = [i for i in range(5) if abs(w[i]) > 0.1 * np.abs(w).max()]
    while len(gates) > 1 and len(P) >= 6:
        trial = []
        for v in gates:
            idx = [k for k in gates if k != v]
            if len(P) > len(idx):
                trial.append((plane(P, idx)[2], v))
        if not trial:
            break
        err, v = min(trial)
        if err > 1.5 * TOL:
            break
        gates.remove(v)
    if len(P) >= 6:
        w, t, _ = plane(P, gates)
    else:
        w = np.where(np.isin(np.arange(5), gates), w, 0.0)
    if (LX[LS] @ w - t).mean() < 0:
        w, t = -w, -t
    margin = (LX @ w - t) * np.where(LS, 1, -1)
    wrong = int((margin < -TOL).sum())

    LY = np.array(lab_y)
    squares = True
    b2, J2, r2 = fit(LX, LY, LS, squares=False)
    if np.abs(r2).max() < 1e-4:
        beta, J, res, squares = b2, J2, r2, False
    else:
        beta, J, res = fit(LX, LY, LS)
    exact = np.abs(res).max() < 1e-4

    # f = g + J*[w.x >= t] = (g + J) - J*[w.x < t]; report the side where the step is upward
    if J < 0:
        w, t, J = -w, -t, -J
        beta = beta.copy()
        beta[0] -= J
    # unit-scale the boundary so the largest gate weight is 1
    sc = np.abs(w).max()
    w, t = w / sc, t / sc

    lin = beta[1:6]
    prods = beta[6:16]
    sq = beta[16:21] if squares else np.zeros(5)
    k = int(np.abs(prods).argmax())
    inter = []
    rel = set(gates)
    if abs(prods[k]) > 1e-3:
        i, j = PAIRS[k]
        rel |= {i, j}
        c = float(prods[k])
        inter.append({"variables": [VARS[i], VARS[j]], "type": "product",
                      "coefficient": round(c, 6),
                      "equivalent_centered_form": f"{c:.6g}*({VARS[i]}-0.5)*({VARS[j]}-0.5) + linear terms"})
    for i in range(5):
        if abs(lin[i]) > 2e-3 or abs(sq[i]) > 2e-3:
            rel.add(i)
    for kk in range(10):
        if kk != k and abs(prods[kk]) > 2e-3:
            rel |= set(PAIRS[kk])

    terms = [f"{beta[0]:.6f}"]
    terms += [f"{lin[i]:+.6f}*{VARS[i]}" for i in range(5) if abs(lin[i]) > 1e-5]
    terms += [f"{prods[m]:+.6f}*{VARS[a]}*{VARS[b]}" for m, (a, b) in enumerate(PAIRS)
              if abs(prods[m]) > 1e-5]
    terms += [f"{sq[i]:+.6f}*{VARS[i]}**2" for i in range(5) if abs(sq[i]) > 1e-5]
    cond = " + ".join(f"{w[i]:.6f}*{VARS[i]}" for i in gates)
    expr = " ".join(terms) + f" + ({J:.6f} if {cond} >= {t:.6f} else 0.0)"

    Pt = np.array(points, float)
    pred = features(Pt, squares) @ beta + J * (Pt @ w >= t)

    gate_names = [VARS[i] for i in gates]
    notes = (f"{len(D)} design points (32 cube vertices, centre, 30 latin hypercube) to fit "
             f"a full quadratic plus a step; the step was found by scanning half-space "
             f"cuts of the residuals until the residual sum dropped to {rss:.1e}. "
             f"{len(bpts)} boundary points were then bracketed by bisection to within "
             f"{TOL} on segments joining points from opposite sides, the plane fitted through them and variables dropped "
             f"from it while the fit stayed within tolerance. Model reproduces every "
             f"query to {np.abs(res).max():.1e}; {wrong} labelled points disagree with the plane. "
             f"Queries used: {oracle.spent}.")
    print(notes)
    return {
        "relevant_variables": [VARS[i] for i in sorted(rel)],
        "irrelevant_variables": [VARS[i] for i in range(5) if i not in rel],
        "interactions": inter,
        "discontinuities": [{
            "gate_variables": gate_names,
            "boundary": {"form": "linear",
                         "coefficients": {v: round(float(w[i]), 6) for i, v in enumerate(VARS)},
                         "threshold": round(float(t), 6),
                         "active_when": "greater_equal"},
            "jump": round(float(J), 6),
            "condition": f"{cond} >= {t:.6f}",
        }],
        "model_expression": expr,
        "predictions": [round(float(v), 6) for v in pred],
        "notes": notes,
        "confidence": "high" if exact and wrong == 0 else "medium",
    }
