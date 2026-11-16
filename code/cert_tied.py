"""Interval-bound certificates for the weight-tied designs (valid for any number of heads k, any length): 'bil' (tied-bilinear) and 'relu' (ReLU-gated original)."""
import numpy as np
V = 6
def _imat(W, lo, hi, eps):
    a, b = W - eps, W + eps; c = np.stack([a * lo, a * hi, b * lo, b * hi]); return c.min(0).sum(1), c.max(0).sum(1)
def ok_tied(design, N, eps, delta, protect):
    if eps > delta: return False
    n = 2 * N + 1
    Pu = np.zeros((n, n)); Pu[np.arange(1, n), np.arange(n - 1)] = 1; Pu[n - 1, n - 1] = 1; Pd = np.zeros((n, n)); Pd[np.arange(n - 1), np.arange(1, n)] = 1; Pd[0, 0] = 1
    P = np.stack([Pu, Pd, np.eye(n)]); T = np.zeros((2, n)); T[0, :N + 1] = 1; T[1, N + 1:] = 1; bias = -np.ones((3, n))
    E = lambda W: eps * (W != 0) if protect else eps
    for i0 in range(n):
        for r in range(3):
            s = np.eye(n)[i0]; nxt = [min(i0 + 1, n - 1), max(i0 - 1, 0), i0][r]; es = np.eye(n)[nxt]; ey = int(nxt > N)
            g = np.eye(3)[r]; eg = eps * (g != 0) if protect else eps * np.ones(3); gl, gu = g - eg, g + eg
            zl = np.zeros(n); zu = np.zeros(n)
            for rr in range(3):
                ql, qu = _imat(P[rr], s - delta, s + delta, E(P[rr]))
                if design == "bil":
                    c = np.stack([gl[rr] * ql, gl[rr] * qu, gu[rr] * ql, gu[rr] * qu]); zl += c.min(0); zu += c.max(0)
                else:
                    pl = ql + gl[rr] + bias[rr] - eps; pu = qu + gu[rr] + bias[rr] + eps; zl += np.maximum(pl, 0); zu += np.maximum(pu, 0)
            s2l, s2u = np.clip(2 * zl - 0.5, 0, 1), np.clip(2 * zu - 0.5, 0, 1)
            if np.any(s2l < es - delta) or np.any(s2u > es + delta): return False
            yl, yu = _imat(T, s2l, s2u, E(T)); oth = np.delete(yu, ey)
            if oth.size and yl[ey] <= oth.max(): return False
    return True
def feasible(ok, eps):
    for d in np.linspace(max(eps, 0.01), 0.49, 40):
        if ok(eps, float(d)): return float(d)
    return None
def cert_radius_tied(design, N, protect):
    ok = lambda e, d: ok_tied(design, N, e, d, protect); lo, hi = 1e-4, 0.45
    if feasible(ok, lo) is None: return 0.0, None
    if feasible(ok, hi) is not None: return hi, feasible(ok, hi)
    for _ in range(14):
        mid = float(np.sqrt(lo * hi))
        if feasible(ok, mid) is not None: lo = mid
        else: hi = mid
    return lo, feasible(ok, lo)
