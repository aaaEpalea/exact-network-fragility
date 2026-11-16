"""Interval-bound certificate and function-level PGD attack for the fast-weight interpreter (FWI).
Certificate: for every (program p, token v, state j) the tube { s : |s - e_j| <= delta } is mapped into the tube of the successor state, with a correct readout,
for ANY perturbation |dF|,|dR|,|dS0| <= eps_F (fast weights, entrywise) and |dA|,|dB|,|dC|,|da|,|db| <= eps_th (slow weights, entrywise), applied independently at every time step.
=> exact execution of all programs for sequences of ANY length (induction on the tube invariant)."""
import numpy as np, jax, jax.numpy as jnp
from fw_lib import V, triples, run_fw
DELTAS = (0.0002, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.03, 0.06)

def _mr(c1, r1, c2, r2): return c1 * c2, np.abs(c1) * r2 + np.abs(c2) * r1 + r1 * r2

def cert_ok(progs, th, phi, eps_F, eps_th, delta, nmax, Kmax, chunk=400):
    A, B, C = [np.asarray(th[k], np.float64) for k in "ABC"]; a0, b0 = float(th["a"]), float(th["b"]); F, R, S0 = [np.asarray(phi[k], np.float64) for k in ("F", "R", "S0")]; dk = A.shape[0]; sq = np.sqrt(dk)
    T = triples(progs, nmax); tabs = [pr["table"] for pr in progs]
    def snap(zc, zr):
        c, r = _mr(a0, eps_th, zc, zr); c = c - b0; r = r + eps_th; return np.clip(c - r, 0, 1), np.clip(c + r, 0, 1)
    def bmul(Bm, sl, su, eps):          # interval of (B +- eps) s for s in [sl,su] (mid-rad)
        sc, sr = (sl + su) / 2, (su - sl) / 2; return sc @ Bm.T, sr @ np.abs(Bm).T + eps * (np.abs(sc) + sr).sum(-1, keepdims=True)
    # ---- initial states
    for p, pr in enumerate(progs):
        dS = S0.shape[1]; cg_c, cg_r = C[:dS, p] / sq, eps_th / sq; zc = S0 @ cg_c; zr = np.abs(S0) @ np.full(dS, cg_r) + eps_F * (np.abs(cg_c) + cg_r).sum()
        l, u = snap(zc, zr); e = np.eye(nmax)[pr["init"]]
        if np.any(l < e - delta) or np.any(u > e + delta): return False
    for s0 in range(0, len(T), chunk):
        Tc = T[s0:s0 + chunk]; M = len(Tc); p, v, j, i = Tc.T
        S = np.eye(nmax)[j]; sl, su = np.clip(S - delta, 0, 1), np.clip(S + delta, 0, 1)
        axc, axr = A[:, v].T, np.full((M, dk), eps_th); cgc, cgr = C[:, p].T, np.full((M, dk), eps_th); bsc, bsr = bmul(B, sl, su, eps_th)
        kc, kr = _mr(axc, axr, bsc, bsr); kc, kr = _mr(kc, kr, cgc, cgr); kc, kr = kc / sq, kr / sq
        zc = kc @ F.T; zr = kr @ np.abs(F).T + eps_F * (np.abs(kc) + kr).sum(-1, keepdims=True)
        l, u = snap(zc, zr); E = np.eye(nmax)[i]
        if np.any(l < E - delta) or np.any(u > E + delta): return False
        # readout from the (computed) successor tube
        dR = R.shape[1]; bc, br = bmul(B, l, u, eps_th); rc, rr = _mr(bc, br, cgc, cgr); rc, rr = rc[:, :dR] / sq, rr[:, :dR] / sq
        yc = rc @ R.T; yr = rr @ np.abs(R).T + eps_F * (np.abs(rc) + rr).sum(-1, keepdims=True); yl, yu = yc - yr, yc + yr
        for m in range(M):
            tc = tabs[p[m]][i[m]]; oth = np.delete(yu[m], tc)
            if yl[m, tc] <= oth.max(): return False
    return True

def cert_any(progs, th, phi, eF, eT, nmax, Kmax):
    return any(cert_ok(progs, th, phi, eF, eT, d, nmax, Kmax) for d in DELTAS)

def radius(fn, lo=1e-6, hi=1.0, it=14):
    """largest x with fn(x) True (monotone), log bisection; 0 if fn(lo) is False; hi if fn(hi) True"""
    if not fn(lo): return 0.0
    if fn(hi): return hi
    for _ in range(it):
        mid = float(np.sqrt(lo * hi)); lo, hi = (mid, hi) if fn(mid) else (lo, mid)
    return lo

def certified_radii(progs, th, phi, nmax, Kmax):
    eF = radius(lambda e: cert_any(progs, th, phi, e, 0.0, nmax, Kmax)); eT = radius(lambda e: cert_any(progs, th, phi, 0.0, e, nmax, Kmax))
    # joint ray: both scaled by t relative to their single-target radii
    t = radius(lambda t: cert_any(progs, th, phi, t * eF, t * eT, nmax, Kmax), lo=1e-3, hi=1.0, it=10) if eF > 0 and eT > 0 else 0.0
    return dict(eps_F=eF, eps_theta=eT, joint_t=t, joint_eps_F=t * eF, joint_eps_theta=t * eT)

# ------------------------------------------------------------------ function-level attack
def attack_set(P, L, seed=0, B=64):
    rng = np.random.default_rng(seed); X = rng.choice(V, size=(B, L))
    for v in range(V): X[v] = v                                           # constant sequences
    for q in range(24): v = rng.integers(V); m = rng.random(L) < 0.85; X[V + q] = np.where(m, v, rng.integers(V, size=L))   # biased sequences (drive counters to extremes)
    return X

def func_attack(progs, th, phi, refs, X, eF, eT, nmax, steps=80, restarts=2, seed=0, mask_theta=True):
    """projected sign-gradient attack on (phi,theta) in boxes eF / eT, one fixed perturbation for the whole run (same at every step); returns the smallest output margin found (<=0: a real wrong output)"""
    P = len(progs); Xj = jnp.array(X); Rj = jnp.array(refs); thj = {k: jnp.asarray(v) for k, v in th.items()}; phj = {k: jnp.asarray(v) for k, v in phi.items()}
    tk = ["A", "B", "C", "a", "b"]; pk = ["F", "R", "S0"]; rng = np.random.default_rng(seed)
    def margins(dth, dph):
        t2 = {k: thj[k] + (dth[k] if k in dth else 0.0) for k in thj}; p2 = {k: phj[k] + dph[k] for k in phj}
        lg = run_fw(t2, p2, Xj, P, nmax); corr = jnp.take_along_axis(lg, Rj[..., None], -1)[..., 0]
        oth = jnp.where(jax.nn.one_hot(Rj, lg.shape[-1]) > 0, -1e9, lg).max(-1); return corr - oth
    obj = lambda dth, dph: -0.05 * jax.nn.logsumexp(-margins(dth, dph).reshape(-1) / 0.05)
    g = jax.jit(jax.value_and_grad(obj, argnums=(0, 1))); mj = jax.jit(margins); best = 1e9
    sg = lambda shp: jnp.asarray(rng.choice([-1.0, 1.0], size=shp), jnp.float32)
    for r in range(restarts):
        dth = {k: eT * sg(np.shape(thj[k])) for k in tk} if eT > 0 else {k: jnp.zeros_like(thj[k]) for k in tk}
        dph = {k: eF * sg(phj[k].shape) for k in pk} if eF > 0 else {k: jnp.zeros_like(phj[k]) for k in pk}
        for t in range(steps):
            _, (gt, gp) = g(dth, dph)
            if eT > 0: dth = {k: jnp.clip(dth[k] - eT / 6 * jnp.sign(gt[k]), -eT, eT) for k in tk}
            if eF > 0: dph = {k: jnp.clip(dph[k] - eF / 6 * jnp.sign(gp[k]), -eF, eF) for k in pk}
        best = min(best, float(mj(dth, dph).min()))
    return best

def first_break(progs, th, phi, refs, X, eF0, eT0, t_cert, nmax, grow=1.6, tmax=64, bis=5, **kw):
    """smallest scale t (>= t_cert) of the box (t*eF0, t*eT0) at which the attack finds a wrong output: doubling search then bisection. Returns (t_break or None, margin at t_cert...)."""
    f = lambda t: func_attack(progs, th, phi, refs, X, t * eF0, t * eT0, nmax, **kw)
    t = max(t_cert, 1e-6); m0 = f(t)
    if m0 <= 0: return t, m0, t
    lo = t; hi = None; tt = t
    while tt < tmax:
        tt *= grow; m = f(tt)
        if m <= 0: hi = tt; break
        lo = tt
    if hi is None: return None, m0, lo
    for _ in range(bis):
        mid = float(np.sqrt(lo * hi)); m = f(mid)
        if m <= 0: hi = mid
        else: lo = mid
    return hi, m0, lo
