"""Digital-first compiler prototype.

Spec  = V tokens + a bundle of finite-state machines (components) + a readout rule.
Output = dense-weight recurrent network with one-hot state, token-gated transitions and a restoring 'snap'.
Certificates:
  (1) exact:  every (component, state, token) transition + readout is verified on the compiled network
              => correct for sequences of ANY length (induction on a one-hot invariant).
  (2) noise:  interval-bound-propagation (IBP) proves a tube invariant under any L_inf parameter perturbation
              |dw| <= eps  => provable noise radius valid for ANY length.
"""
from dataclasses import dataclass
import numpy as np, jax, jax.numpy as jnp


@dataclass
class FSM:
    n: int
    init: int
    delta: np.ndarray            # (V, n): next state for each token and state


@dataclass
class Spec:
    V: int
    comps: list
    readout: dict                # {"type":"selected"} or {"type":"table","comp":c,"table":[...],"n_out":K}
    name: str = ""


def heads_of(spec):
    return spec.readout if isinstance(spec.readout, list) else [spec.readout]


def reference(spec, tokens):
    """Single head -> list of ints; several heads -> list of tuples."""
    st = [c.init for c in spec.comps]; out = []; hs = heads_of(spec)
    for v in tokens:
        st = [int(c.delta[v, s]) for c, s in zip(spec.comps, st)]
        ys = [st[v] if h["type"] == "selected" else int(h["table"][st[h["comp"]]]) for h in hs]
        out.append(ys[0] if len(hs) == 1 else tuple(ys))
    return out


def compile_spec(spec, lo=0.25, hi=0.75):
    V = spec.V; ns = [c.n for c in spec.comps]; off = np.concatenate([[0], np.cumsum(ns)]); D = int(off[-1]); H = V * D
    g = 1.0 / (hi - lo)
    A = np.zeros((H, D)); B = np.zeros((H, V)); C = np.zeros((D, H)); owner = np.zeros(D, int)
    for ci, c in enumerate(spec.comps):
        owner[off[ci]:off[ci + 1]] = ci
        for v in range(V):
            for j in range(c.n):
                i = int(c.delta[v, j]); A[v * D + off[ci] + i, off[ci] + j] = 1.0
    for v in range(V):
        B[v * D:(v + 1) * D, v] = 1.0
        for gi in range(D): C[gi, v * D + gi] = 1.0
    I = np.eye(D)
    p = {"A": A, "B": B, "b": -np.ones(H), "C": C, "S1": g * I, "bs1": -g * lo * np.ones(D), "S2": g * I, "bs2": -g * hi * np.ones(D),
         "C3": I, "C4": -I, "s0": np.concatenate([np.eye(c.n)[c.init] for c in spec.comps])}
    hs = heads_of(spec)
    if hs[0]["type"] == "selected":
        assert len(hs) == 1 and len(spec.comps) == V, "'selected' readout: single head, one component per token"
        n_out = max(ns); Bt = np.zeros((D, V)); R = np.zeros((n_out, D))
        for ci, c in enumerate(spec.comps):
            Bt[off[ci]:off[ci + 1], ci] = 1.0
            for i in range(c.n): R[i, off[ci] + i] = 1.0
        p.update({"I3": I, "Bt": Bt, "b3": -np.ones(D), "R": R})
    else:
        for k, ro in enumerate(hs):
            K = ro["n_out"]; T = np.zeros((K, D)); ci = ro["comp"]
            for i in range(spec.comps[ci].n): T[ro["table"][i], off[ci] + i] = 1.0
            p[f"T{k}"] = T
    meta = {"ns": ns, "off": off, "D": D, "H": H, "owner": owner, "type": hs[0]["type"], "nheads": len(hs)}
    return {k: jnp.asarray(v, jnp.float32) for k, v in p.items()}, meta


def step(p, meta, s, x):
    h = jax.nn.relu(s @ p["A"].T + x @ p["B"].T + p["b"]); z = h @ p["C"].T
    s2 = jax.nn.relu(z @ p["S1"].T + p["bs1"]) @ p["C3"].T + jax.nn.relu(z @ p["S2"].T + p["bs2"]) @ p["C4"].T
    if meta["type"] == "selected":
        r = jax.nn.relu(s2 @ p["I3"].T + x @ p["Bt"].T + p["b3"]); y = r @ p["R"].T
        return s2, [y]
    return s2, [s2 @ p[f"T{k}"].T for k in range(meta["nheads"])]


def run(p, meta, X, V):
    xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); s = jnp.broadcast_to(p["s0"], (X.shape[0], p["s0"].shape[0]))
    def f(s, x):
        s2, ys = step(p, meta, s, x); return s2, jnp.stack([jnp.argmax(y, -1) for y in ys], -1)
    _, ys = jax.lax.scan(f, s, xs); ys = jnp.swapaxes(ys, 0, 1)
    return ys[..., 0] if meta["nheads"] == 1 else ys


def _configs(spec):
    base = [c.init for c in spec.comps]
    for ci, c in enumerate(spec.comps):
        for i in range(c.n):
            for v in range(spec.V):
                cfg = list(base); cfg[ci] = i
                s = np.concatenate([np.eye(cc.n)[x] for cc, x in zip(spec.comps, cfg)])
                exp_cfg = [int(cc.delta[v, x]) for cc, x in zip(spec.comps, cfg)]
                exp_s = np.concatenate([np.eye(cc.n)[x] for cc, x in zip(spec.comps, exp_cfg)])
                exp_y = [exp_cfg[v] if h["type"] == "selected" else int(h["table"][exp_cfg[h["comp"]]]) for h in heads_of(spec)]
                yield v, s, exp_s, exp_y


def certify_exact(spec, p, meta, tol=1e-5):
    """Check every (component, state, token) transition and readout on the compiled network.
    Valid for all lengths because nominal transition weights are block-diagonal per component (checked)
    and the one-step map depends only on (state, token)."""
    V, D = spec.V, meta["D"]; ok = tot = 0
    A = np.asarray(p["A"]); owner = meta["owner"]
    for row in range(A.shape[0]):
        g = row % D; cols = np.nonzero(A[row])[0]
        assert all(owner[c] == owner[g] for c in cols), "non-block-diagonal transition"
    for v, s, exp_s, exp_y in _configs(spec):
        s2, ys = step(p, meta, jnp.array(s[None], jnp.float32), jnp.array(np.eye(V)[v][None], jnp.float32))
        good = float(np.abs(np.asarray(s2)[0] - exp_s).max()) < tol and all(int(np.argmax(np.asarray(y)[0])) == e for y, e in zip(ys, exp_y))
        ok += good; tot += 1
    return ok, tot


def _imat(W, lo, hi, eps):
    """Interval of (W' @ v) for W' in [W-eps, W+eps] entrywise, v in [lo,hi] (per coordinate)."""
    a, b = W - eps, W + eps
    c = np.stack([a * lo, a * hi, b * lo, b * hi]); return c.min(0).sum(1), c.max(0).sum(1)


def _relu(l, u): return np.maximum(l, 0), np.maximum(u, 0)


def tube_ok(spec, p, meta, eps, delta, protect_zeros=False):
    """Is the tube {|s - onehot| <= delta} invariant, with correct readout, under ANY |dparam| <= eps ?
    protect_zeros=True: entries that are exactly 0 in the compiled weights stay 0 (only nonzero entries are perturbed)."""
    P = {k: np.asarray(v, np.float64) for k, v in p.items()}; V = spec.V; sel = meta["type"] == "selected"
    E = (lambda k: eps * (P[k] != 0)) if protect_zeros else (lambda k: eps)
    for v, s, exp_s, exp_y in _configs(spec):
        x1 = np.eye(V)[v]; sl, su = s - delta, s + delta
        hl, hu = _imat(P["A"], sl, su, E("A")); bl, bu = _imat(P["B"], x1, x1, E("B")); eb = E("b")
        hl, hu = _relu(hl + bl + P["b"] - eb, hu + bu + P["b"] + eb)
        zl, zu = _imat(P["C"], hl, hu, E("C"))
        u1l, u1u = [t + o for t, o in zip(_imat(P["S1"], zl, zu, E("S1")), (P["bs1"] - E("bs1"), P["bs1"] + E("bs1")))]
        zabs = np.maximum(np.abs(zl), np.abs(zu))
        eta = (2 * eps * zabs + 2 * eps) if protect_zeros else (2 * eps * zabs.sum() + 2 * eps)
        sl_, su_ = np.minimum(np.maximum(u1l, 0), 1 - eta), np.minimum(np.maximum(u1u, 0), 1 + eta)
        a1u, a2u = np.maximum(u1u, 0), np.maximum(u1u - 1 + eta, 0)
        slack = eps * (a1u + a2u) if protect_zeros else eps * (a1u.sum() + a2u.sum())
        s2l, s2u = sl_ - slack, su_ + slack
        if np.any(s2l < exp_s - delta) or np.any(s2u > exp_s + delta): return False
        if sel:
            rl, ru = [t + o for t, o in zip(_imat(P["I3"], s2l, s2u, E("I3")), (P["b3"] - E("b3"), P["b3"] + E("b3")))]
            tl, tu = _imat(P["Bt"], x1, x1, E("Bt")); rl, ru = _relu(rl + tl, ru + tu)
            heads_yl_yu = [_imat(P["R"], rl, ru, E("R"))]
        else:
            heads_yl_yu = [_imat(P[f"T{k}"], s2l, s2u, E(f"T{k}")) for k in range(meta["nheads"])]
        for (yl, yu), ey in zip(heads_yl_yu, exp_y):
            others = np.delete(yu, ey)
            if others.size and yl[ey] <= others.max(): return False
    return True


def certified_radius(spec, p, meta, deltas=(0.04, 0.06, 0.1, 0.15, 0.2)):
    """Largest eps on a log grid for which some tube width delta is provably invariant. Returns (eps, delta)."""
    best = (0.0, None)
    for eps in np.logspace(-5, -1, 17):
        found = next((d for d in deltas if eps <= d and tube_ok(spec, p, meta, eps, d)), None)
        if found is None: break
        best = (float(eps), found)
    return best


def sat(n, i, d): return int(min(max(i + d, 0), n - 1))


def prefix_hist(N, V=6):
    comps = []
    for c in range(V):
        delta = np.array([[sat(N + 1, i, 1) if t == c else i for i in range(N + 1)] for t in range(V)]); comps.append(FSM(N + 1, 0, delta))
    return Spec(V, comps, {"type": "selected"}, f"prefix_hist(N={N})")


def majority(N, V=6):
    n = 2 * N + 1; delta = np.array([[sat(n, i, 1) if t == 0 else sat(n, i, -1) if t == 1 else i for i in range(n)] for t in range(V)])
    return Spec(V, [FSM(n, N, delta)], {"type": "table", "comp": 0, "table": [int(i > N) for i in range(n)], "n_out": 2}, f"majority(N={N})")


def parity(V=6):
    delta = np.array([[(i + 1) % 2 if t == 0 else i for i in range(2)] for t in range(V)])
    return Spec(V, [FSM(2, 0, delta)], {"type": "table", "comp": 0, "table": [0, 1], "n_out": 2}, "parity_of_token0")


def mod_count(m, V=6):
    delta = np.array([[(i + 1) % m if t == 0 else i for i in range(m)] for t in range(V)])
    return Spec(V, [FSM(m, 0, delta)], {"type": "table", "comp": 0, "table": list(range(m)), "n_out": m}, f"count_mod{m}")


def dyck(N, V=6):
    n = N + 2; dead = N + 1; rows = []
    for t in range(V):
        row = []
        for i in range(n):
            if i == dead: row.append(dead)
            elif t == 0: row.append(sat(N + 1, i, 1) if i < N else N)
            elif t == 1: row.append(i - 1 if i > 0 else dead)
            else: row.append(i)
        rows.append(row)
    return Spec(V, [FSM(n, 0, np.array(rows))], {"type": "table", "comp": 0, "table": [1] * (N + 1) + [0], "n_out": 2}, f"dyck_valid_prefix(N={N})")


def compose_parallel(specs, name=None):
    """Capability composition: one network that runs all programs on the same token stream (several heads).
    Components are concatenated, so the weight matrices are block-diagonal; off-block zeros are real (noisy) parameters."""
    V = specs[0].V; comps, heads, shift = [], [], 0
    for sp in specs:
        assert sp.V == V and all(h["type"] == "table" for h in heads_of(sp)), "parallel composition: same V, table readouts"
        for h in heads_of(sp): heads.append({**h, "comp": h["comp"] + shift})
        comps += sp.comps; shift += len(sp.comps)
    return Spec(V, comps, heads, name or "+".join(sp.name for sp in specs))
