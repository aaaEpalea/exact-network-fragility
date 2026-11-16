"""Baselines for the shared-capacity question, all on the same programs / token streams / noise protocols as the fast-weight interpreter (FWI):
 (1) independent networks: one second-order (bilinear) restoring network per program      z = sum_v x_v W_v s,   s' = clip(2z-.5,0,1)
 (2) block-diagonal stacking: ONE dense network whose W_v are block-diagonal over programs (off-block zeros are real, noisy parameters)
 (3) existing weight-tied composition: shared role matrices {up,down,hold} + per-program role gates (template programs only)
"""
import numpy as np, jax, jax.numpy as jnp
from fw_lib import V, ref_run, perturb, s50
def build_indep(progs):
    out = []
    for pr in progs:
        n = pr["n"]; W = np.zeros((V, n, n), np.float32); T = np.zeros((pr["K"], n), np.float32)
        for v in range(V):
            for j in range(n): W[v, pr["delta"][v, j], j] = 1.0
        for i in range(n): T[pr["table"][i], i] = 1.0
        out.append(dict(W=W, T=T, s0=np.eye(n, dtype=np.float32)[pr["init"]]))
    return out
def build_blockdiag(progs):
    ns = [pr["n"] for pr in progs]; off = np.concatenate([[0], np.cumsum(ns)]); D = int(off[-1]); Ks = [pr["K"] for pr in progs]; koff = np.concatenate([[0], np.cumsum(Ks)])
    W = np.zeros((V, D, D), np.float32); T = np.zeros((int(koff[-1]), D), np.float32); s0 = np.zeros(D, np.float32)
    for p, pr in enumerate(progs):
        for v in range(V):
            for j in range(pr["n"]): W[v, off[p] + pr["delta"][v, j], off[p] + j] = 1.0
        for i in range(pr["n"]): T[koff[p] + pr["table"][i], off[p] + i] = 1.0
        s0[off[p] + pr["init"]] = 1.0
    return dict(W=W, T=T, s0=s0), off, koff
def run_bil(p, X):
    xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); s = jnp.broadcast_to(p["s0"], (X.shape[0], p["s0"].shape[0]))
    def f(s, x): z = jnp.einsum("bv,vij,bj->bi", x, p["W"], s); s2 = jnp.clip(2.0 * z - 0.5, 0.0, 1.0); return s2, s2 @ p["T"].T
    _, ys = jax.lax.scan(f, s, xs); return jnp.swapaxes(ys, 0, 1)                      # (B,L,K)
def noisy(P, s, key, rel=False):
    return {k: v + s * (jnp.sqrt(jnp.mean(v ** 2)) if rel else 1.0) * jax.random.normal(kk, v.shape) for (k, v), kk in zip(P.items(), jax.random.split(key, len(P)))}
def params_count(d): return int(sum(np.prod(np.shape(v)) for v in d.values()))
def indep_params(progs): return sum(params_count(p) for p in build_indep(progs))
def blockdiag_params(progs): return params_count(build_blockdiag(progs)[0])
def indep_nonzero(progs): return sum(int((np.asarray(v) != 0).sum()) for p in build_indep(progs) for v in p.values())
def tied_dictionary_params(progs):
    """parameter count of a role-dictionary weight-tied design that could express these programs: one n x n matrix per DISTINCT token-transition function (U of them)
    + per-program token->role gates + readouts.  U = number of distinct token maps (== 3 for the template majority programs: up/down/hold)."""
    nmax = max(pr["n"] for pr in progs); U = len({tuple(pr["delta"][v]) for pr in progs for v in range(V)}); P = len(progs)
    return U * nmax * nmax + P * V * U + P * 2 * nmax + P * nmax, U
def macs_per_step(kind, progs, dk=None):
    nmax = max(pr["n"] for pr in progs); P = len(progs)
    if kind == "indep": return int(np.mean([pr["n"] ** 2 + pr["K"] * pr["n"] for pr in progs]))     # one program is active
    if kind == "blockdiag": D = sum(pr["n"] for pr in progs); return D * D + sum(pr["K"] for pr in progs) * D     # all programs run on every token (dense W_v[D,D])
    if kind == "fwi": Kmax = max(pr["K"] for pr in progs); return 2 * dk * nmax + dk * (V + 2) + nmax * dk + Kmax * dk     # B s, B s', key products, F k, R r
class Evaluator:
    """per-program and joint exact-sequence accuracy for a baseline under static noise"""
    def __init__(self, progs, L, B=32, seed=0):
        rng = np.random.default_rng(seed); self.X = rng.choice(V, size=(B, L)); self.refs = [ref_run(pr, self.X) for pr in progs]; self.Xj = jnp.array(self.X); self.progs = progs
        self.ind = build_indep(progs); self.bd, self.off, self.koff = build_blockdiag(progs)
        self.f = jax.jit(run_bil)
    def acc_indep(self, plist): 
        ok = [(np.asarray(self.f(p, self.Xj)).argmax(-1) == r).all(1) for p, r in zip(plist, self.refs)]; return np.mean([o.mean() for o in ok]), np.stack(ok).all(0).mean()
    def acc_bd(self, p):
        lg = np.asarray(self.f(p, self.Xj)); ok = [(lg[..., self.koff[q]:self.koff[q + 1]].argmax(-1) == r).all(1) for q, r in enumerate(self.refs)]; return np.mean([o.mean() for o in ok]), np.stack(ok).all(0).mean()
SIGB = [float(x) for x in np.logspace(-3, 0.3, 14)]
def baseline_s50(ev, kind, rel=False, NS=3):
    pp, jj = {}, {}
    for s in SIGB:
        a, j = [], []
        for i in range(NS):
            if kind == "indep": r = ev.acc_indep([noisy({k: jnp.asarray(v) for k, v in p.items()}, s, jax.random.PRNGKey(100 * q + i), rel) for q, p in enumerate(ev.ind)])
            else: r = ev.acc_bd(noisy({k: jnp.asarray(v) for k, v in ev.bd.items()}, s, jax.random.PRNGKey(i), rel))
            a.append(r[0]); j.append(r[1])
        pp[str(s)] = a; jj[str(s)] = j
    return dict(per_program=s50(pp), joint=s50(jj))
