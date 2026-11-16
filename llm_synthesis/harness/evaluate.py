"""Unified evaluation: the SAME procedures for every network that is evaluated on these tasks (Tracr-compiled, analytic, LLM-written, trained).
 clean exactness (hidden test set = project protocol; exhaustive gate for the transformer family) ; static Gaussian noise: absolute, relative, zero-only, nonzero-only, per parameter group, sigma grid
 10^(-5..0) (a superset of the project grid 10^(-5..-1)), 5 seeds, exact-sequence accuracy, sigma50 = log-interpolated 50% crossing ; interval-bound certificate (sample-certified radius, absolute box on ALL
 parameters) ; function-level L-inf attack with doubling+bisection (first failure found)."""
import numpy as np, jax, jax.numpy as jnp
import tasks as T, family as F
SIG = [float(x) for x in np.logspace(-5, 0, 16)]; SEEDS = 5
def s50(curve):
    xs = np.array([float(k) for k in curve]); ys = np.array([np.mean(v) for v in curve.values()])
    if ys[0] < 0.5: return float("nan")
    for i in range(1, len(xs)):
        if ys[i] < 0.5:
            f = (ys[i - 1] - 0.5) / (ys[i - 1] - ys[i]); return float(np.exp(np.log(xs[i - 1]) + f * (np.log(xs[i]) - np.log(xs[i - 1]))))
    return float("inf")
def _names(params):
    leaves, tree = jax.tree_util.tree_flatten_with_path(params)
    return ["/".join(str(getattr(k, "key", k)) for k in path) for path, _ in leaves], [l for _, l in leaves], tree
class FamilyEval:
    """a model of the permitted family, bound to a task"""
    def __init__(self, task, cfg, params):
        self.task, self.cfg, self.t = task, cfg, T.TASKS[task]; self.params = {k: jnp.asarray(v, jnp.float32) for k, v in params.items()}
        data = T.sample(task, "test"); self.lens = sorted(data); self.tok = {L: jnp.array(T.tokens_with_bos(data[L])) for L in self.lens}; self.exp = {L: T.expected(task, data[L]) for L in self.lens}
        self.exp_arr = {L: jnp.array(np.array(self.exp[L], np.float32 if self.t["kind"] == "num" else np.int32)) for L in self.lens}
        self._acc = jax.jit(lambda p: self._correct(p))
        self.group_pred = lambda g, name: F.GROUPS[g](name); self.groups = list(F.GROUPS)
    def _ok(self, out, exp):          # out (B,T,d_out) -> (B,) exact sequence correctness over positions 1..L
        o = out[:, 1:]
        if self.t["kind"] == "cat": return (jnp.argmax(o, -1) == exp).all(1)
        return (jnp.abs(o[..., 0] - exp) <= self.t["atol"]).all(1)
    def _correct(self, p): return jnp.concatenate([self._ok(F.forward(p, self.cfg, self.tok[L]), self.exp_arr[L]) for L in self.lens]).mean()
    def acc(self, p): return float(self._acc(p))
    def margins(self, p):             # (N,) min over positions of (cat: target logit - best other; num: atol - |y - target|)
        ms = []
        for L in self.lens:
            o = F.forward(p, self.cfg, self.tok[L])[:, 1:]; e = self.exp_arr[L]
            if self.t["kind"] == "cat":
                corr = jnp.take_along_axis(o, e[..., None], -1)[..., 0]; oth = jnp.where(jax.nn.one_hot(e, o.shape[-1]) > 0, -1e9, o).max(-1); m = corr - oth
            else: m = self.t["atol"] - jnp.abs(o[..., 0] - e)
            ms.append(m.min(1))
        return jnp.concatenate(ms)
    def exhaustive_clean(self, chunk=40000):
        wrong = tot = 0
        f = jax.jit(lambda p, tok, e: self._ok(F.forward(p, self.cfg, tok), e))
        for L, xs in T.exhaustive(self.task):
            exp = np.array(T.expected(self.task, xs), np.float32 if self.t["kind"] == "num" else np.int32)
            for i in range(0, len(xs), chunk):
                ok = np.asarray(f(self.params, jnp.array(T.tokens_with_bos(xs[i:i + chunk])), jnp.array(exp[i:i + chunk]))); wrong += int((~ok).sum()); tot += len(ok)
        return dict(all_correct=wrong == 0, n_wrong=wrong, n_total=tot)
def perturb(ev, params, sigma, key, mode, group=None):
    names, leaves, tree = _names(params); keys = jax.random.split(key, len(leaves)); out = []
    for nm, a, k in zip(names, leaves, keys):
        z = jax.random.normal(k, a.shape)
        if mode == "abs": b = a + sigma * z
        elif mode == "rel": b = a * (1 + sigma * z)
        elif mode == "zero_only": b = a + sigma * z * (a == 0)
        elif mode == "nonzero_only": b = a + sigma * z * (a != 0)
        elif mode == "group": b = a + sigma * z if ev.group_pred(group, nm) else a
        out.append(b)
    return jax.tree_util.tree_unflatten(tree, out)
def noise_suite(ev, sigmas=SIG, seeds=SEEDS, groups=True):
    modes = [("abs", None), ("rel", None), ("zero_only", None), ("nonzero_only", None)] + ([("group", g) for g in ev.groups] if groups else []); res = {}
    for mode, g in modes:
        cur = {str(s): [ev.acc(perturb(ev, ev.params, s, jax.random.PRNGKey(k), mode, g)) for k in range(seeds)] for s in sigmas}
        res[mode if g is None else f"group/{g}"] = dict(s50=s50(cur), curve={k: float(np.mean(v)) for k, v in cur.items()})
    return res
# ---------------------------------------------------------------- interval-bound certificate (family only)
def _ilin(xc, xr, W, eps, eq):
    c = jnp.einsum(eq, xc, W); r = jnp.einsum(eq, xr, jnp.abs(W)) + eps * jnp.einsum(eq, jnp.abs(xc) + xr, jnp.ones_like(W)); return c, r
def ibp_forward(params, cfg, tokens, eps):
    """interval of the outputs when EVERY parameter entry may move by +-eps (inputs exact). Returns (center, radius), each (B,T,d_out)."""
    P = params; T_ = tokens.shape[1]; dh = cfg["d_head"]; allowed = jnp.tril(jnp.ones((T_, T_), bool)) if cfg["causal"] else jnp.ones((T_, T_), bool)
    xc = P["embed"][tokens] + P["pos_embed"][:T_][None]; xr = jnp.full_like(xc, 2 * eps)
    for i in range(cfg["n_layers"]):
        g = lambda n: P[f"l{i}.{n}"]
        def proj(w, b):
            c, r = _ilin(xc, xr, g(w), eps, "btd,hde->bhte"); return c + g(b)[None, :, None], r + eps
        qc, qr = proj("wq", "bq"); kc, kr = proj("wk", "bk"); vc, vr = proj("wv", "bv")
        E = lambda a, b: jnp.einsum("bhte,bhse->bhts", a, b)
        sc = E(qc, kc) / jnp.sqrt(dh * 1.0); sr = (E(jnp.abs(qc), kr) + E(qr, jnp.abs(kc)) + E(qr, kr)) / jnp.sqrt(dh * 1.0); l, u = sc - sr, sc + sr; msk = allowed[None, None]
        m = jnp.max(jnp.where(msk, u, -1e30), -1, keepdims=True); el = jnp.where(msk, jnp.exp(l - m), 0.0); eu = jnp.where(msk, jnp.exp(u - m), 0.0); sl, su = el.sum(-1, keepdims=True), eu.sum(-1, keepdims=True)
        pl = el / (el + su - eu + 1e-30); ph = eu / (eu + sl - el + 1e-30); pl = jnp.where(msk, pl, 0.0); ph = jnp.where(msk, ph, 0.0)
        vl, vu = vc - vr, vc + vr; cs = jnp.stack([pl[..., None] * vl[:, :, None], pl[..., None] * vu[:, :, None], ph[..., None] * vl[:, :, None], ph[..., None] * vu[:, :, None]])   # (4,B,H,T,S,dh)
        ol, ou = cs.min(0).sum(3), cs.max(0).sum(3); oc, orad = (ol + ou) / 2, (ou - ol) / 2
        ac, ar = _ilin(oc, orad, g("wo"), eps, "bhte,hed->btd"); xc, xr = xc + ac + g("bo"), xr + ar + eps
        hc, hr = _ilin(xc, xr, g("w1"), eps, "btd,dm->btm"); hc, hr = hc + g("b1"), hr + eps; lo, hi = jnp.maximum(hc - hr, 0), jnp.maximum(hc + hr, 0); hc, hr = (lo + hi) / 2, (hi - lo) / 2
        mc, mr = _ilin(hc, hr, g("w2"), eps, "btm,md->btd"); xc, xr = xc + mc + g("b2"), xr + mr + eps
    yc, yr = _ilin(xc, xr, P["unembed.w"], eps, "btd,do->bto"); return yc + P["unembed.b"], yr + eps
def certified(ev, eps):
    """True iff every sampled (sequence, position) keeps its correct output for ALL perturbations in the box (sample-certified, not a domain-wide proof)"""
    for L in ev.lens:
        yc, yr = ibp_forward(ev.params, ev.cfg, ev.tok[L], eps); yc, yr = yc[:, 1:], yr[:, 1:]; e = ev.exp_arr[L]
        if ev.t["kind"] == "cat":
            lo, hi = yc - yr, yc + yr; corr = jnp.take_along_axis(lo, e[..., None], -1)[..., 0]; oth = jnp.where(jax.nn.one_hot(e, yc.shape[-1]) > 0, -1e9, hi).max(-1); ok = (corr > oth).all()
        else: ok = ((yc[..., 0] - yr[..., 0] >= e - ev.t["atol"]) & (yc[..., 0] + yr[..., 0] <= e + ev.t["atol"])).all()
        if not bool(ok): return False
    return True
def cert_radius(ev, lo=1e-9, hi=1.0, it=26):
    if not certified(ev, lo): return 0.0
    if certified(ev, hi): return hi
    for _ in range(it):
        mid = float(np.sqrt(lo * hi)); lo, hi = (mid, hi) if certified(ev, mid) else (lo, mid)
    return lo
# ---------------------------------------------------------------- function-level L-inf attack
def attack_margin(ev, eps, steps=30, restarts=1, seed=0):
    rng = np.random.default_rng(seed); names, leaves, tree = _names(ev.params); best = 1e9
    if not hasattr(ev, "_atk"):      # performance-only change (v2): compile the attack functions once per model instead of once per call; numerics identical
        obj = lambda d: -0.05 * jax.nn.logsumexp(-ev.margins({k: ev.params[k] + d[k] for k in ev.params}) / 0.05)
        ev._atk = (jax.jit(jax.value_and_grad(obj)), jax.jit(lambda d: ev.margins({k: ev.params[k] + d[k] for k in ev.params}).min()))
    g, mj = ev._atk
    for r in range(restarts):
        d = {k: eps * jnp.asarray(rng.choice([-1.0, 1.0], size=v.shape), jnp.float32) for k, v in ev.params.items()}
        for t in range(steps):
            _, gr = g(d); d = {k: jnp.clip(d[k] - eps / 4 * jnp.sign(jnp.nan_to_num(gr[k])), -eps, eps) for k in d}
        best = min(best, float(mj(d)))
    return best
def attack_break(ev, eps0, grow=2.0, emax=2.0, bis=6, **kw):
    """smallest L-inf budget at which the attack finds a wrong output (upper bound on the true robustness radius); None if none up to emax"""
    if attack_margin(ev, eps0, **kw) <= 0: return eps0
    lo, hi, e = eps0, None, eps0
    while e < emax:
        e *= grow
        if attack_margin(ev, e, **kw) <= 0: hi = e; break
        lo = e
    if hi is None: return None
    for _ in range(bis):
        mid = float(np.sqrt(lo * hi)); lo, hi = (lo, mid) if attack_margin(ev, mid, **kw) <= 0 else (mid, hi)
    return hi
def param_stats(params):
    a = np.concatenate([np.asarray(v).ravel() for v in jax.tree_util.tree_leaves(params)]); nz = a != 0
    return dict(n_params=int(a.size), nonzero_frac=float(nz.mean()), max_abs=float(np.abs(a).max()), mean_abs_nonzero=float(np.abs(a[nz]).mean()) if nz.any() else 0.0)
