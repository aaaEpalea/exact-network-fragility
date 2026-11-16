"""Certified training of the digital network: optimise weights (init = compiled) to maximise a PROVABLE parameter-noise radius (general IBP, valid for any weights)."""
import json, sys, time, warnings
import numpy as np, jax, jax.numpy as jnp, optax
warnings.filterwarnings("ignore")
import digital_compiler as dc
from zeros_empirical import majority_ab, dyck_ab
from scaling import s50
KEYS = ["A", "B", "b", "C", "S1", "bs1", "S2", "bs2", "C3", "C4", "T0"]

def imat(W, lo, hi, e):                       # W (R,C); lo,hi (M,C) -> (M,R) interval of W' v, W' in [W-e, W+e]
    a, b = W - e, W + e
    c = jnp.stack([a[None] * lo[:, None, :], a[None] * hi[:, None, :], b[None] * lo[:, None, :], b[None] * hi[:, None, :]])
    return c.min(0).sum(-1), c.max(0).sum(-1)

def build(spec):
    p, meta = dc.compile_spec(spec); cfgs = list(dc._configs(spec)); V = spec.V
    S = jnp.array(np.stack([c[1] for c in cfgs]), jnp.float32); X = jnp.array(np.stack([np.eye(V)[c[0]] for c in cfgs]), jnp.float32)
    ES = jnp.array(np.stack([c[2] for c in cfgs]), jnp.float32); EY = jnp.array(np.array([c[3] for c in cfgs]), jnp.int32)
    def viol(P, eps, delta):                  # general IBP, per-config violation (<0 means certified for that config)
        hl, hu = imat(P["A"], S - delta, S + delta, eps); bl, bu = imat(P["B"], X, X, eps)
        hl, hu = jax.nn.relu(hl + bl + P["b"] - eps), jax.nn.relu(hu + bu + P["b"] + eps)
        zl, zu = imat(P["C"], hl, hu, eps)
        u1l, u1u = imat(P["S1"], zl, zu, eps); u2l, u2u = imat(P["S2"], zl, zu, eps)
        a1l, a1u = jax.nn.relu(u1l + P["bs1"] - eps), jax.nn.relu(u1u + P["bs1"] + eps)
        a2l, a2u = jax.nn.relu(u2l + P["bs2"] - eps), jax.nn.relu(u2u + P["bs2"] + eps)
        c3l, c3u = imat(P["C3"], a1l, a1u, eps); c4l, c4u = imat(P["C4"], a2l, a2u, eps); s2l, s2u = c3l + c4l, c3u + c4u
        v = jnp.maximum(s2u - (ES + delta), (ES - delta) - s2l).max(-1)
        yl, yu = imat(P["T0"], s2l, s2u, eps); correct = jnp.take_along_axis(yl, EY[:, :1], 1)[:, 0]
        masked = yu.at[jnp.arange(yu.shape[0]), EY[:, 0]].set(-1e9); return jnp.maximum(v, masked.max(-1) - correct)
    return p, meta, viol

def max_eps(viol, P, deltas):                 # largest eps (log bisection) certified for some delta
    best = (0.0, None)
    for d in deltas:
        lo, hi = 1e-7, 0.2
        if float(viol(P, lo, d).max()) >= 0: continue
        for _ in range(30):
            mid = float(np.sqrt(lo * hi))
            if float(viol(P, mid, d).max()) < 0: lo = mid
            else: hi = mid
        if lo > best[0]: best = (lo, d)
    return best

def noisy(P, s, key, rel=False):
    ks = jax.random.split(key, len(P)); return {k: v + s * (jnp.sqrt(jnp.mean(v ** 2)) if rel else 1.0) * jax.random.normal(kk, v.shape) for (k, v), kk in zip(P.items(), ks)}

def run(spec, N, steps=2500, delta_train=0.05):
    t0 = time.time(); p, meta, viol = build(spec); P0 = {k: p[k] for k in KEYS}; vj = jax.jit(viol)
    e_init, d_init = max_eps(lambda P, e, d: vj({**p, **P}, e, d), P0, [0.02, 0.03, 0.05, 0.08, 0.1])
    print(f"[{spec.name}] general-IBP certified radius of the COMPILED weights: {e_init:.2e} (tube {d_init})   ({time.time()-t0:.0f}s)", flush=True)
    P = dict(P0); eps = max(e_init, 1e-5); opt = optax.adam(2e-3); st = opt.init(P)
    def loss(Pp, eps_): return jnp.mean(jax.nn.relu(viol({**p, **Pp}, eps_, delta_train) + 0.01))
    gfn = jax.jit(jax.value_and_grad(loss)); hist = []
    for i in range(steps):
        l, g = gfn(P, eps); u, st = opt.update(g, st, P); P = optax.apply_updates(P, u)
        if i % 25 == 0:
            hist.append((i, float(l), float(eps)))
            if float(l) < 2e-4 and eps < 0.2: eps *= 1.2           # curriculum: grow the target radius once certified
        if i % 500 == 0: print(f"   step {i}: hinge loss {float(l):.5f}  target eps {eps:.2e}  ({time.time()-t0:.0f}s)", flush=True)
    Pf = {**p, **P}; e_fin, d_fin = max_eps(lambda Pp, e, d: vj(Pp, e, d), Pf, [0.02, 0.03, 0.05, 0.08, 0.1])
    # empirical Gaussian tolerance of the trained network vs compiled
    rng = np.random.default_rng(0); X = rng.choice(6, size=(24, N - 1)); ref = np.array([dc.reference(spec, list(r)) for r in X]); Xj = jnp.array(X)
    f = jax.jit(lambda Q: dc.run(Q, meta, Xj, 6)); acc = lambda Q: float((np.asarray(f(Q)) == ref).all(1).mean())
    SIG = [float(x) for x in np.logspace(-3, 0.3, 17)]
    emp = lambda Q, rel: s50({str(s): [acc(noisy(Q, s, jax.random.PRNGKey(k), rel)) for k in range(4)] for s in SIG})
    Lr = rng.choice(6, size=(64, 400)); refl = np.array([dc.reference(spec, list(r)) for r in Lr]); long_ok = float((np.asarray(jax.jit(lambda Q: dc.run(Q, meta, jnp.array(Lr), 6))(Pf)) == refl).all(1).mean())
    out = {"cert_compiled": e_init, "cert_trained": e_fin, "tube_trained": d_fin, "clean": acc(Pf), "clean_len400": long_ok,
           "sigma50_compiled": emp(p, False), "sigma50_trained": emp(Pf, False), "alpha50_compiled": emp(p, True), "alpha50_trained": emp(Pf, True)}
    print(f"== [{spec.name}] certified eps: {e_init:.2e} -> {e_fin:.2e} (tube {d_fin}); clean acc {out['clean']:.2f}, length-400 acc {long_ok:.2f}\n"
          f"   empirical sigma50: {out['sigma50_compiled']:.2e} -> {out['sigma50_trained']:.2e};  relative alpha50: {out['alpha50_compiled']:.2e} -> {out['alpha50_trained']:.2e}  ({time.time()-t0:.0f}s)", flush=True)
    return out

if __name__ == "__main__":
    res = {}
    for spec, N in [(majority_ab(16, 0, 1), 16), (dyck_ab(16, 0, 1), 16)]:
        res[spec.name] = run(spec, N); json.dump(res, open("ibp_train.json", "w"))
    print("DONE", flush=True)
