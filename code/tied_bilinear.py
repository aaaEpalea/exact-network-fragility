"""Weight-tied + bilinear: z_i = sum_r gate_r(x) * (P_r s)_i , gate_r = sum_v G[k,r,v] x_v.  No hidden gates, no biases."""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
import digital_compiler as dc
from zeros_empirical import majority_ab
from scaling import s50
V = 6; SIG = [float(x) for x in np.logspace(-3, 0.5, 18)]; NS = 6; N = 16
def build(N, pairs):
    n = 2 * N + 1; Pu = np.zeros((n, n)); Pu[np.arange(1, n), np.arange(n - 1)] = 1; Pu[n - 1, n - 1] = 1; Pd = np.zeros((n, n)); Pd[np.arange(n - 1), np.arange(1, n)] = 1; Pd[0, 0] = 1
    G = np.zeros((len(pairs), 3, V))
    for j, (a, b) in enumerate(pairs):
        for t in range(V): G[j, 0 if t == a else 1 if t == b else 2, t] = 1
    T = np.zeros((2, n)); T[0, :N + 1] = 1; T[1, N + 1:] = 1
    return {"P": jnp.array(np.stack([Pu, Pd, np.eye(n)]), jnp.float32), "G": jnp.array(G, jnp.float32), "T": jnp.array(T, jnp.float32), "s0": jnp.array(np.eye(n)[N], jnp.float32)}
def run(p, X, k):
    xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); s = jnp.broadcast_to(p["s0"], (X.shape[0], k, p["s0"].shape[0]))
    def f(s, x):
        g = jnp.einsum("krv,bv->bkr", p["G"], x); z = jnp.einsum("bkr,rij,bkj->bki", g, p["P"], s); s2 = jnp.clip(2.0 * z - 0.5, 0.0, 1.0)
        return s2, jnp.argmax(s2 @ p["T"].T, -1)
    _, ys = jax.lax.scan(f, s, xs); return jnp.swapaxes(ys, 0, 1)
noisy = lambda P, s, key, rel=False: {k: v + s * (jnp.sqrt(jnp.mean(v ** 2)) if rel else 1.0) * jax.random.normal(kk, v.shape) for (k, v), kk in zip(P.items(), jax.random.split(key, len(P)))}
rng = np.random.default_rng(0); res = {}; allpairs = [(a, b) for a in range(V) for b in range(V) if a != b]
for k in [1, 4, 15, 30]:
    pairs = allpairs[:k]; p = build(N, pairs); specs = [majority_ab(N, a, b) for a, b in pairs]
    X = rng.choice(V, size=(48, N - 1)); ref = np.stack([np.array([dc.reference(sp, list(r)) for r in X]) for sp in specs], -1); Xj = jnp.array(X)
    f = jax.jit(lambda Q: run(Q, Xj, k)); ok = lambda Q: np.asarray(f(Q)) == ref; clean = float(ok(p).all((1, 2)).mean())
    out = {"clean": clean, "params": int(sum(np.prod(v.shape) for v in p.values()))}
    for rel in [False, True]:
        ph, jt = {}, {}
        for s in SIG:
            v = [ok(noisy(p, s, jax.random.PRNGKey(i), rel)) for i in range(NS)]; ph[str(s)] = [float(o.all(1).mean()) for o in v]; jt[str(s)] = [float(o.all((1, 2)).mean()) for o in v]
        out["rel" if rel else "abs"] = {"per_head": s50(ph), "joint": s50(jt)}
    res[f"k{k}"] = out
    print(f"TIED-BILINEAR k={k:2d}: clean={clean:.2f} params={out['params']:5d} | sigma50 per-head {out['abs']['per_head']:.2e} joint {out['abs']['joint']:.2e} | alpha50 per-head {out['rel']['per_head']:.2e} joint {out['rel']['joint']:.2e}", flush=True)
    json.dump(res, open("tied_bilinear.json", "w"))
