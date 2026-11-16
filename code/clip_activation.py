"""Restoring step as a weight-free ACTIVATION  s' = clip(2*z - 0.5, 0, 1)  instead of 4 dense noisy matrices."""
import json, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
import digital_compiler as dc
from zeros_empirical import majority_ab, dyck_ab
from scaling import s50
V = 6; SIG = [float(x) for x in np.logspace(-3, 0.3, 17)]; NS = 6
def run_clip(p, X, nheads=1):
    xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); s = jnp.broadcast_to(p["s0"], (X.shape[0], p["s0"].shape[0]))
    def f(s, x):
        h = jax.nn.relu(s @ p["A"].T + x @ p["B"].T + p["b"]); z = h @ p["C"].T
        s2 = jnp.clip(2.0 * z - 0.5, 0.0, 1.0)                       # weight-free restoring activation
        return s2, jnp.argmax(s2 @ p["T0"].T, -1)
    _, ys = jax.lax.scan(f, s, xs); return jnp.swapaxes(ys, 0, 1)
def noisy(P, s, key, rel=False):
    ks = jax.random.split(key, len(P)); return {k: v + s * (jnp.sqrt(jnp.mean(v ** 2)) if rel else 1.0) * jax.random.normal(kk, v.shape) for (k, v), kk in zip(P.items(), ks)}
res = {}; rng = np.random.default_rng(0)
for spec, N in [(majority_ab(16, 0, 1), 16), (dyck_ab(16, 0, 1), 16), (majority_ab(32, 0, 1), 32), (dyck_ab(32, 0, 1), 32)]:
    p, meta = dc.compile_spec(spec); pc = {k: p[k] for k in ["A", "B", "b", "C", "T0", "s0"]}
    X = rng.choice(V, size=(24, N - 1)); ref = np.array([dc.reference(spec, list(r)) for r in X]); Xj = jnp.array(X)
    f_old = jax.jit(lambda Q: dc.run(Q, meta, Xj, V)); f_new = jax.jit(lambda Q: run_clip(Q, Xj))
    acc_o = lambda Q: float((np.asarray(f_old(Q)) == ref).all(1).mean()); acc_n = lambda Q: float((np.asarray(f_new(Q)) == ref).all(1).mean())
    clean_n = acc_n(pc)
    r = {"clean_new": clean_n, "params_old": int(sum(np.prod(v.shape) for v in p.values())), "params_new": int(sum(np.prod(v.shape) for v in pc.values()))}
    for rel in [False, True]:
        grid = SIG
        so = s50({str(s): [acc_o(noisy(p, s, jax.random.PRNGKey(k), rel)) for k in range(NS)] for s in grid})
        sn = s50({str(s): [acc_n(noisy(pc, s, jax.random.PRNGKey(k), rel)) for k in range(NS)] for s in grid})
        r["rel" if rel else "abs"] = {"snap_matrices": so, "clip_activation": sn, "gain": sn / so}
    res[spec.name + f"|N{N}"] = r
    print(f"{spec.name:12s} N={N:2d}: clean(new)={clean_n:.2f} params {r['params_old']}->{r['params_new']} | sigma50 snap-matrices {r['abs']['snap_matrices']:.2e} -> clip-activation {r['abs']['clip_activation']:.2e} ({r['abs']['gain']:.1f}x) | alpha50 {r['rel']['snap_matrices']:.2e} -> {r['rel']['clip_activation']:.2e} ({r['rel']['gain']:.1f}x)", flush=True)
    json.dump(res, open("clip_activation.json", "w"))
