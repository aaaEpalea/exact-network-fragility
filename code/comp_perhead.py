"""Composition: separate crosstalk from the 'all heads must be right' statistical effect (per-head accuracy), plus composite certificates."""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
import digital_compiler as dc
from zeros_empirical import majority_ab, dyck_ab, noise
from scaling import s50
V = 6; N = 16; SIG = [float(x) for x in np.logspace(-3, 0, 15)]; NS = 6
pool = [majority_ab(N, 0, 1), dyck_ab(N, 0, 1), majority_ab(N, 2, 3), dyck_ab(N, 2, 3), majority_ab(N, 4, 5), dyck_ab(N, 4, 5), majority_ab(N, 0, 2), dyck_ab(N, 1, 3)]
rng = np.random.default_rng(0); X = rng.choice(V, size=(48, N - 1)); Xj = jnp.array(X)
res = {}; t0 = time.time()
for k in [1, 2, 4, 8]:
    spec = pool[0] if k == 1 else dc.compose_parallel(pool[:k]); p, meta = dc.compile_spec(spec); masks = {a: (v != 0).astype(jnp.float32) for a, v in p.items()}
    ref = np.array([dc.reference(spec, list(r)) for r in X]).reshape(48, N - 1, -1); f = jax.jit(lambda P: dc.run(P, meta, Xj, V))
    def accs(P):
        out = np.asarray(f(P)).reshape(48, N - 1, -1); per_head = (out == ref).all(1).mean(0)        # (heads,) exact-sequence accuracy per head
        return float(per_head.mean()), float((out == ref).all((1, 2)).mean())
    for mode in ["all", "nonzero"]:
        mean_c, joint_c = {}, {}
        for s in SIG:
            vals = [accs(noise(p, s, jax.random.PRNGKey(i), mode, masks)) for i in range(NS)]; mean_c[str(s)] = [v[0] for v in vals]; joint_c[str(s)] = [v[1] for v in vals]
        res[f"k{k}|{mode}"] = {"per_head_s50": s50(mean_c), "joint_s50": s50(joint_c)}
        print(f"k={k} noise on {mode:8s}: sigma50 per-head (mean over heads)={res[f'k{k}|{mode}']['per_head_s50']:.2e}   joint (all heads exact)={res[f'k{k}|{mode}']['joint_s50']:.2e}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open("comp_perhead.json", "w"))
print("DONE", flush=True)
