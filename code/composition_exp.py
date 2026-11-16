"""Parallel composition: k programs in ONE dense network. Exactness certificate, certified radius, empirical noise tolerance vs k."""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
import digital_compiler as dc
from digital_compiler import FSM, Spec, sat
from scaling import s50
SIG = [float(x) for x in np.logspace(-3, -0.3, 13)]; NS = 4; N = 16; V = 6

def majority_ab(N, a, b):
    n = 2 * N + 1; delta = np.array([[sat(n, i, 1) if t == a else sat(n, i, -1) if t == b else i for i in range(n)] for t in range(V)])
    return Spec(V, [FSM(n, N, delta)], {"type": "table", "comp": 0, "table": [int(i > N) for i in range(n)], "n_out": 2}, f"maj({a}>{b})")
def dyck_ab(N, a, b):
    n = N + 2; dead = N + 1; rows = []
    for t in range(V):
        rows.append([dead if i == dead else (min(i + 1, N) if t == a else (i - 1 if i > 0 else dead) if t == b else i) for i in range(n)])
    return Spec(V, [FSM(n, 0, np.array(rows))], {"type": "table", "comp": 0, "table": [1] * (N + 1) + [0], "n_out": 2}, f"dyck({a},{b})")
pool = [majority_ab(N, 0, 1), dyck_ab(N, 0, 1), majority_ab(N, 2, 3), dyck_ab(N, 2, 3), majority_ab(N, 4, 5), dyck_ab(N, 4, 5), majority_ab(N, 0, 2), dyck_ab(N, 1, 3)]
noisy = lambda P, s, key: {k: v + s * jax.random.normal(jax.random.split(key, len(P))[i], v.shape) for i, (k, v) in enumerate(P.items())}
def emp(spec, p, meta, rng):
    pr = [1 / V] * V; X = rng.choice(V, size=(24, N - 1), p=pr); ref = np.array([dc.reference(spec, list(r)) for r in X]).reshape(24, N - 1, -1); Xj = jnp.array(X)
    f = jax.jit(lambda P: dc.run(P, meta, Xj, V)); acc = lambda P: float((np.asarray(f(P)).reshape(24, N - 1, -1) == ref).all((1, 2)).mean())
    return acc(p), s50({str(s): [acc(noisy(p, s, jax.random.PRNGKey(k))) for k in range(NS)] for s in SIG})
if __name__ == '__main__':
    res = {}; t0 = time.time(); rng = np.random.default_rng(0)
    for k in [1, 2, 4, 8]:
        spec = pool[0] if k == 1 else dc.compose_parallel(pool[:k]); p, meta = dc.compile_spec(spec)
        ok, tot = dc.certify_exact(spec, p, meta); clean, v = emp(spec, p, meta, rng)
        rec = {"k": k, "D": meta["D"], "params": int(sum(np.prod(a.shape) for a in p.values())), "exact_cert": [ok, tot], "clean": clean, "s50": v}
        if meta["D"] <= 120:
            eps, d = dc.certified_radius(spec, p, meta); rec["certified_radius"] = eps
        res[f"k={k}"] = rec
        print(f"k={k}: D={rec['D']:3d} params={rec['params']:6d} exact cert {ok}/{tot} clean={clean:.2f} empirical sigma50={v:.2e}" + (f" certified eps*={rec['certified_radius']:.2e}" if 'certified_radius' in rec else "") + f" ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open("composition_exp.json", "w"))
    print("DONE", flush=True)
