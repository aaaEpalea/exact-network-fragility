import json, sys, time, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
import digital_compiler as dc
from scaling import s50
SIG = [float(x) for x in np.logspace(-3, -0.3, 13)]; NS = 4
def noisy(P, s, key):
    ks = dict(zip(P.keys(), jax.random.split(key, len(P)))); return {k: v + s * jax.random.normal(ks[k], v.shape) for k, v in P.items()}
def seqs(spec, N, rng, B=24):
    pr = [0.3, 0.3] + [0.4 / (spec.V - 2)] * (spec.V - 2); return rng.choice(spec.V, size=(B, N - 1), p=pr)
res = {}; t0 = time.time()
# ---- 1+2: certificates
suites = [(dc.prefix_hist, [4, 8, 16]), (dc.majority, [4, 8, 16, 32]), (dc.dyck, [4, 8, 16, 32]), (dc.parity, [None]), (lambda: dc.mod_count(3), [None])]
for mk, Ns in suites:
    for N in Ns:
        spec = mk(N) if N is not None else mk(); p, meta = dc.compile_spec(spec)
        ok, tot = dc.certify_exact(spec, p, meta)
        rng = np.random.default_rng(0); L = 3 * (N or 8); X = rng.choice(spec.V, size=(200, L)); got = np.asarray(dc.run(p, meta, jnp.array(X), spec.V))
        agree = float(np.mean([list(g) == dc.reference(spec, list(x)) for g, x in zip(got, X)]))
        rec = {"exact_cert": [ok, tot], "random_agree": agree, "params": int(sum(np.prod(v.shape) for v in p.values()))}
        small = (meta["D"] <= 40) or (spec.name.startswith("prefix_hist") and N <= 4)
        if small:
            eps, d = dc.certified_radius(spec, p, meta); rec["certified_radius"] = eps; rec["tube"] = d
        res[spec.name] = rec
        print(f"{spec.name:26s} exact certificate {ok}/{tot}  random-seq agreement {agree:.3f}  params={rec['params']}" + (f"  CERTIFIED eps*={rec['certified_radius']:.2e} (tube {rec['tube']})" if small else "") + f"  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open("compiler_tests.json", "w"))
# ---- 3: empirical noise tolerance vs N
for mk, name in [(dc.prefix_hist, "prefix_hist"), (dc.majority, "majority"), (dc.dyck, "dyck_valid_prefix")]:
    for N in [8, 16, 32, 64]:
        spec = mk(N); p, meta = dc.compile_spec(spec); rng = np.random.default_rng(1)
        X = seqs(spec, N, rng); exp = np.array([dc.reference(spec, list(r)) for r in X]); Xj = jnp.array(X)
        f = jax.jit(lambda P: dc.run(P, meta, Xj, spec.V)); acc = lambda P: float((np.asarray(f(P)) == exp).all(1).mean())
        clean = acc(p); v = s50({str(s): [acc(noisy(p, s, jax.random.PRNGKey(k))) for k in range(NS)] for s in SIG})
        res[f"emp|{name}|{N}"] = {"clean": clean, "s50": v}; print(f"empirical {name:18s} N={N:3d} clean={clean:.2f} sigma50={v:.2e} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open("compiler_tests.json", "w"))
print("DONE", flush=True)
