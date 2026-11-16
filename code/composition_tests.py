import json, time, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
import digital_compiler as dc
from scaling import s50
SIG = [float(x) for x in np.logspace(-3, -0.3, 13)]; NS = 4; N = 16
progs = [dc.majority(N), dc.dyck(N), dc.parity(), dc.mod_count(3), dc.mod_count(5), dc.mod_count(7), dc.mod_count(11), dc.mod_count(13)]
def noisy(P, s, key):
    ks = dict(zip(P.keys(), jax.random.split(key, len(P)))); return {k: v + s * jax.random.normal(ks[k], v.shape) for k, v in P.items()}
def emp(spec, rng):
    p, m = dc.compile_spec(spec); pr = [0.3, 0.3] + [0.4 / (spec.V - 2)] * (spec.V - 2)
    X = rng.choice(spec.V, size=(24, N - 1), p=pr); ref = [dc.reference(spec, list(r)) for r in X]
    exp = np.array(ref); exp = exp if exp.ndim == 3 else exp[..., None]
    f = jax.jit(lambda P: dc.run(P, m, jnp.array(X), spec.V))
    def acc(P):
        o = np.asarray(f(P)); o = o if o.ndim == 3 else o[..., None]; return float((o == exp).all((1, 2)).mean())
    return s50({str(s): [acc(noisy(p, s, jax.random.PRNGKey(k))) for k in range(NS)] for s in SIG}), acc(p)
res = {}; t0 = time.time(); rng = np.random.default_rng(1)
singles = {sp.name: emp(sp, rng)[0] for sp in progs}
print("single-program sigma50:", {k: f"{v:.1e}" for k, v in singles.items()}, flush=True)
for k in [1, 2, 3, 4, 6, 8]:
    C = dc.compose_parallel(progs[:k]) if k > 1 else progs[0]; p, m = dc.compile_spec(C)
    ok, tot = dc.certify_exact(C, p, m); v, clean = emp(C, rng)
    rec = {"k": k, "D": m["D"], "params": int(sum(np.prod(x.shape) for x in p.values())), "exact_cert": [ok, tot], "joint_s50": v, "clean": clean,
           "best_single_s50": max(singles[sp.name] for sp in progs[:k]), "worst_single_s50": min(singles[sp.name] for sp in progs[:k])}
    if k <= 4:
        eps, d = dc.certified_radius(C, p, m); rec["certified_radius"] = eps; rec["tube"] = d
    res[k] = rec
    print(f"k={k} D={m['D']:3d} params={rec['params']:6d} exact {ok}/{tot}  joint sigma50={v:.2e} (worst single {rec['worst_single_s50']:.1e})" + (f"  CERTIFIED eps*={rec['certified_radius']:.1e}" if k <= 4 else "") + f"  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open("composition_tests.json", "w"))
print("DONE", flush=True)
