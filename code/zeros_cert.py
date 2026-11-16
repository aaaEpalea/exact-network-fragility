"""Certified radius with zeros PROTECTED vs unprotected, and PGD validation of the protected certificate."""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
import digital_compiler as dc
DELTAS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2)
def refined(spec, protect):
    p, meta = dc.compile_spec(spec)
    ok = lambda e: next((d for d in DELTAS if e <= d and dc.tube_ok(spec, p, meta, e, d, protect)), None)
    lo = hi = None
    for e in np.logspace(-5, -0.5, 10):
        if ok(e) is not None: lo = e
        else: hi = e; break
    if lo is None: return 0.0, None, meta["D"]
    if hi is None: return lo, ok(lo), meta["D"]
    for _ in range(7):
        mid = float(np.sqrt(lo * hi))
        if ok(mid) is not None: lo = mid
        else: hi = mid
    return lo, ok(lo), meta["D"]
res = {}; t0 = time.time()
for fam, mk, Ns in [("majority", dc.majority, [8, 16, 32, 64]), ("dyck", dc.dyck, [8, 16, 32, 64])]:
    for N in Ns:
        e0, d0, D = refined(mk(N), False); e1, d1, _ = refined(mk(N), True)
        res[f"{fam}|{N}"] = {"D": D, "dense_eps": e0, "dense_tube": d0, "protected_eps": e1, "protected_tube": d1}
        print(f"{fam:9s} N={N:3d} D={D:4d}  certified eps*: all-entries-noisy {e0:.2e} (tube {d0})   zeros-protected {e1:.2e} (tube {d1})   ratio {e1/e0:.1f}x  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open("zeros_cert.json", "w"))
# PGD validation of the protected certificate (perturb ONLY nonzero entries)
PERT = ["A", "B", "b", "C", "S1", "bs1", "S2", "bs2", "C3", "C4", "T0", "I3", "Bt", "b3", "R"]
def attack(spec, eps, delta, steps=150, restarts=4, seed=0):
    p, meta = dc.compile_spec(spec); V = spec.V; cfgs = list(dc._configs(spec))
    S = jnp.array(np.stack([c[1] for c in cfgs]), jnp.float32); Xo = jnp.array(np.stack([np.eye(V)[c[0]] for c in cfgs]), jnp.float32)
    ES = jnp.array(np.stack([c[2] for c in cfgs]), jnp.float32); EY = jnp.array(np.array([c[3] for c in cfgs]), jnp.int32)
    keys = [k for k in p if k in PERT]; masks = {k: (p[k] != 0).astype(jnp.float32) for k in keys}
    def viol(dp_, eta):
        q = {k: (p[k] + dp_[k] if k in dp_ else p[k]) for k in p}; s2, ys = dc.step(q, meta, S + eta, Xo); v = jnp.abs(s2 - ES).max(-1) - delta
        for h, y in enumerate(ys):
            ey = EY[:, h] if (len(ys) > 1 or meta["type"] != "selected") else EY[:, 0]
            correct = jnp.take_along_axis(y, ey[:, None], 1)[:, 0]; masked = y.at[jnp.arange(y.shape[0]), ey].set(-1e9); v = jnp.maximum(v, masked.max(-1) - correct)
        return v
    g = jax.jit(jax.value_and_grad(lambda a, b: jax.nn.logsumexp(viol(a, b) / 0.02) * 0.02, argnums=(0, 1))); vj = jax.jit(viol)
    rng = np.random.default_rng(seed); worst = -1e9
    for r in range(restarts):
        dp_ = {k: eps * masks[k] * jnp.asarray(rng.choice([-1.0, 1.0], size=p[k].shape), jnp.float32) for k in keys}
        eta = delta * jnp.asarray(rng.choice([-1.0, 1.0], size=S.shape), jnp.float32)
        for t in range(steps):
            _, (gp, ge) = g(dp_, eta)
            dp_ = {k: jnp.clip(dp_[k] + (eps / 8) * jnp.sign(gp[k]), -eps, eps) * masks[k] for k in keys}; eta = jnp.clip(eta + (delta / 8) * jnp.sign(ge), -delta, delta)
        worst = max(worst, float(vj(dp_, eta).max()))
    return worst
for name, spec in [("majority(N=16)", dc.majority(16)), ("dyck(N=16)", dc.dyck(16)), ("majority(N=32)", dc.majority(32))]:
    key = f"{'majority' if name.startswith('maj') else 'dyck'}|{name.split('=')[1][:-1]}"; e, d = res[key]["protected_eps"], res[key]["protected_tube"]
    row = {m: max(attack(spec, m * e, d, seed=s) for s in range(2)) for m in [1.0, 1.1, 1.25, 1.5]}
    res[f"attack|{name}"] = {str(k): v for k, v in row.items()}
    print(f"PGD attack on PROTECTED certificate {name} (eps*={e:.2e}, tube {d}): " + "  ".join(f"{m}x:{v:+.4f}" for m, v in row.items()), flush=True)
    json.dump(res, open("zeros_cert.json", "w"))
print("DONE", flush=True)
