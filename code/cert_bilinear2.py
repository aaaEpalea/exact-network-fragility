"""Tighter certificates (fine tube search) + FUNCTION-LEVEL adversarial attack (one fixed perturbation, long sequences) for the bilinear designs."""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
import digital_compiler as dc
from zeros_empirical import majority_ab, dyck_ab
from bilinear_lib import build_bil, build_tied
from cert_bilinear import ok_bil, ok_tied
V = 6

def feasible(ok, eps):
    for d in np.linspace(max(eps, 0.01), 0.49, 40):
        if ok(eps, float(d)): return float(d)
    return None

def best_radius(ok):
    lo, hi = 1e-4, 0.45
    if feasible(ok, lo) is None: return 0.0, None
    if feasible(ok, hi) is not None: return hi, feasible(ok, hi)
    for _ in range(14):
        mid = float(np.sqrt(lo * hi))
        if feasible(ok, mid) is not None: lo = mid
        else: hi = mid
    return lo, feasible(ok, lo)

def run_logits(p, X):
    xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); s = jnp.broadcast_to(p["s0"], (X.shape[0], p["s0"].shape[0]))
    def f(s, x):
        z = jnp.einsum("bv,vij,bj->bi", x, p["W"], s); s2 = jnp.clip(2.0 * z - 0.5, 0.0, 1.0); return s2, s2 @ p["T"].T
    _, ys = jax.lax.scan(f, s, xs); return jnp.swapaxes(ys, 0, 1)

def func_attack(spec, p, eps, protect, steps=120, restarts=3, seed=0):
    rng = np.random.default_rng(seed); N = (p["s0"].shape[0] - 1) // 2 if spec.name.startswith("maj") else p["s0"].shape[0]
    L = 4 * N; X = rng.choice(V, size=(96, L)); X[0] = 0; X[1] = 1                       # + constant 'up' and constant 'down' sequences
    ref = jnp.array([dc.reference(spec, list(r)) for r in X]); Xj = jnp.array(X)
    masks = {k: ((p[k] != 0).astype(jnp.float32) if protect else jnp.ones_like(p[k])) for k in p}
    def margins(delta):
        q = {k: p[k] + delta[k] for k in p}; lg = run_logits(q, Xj)
        corr = jnp.take_along_axis(lg, ref[..., None], -1)[..., 0]; oth = jnp.where(jax.nn.one_hot(ref, lg.shape[-1]) > 0, -1e9, lg).max(-1); return corr - oth
    obj = lambda d: -0.05 * jax.nn.logsumexp(-margins(d).reshape(-1) / 0.05)               # softmin of margins (minimise)
    g = jax.jit(jax.value_and_grad(obj)); mj = jax.jit(margins); best = 1e9
    for r in range(restarts):
        d = {k: eps * masks[k] * jnp.asarray(rng.choice([-1., 1.], size=p[k].shape), jnp.float32) for k in p}
        for t in range(steps):
            _, gr = g(d); d = {k: jnp.clip(d[k] - eps / 6 * jnp.sign(gr[k]), -eps, eps) * masks[k] for k in p}
        best = min(best, float(mj(d).min()))
    return best                                                                            # <= 0 means a real wrong output was found

if __name__ == "__main__":
    res = {}; t0 = time.time()
    for fam, mk in [("majority", majority_ab), ("dyck", dyck_ab)]:
        for N in [8, 16, 32]:
            spec = mk(N, 0, 1); p = build_bil(spec)
            for protect in [False, True]:
                e, d = best_radius(lambda e_, d_: ok_bil(spec, p, e_, d_, protect)); res[f"bilinear|{fam}|{N}|{'protected' if protect else 'dense'}"] = {"fine_eps": e, "tube": d}
                print(f"BILINEAR {fam:8s} N={N:2d} {'zeros-protected' if protect else 'dense':15s}: fine certified eps* = {e:.4f} (tube {d:.3f})  ({time.time()-t0:.0f}s)", flush=True); json.dump(res, open("cert_bilinear2.json", "w"))
    for N in [16, 32]:
        p = build_tied(N, [(0, 1)])
        for protect in [False, True]:
            e, d = best_radius(lambda e_, d_: ok_tied(p, N, e_, d_, protect)); res[f"tied|N{N}|{'protected' if protect else 'dense'}"] = {"fine_eps": e, "tube": d}
            print(f"TIED N={N:2d} {'zeros-protected' if protect else 'dense':15s}: fine certified eps* = {e:.4f} (tube {d:.3f})  ({time.time()-t0:.0f}s)", flush=True); json.dump(res, open("cert_bilinear2.json", "w"))
    for fam, mk, N in [("majority", majority_ab, 8), ("majority", majority_ab, 16), ("dyck", dyck_ab, 16)]:
        spec = mk(N, 0, 1); p = build_bil(spec)
        for protect in [False, True]:
            key = f"bilinear|{fam}|{N}|{'protected' if protect else 'dense'}"; e = res[key]["fine_eps"]; row = {}
            for m in [1, 2, 3, 5, 8]:
                row[m] = func_attack(spec, p, m * e, protect)
                if row[m] <= 0: break
            res[f"funcattack|{fam}|{N}|{'protected' if protect else 'dense'}"] = {"certified_eps": e, "min_margin_found": {str(k): v for k, v in row.items()}, "first_break_multiple": next((m for m, v in row.items() if v <= 0), None)}
            print(f"FUNCTION ATTACK {fam} N={N} {'zeros-protected' if protect else 'dense':15s} (certified eps*={e:.4f}): min output margin found at multiples " + "  ".join(f"{m}x:{v:+.3f}" for m, v in row.items()) + ("  -> real wrong output found" if min(row.values()) <= 0 else "  -> no wrong output found") + f"  ({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open("cert_bilinear2.json", "w"))
    print("DONE", flush=True)
