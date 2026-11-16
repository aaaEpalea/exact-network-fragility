"""Function-level adversarial attack: (a) bilinear N=32, (b) weight-tied design with k = 1, 4, 15 instances (ONE shared perturbation of the shared matrices)."""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
import digital_compiler as dc
from zeros_empirical import majority_ab, dyck_ab
from bilinear_lib import build_bil, build_tied
from cert_bilinear2 import func_attack
import os
V = 6; C2 = json.load(open("cert_bilinear2.json")); res = json.load(open("func_attack_round2.json")) if os.path.exists("func_attack_round2.json") else {}; t0 = time.time()

def run_tied_logits(p, X, k):
    xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); s = jnp.broadcast_to(p["s0"], (X.shape[0], k, p["s0"].shape[0]))
    def f(s, x):
        g = jnp.einsum("krv,bv->bkr", p["G"], x); z = jnp.einsum("bkr,rij,bkj->bki", g, p["P"], s); s2 = jnp.clip(2.0 * z - 0.5, 0.0, 1.0); return s2, s2 @ p["T"].T
    _, ys = jax.lax.scan(f, s, xs); return jnp.swapaxes(ys, 0, 1)                      # (B, L, k, 2)

def func_attack_tied(N, k, eps, protect, steps=120, restarts=3, seed=0):
    allpairs = [(a, b) for a in range(V) for b in range(V) if a != b]; pairs = allpairs[:k]; p = build_tied(N, pairs); specs = [majority_ab(N, a, b) for a, b in pairs]
    rng = np.random.default_rng(seed); L = 4 * N; X = rng.choice(V, size=(64, L))
    for j, (a, b) in enumerate(pairs[:min(k, 8)]): X[2 * j % 64] = a; X[(2 * j + 1) % 64] = b          # constant 'up' / 'down' sequences for several instances
    ref = jnp.array(np.stack([np.array([dc.reference(sp, list(r)) for r in X]) for sp in specs], -1)); Xj = jnp.array(X)
    masks = {kk: ((p[kk] != 0).astype(jnp.float32) if protect else jnp.ones_like(p[kk])) for kk in p}
    def margins(d):
        q = {kk: p[kk] + d[kk] for kk in p}; lg = run_tied_logits(q, Xj, k); corr = jnp.take_along_axis(lg, ref[..., None], -1)[..., 0]
        oth = jnp.where(jax.nn.one_hot(ref, 2) > 0, -1e9, lg).max(-1); return corr - oth
    g = jax.jit(jax.value_and_grad(lambda d: -0.05 * jax.nn.logsumexp(-margins(d).reshape(-1) / 0.05))); mj = jax.jit(margins); best = 1e9
    for r in range(restarts):
        d = {kk: eps * masks[kk] * jnp.asarray(rng.choice([-1., 1.], size=p[kk].shape), jnp.float32) for kk in p}
        for t in range(steps):
            _, gr = g(d); d = {kk: jnp.clip(d[kk] - eps / 6 * jnp.sign(gr[kk]), -eps, eps) * masks[kk] for kk in p}
        best = min(best, float(mj(d).min()))
    return best

for fam, mk in [("majority", majority_ab), ("dyck", dyck_ab)]:
    N = 32; spec = mk(N, 0, 1); p = build_bil(spec)
    for protect in [False, True]:
        if f"bilinear|{fam}|{N}|{'protected' if protect else 'dense'}" in res: continue
        e = C2[f"bilinear|{fam}|{N}|{'protected' if protect else 'dense'}"]["fine_eps"]; row = {}
        for m in [1, 2, 3, 5]:
            row[m] = func_attack(spec, p, m * e, protect)
            if row[m] <= 0: break
        res[f"bilinear|{fam}|{N}|{'protected' if protect else 'dense'}"] = {"certified_eps": e, "min_margin_found": {str(a): b for a, b in row.items()}, "first_break_multiple": next((m for m, v in row.items() if v <= 0), None)}
        print(f"FUNCTION ATTACK bilinear {fam} N={N} {'zeros-protected' if protect else 'dense':15s} (certified {e:.4f}): " + "  ".join(f"{m}x:{v:+.3f}" for m, v in row.items()) + f"  ({time.time()-t0:.0f}s)", flush=True); json.dump(res, open("func_attack_round2.json", "w"))
for k in [1, 4, 15]:
    for protect in [False, True]:
        if f"tied|N16|k{k}|{'protected' if protect else 'dense'}" in res: continue
        e = C2[f"tied|N16|{'protected' if protect else 'dense'}"]["fine_eps"]; row = {}
        for m in [1, 2, 3, 5]:
            row[m] = func_attack_tied(16, k, m * e, protect)
            if row[m] <= 0: break
        res[f"tied|N16|k{k}|{'protected' if protect else 'dense'}"] = {"certified_eps": e, "min_margin_found": {str(a): b for a, b in row.items()}, "first_break_multiple": next((m for m, v in row.items() if v <= 0), None)}
        print(f"FUNCTION ATTACK tied N=16 k={k:2d} {'zeros-protected' if protect else 'dense':15s} (certified {e:.4f}): " + "  ".join(f"{m}x:{v:+.3f}" for m, v in row.items()) + f"  ({time.time()-t0:.0f}s)", flush=True); json.dump(res, open("func_attack_round2.json", "w"))
print("DONE", flush=True)
