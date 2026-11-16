"""Margin tuning of the digital designs: snap thresholds (lo,hi) and global state scale k (exact-preserving)."""
import json, time, itertools, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
from scaling import s50
import digital_prefix_hist as dph, second_program as sp
V = 6; SIG = [float(x) for x in np.logspace(-3, -0.3, 13)]; ALPH = [float(x) for x in np.logspace(-2, 0.5, 13)]; NS = 4

def tune_hist(p, N, lo, hi, k):
    n = N + 1; D = V * n; g = 1.0 / (hi - lo); q = dict(p)
    G = np.kron(np.eye(V), np.ones((n, 1)))
    q["G1"], q["b1"], q["G2"], q["G3"], q["b3"] = k * G, -k * np.ones(D), -k * G, k * G, -k * np.ones(D)
    q["S1"], q["S2"] = g * np.eye(D), g * np.eye(D); q["bs1"], q["bs2"] = -g * lo * k * np.ones(D), -g * hi * k * np.ones(D); q["s0"] = k * np.asarray(p["s0"])
    return {a: jnp.asarray(b, jnp.float32) for a, b in q.items()}

def tune_maj(p, N, lo, hi, k):
    n = 2 * N + 1; g = 1.0 / (hi - lo); q = {a: np.asarray(b) for a, b in p.items()}
    q["G1a"], q["G2a"], q["G3"] = k * q["G1a"], k * q["G2a"], k * q["G3"]; q["b1"], q["b2"] = -k * np.ones(n), -k * np.ones(n)
    q["S1"], q["S2"] = g * np.eye(n), g * np.eye(n); q["bs1"], q["bs2"] = -g * lo * k * np.ones(n), -g * hi * k * np.ones(n)
    q["s0"] = k * q["s0"]; q["bA"] = k * q["bA"]
    return {a: jnp.asarray(b, jnp.float32) for a, b in q.items()}

def make_eval(task, N, rng):
    L = N - 1
    if task == "hist":
        X = np.stack([rng.integers(0, V, size=L) for _ in range(24)]); exp = np.array([dph.prefix_hist([int(t) + 1 for t in r]) for r in X])
        f = jax.jit(lambda P, X: dph.run_digital(P, X, True)); run = lambda P: np.asarray(f(P, jnp.array(X))); base = dph.build_digital(N); tune = tune_hist
    else:
        X = np.stack([rng.choice(V, size=L, p=sp.PROBS) for _ in range(24)]); exp = np.array([sp.maj_expected([int(t) + 1 for t in r]) for r in X])
        f = jax.jit(lambda P, X: sp.run_digital(P, X, True)); run = lambda P: np.asarray(f(P, jnp.array(X))); base = sp.build_digital(N); tune = tune_maj
    return base, tune, (lambda P: float((run(P) == exp).all(1).mean()))

def noise(P, s, key, rel):
    ks = dict(zip(P.keys(), jax.random.split(key, len(P))))
    return {k: v + s * (jnp.sqrt(jnp.mean(v ** 2)) if rel else 1.0) * jax.random.normal(ks[k], v.shape) for k, v in P.items()}

res = {}; t0 = time.time()
for task in ["hist", "majority"]:
    rng = np.random.default_rng(0); base, tune, acc = make_eval(task, 16, rng)
    best = (0, None)
    for lo, hi in [(0.25, 0.75), (0.1, 0.5), (0.5, 0.9), (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.45, 0.55)]:
        for k in [1.0, 2.0, 4.0]:
            P = tune(base, 16, lo, hi, k); clean = acc(P)
            if clean < 0.999: continue
            v = s50({str(s): [acc(noise(P, s, jax.random.PRNGKey(i), False)) for i in range(NS)] for s in SIG})
            res[f"{task}|lo{lo}|hi{hi}|k{k}"] = v
            if v > best[0]: best = (v, (lo, hi, k))
    default = res[f"{task}|lo0.25|hi0.75|k1.0"]
    print(f"[{task}] N=16 default sigma50={default:.2e}; best={best[0]:.2e} at (lo,hi,k)={best[1]}  gain {best[0]/default:.1f}x ({time.time()-t0:.0f}s)", flush=True)
    lo, hi, k = best[1]
    for N in [8, 32, 64]:
        rng = np.random.default_rng(0); b2, t2, acc2 = make_eval(task, N, rng)
        d = s50({str(s): [acc2(noise(b2, s, jax.random.PRNGKey(i), False)) for i in range(NS)] for s in SIG})
        Pt = t2(b2, N, lo, hi, k)
        t = s50({str(s): [acc2(noise(Pt, s, jax.random.PRNGKey(i), False)) for i in range(NS)] for s in SIG}) if acc2(Pt) > 0.999 else float("nan")
        ar = s50({str(a): [acc2(noise(Pt, a, jax.random.PRNGKey(i), True)) for i in range(NS)] for a in ALPH}) if acc2(Pt) > 0.999 else float("nan")
        res[f"{task}|N{N}|default"], res[f"{task}|N{N}|tuned"], res[f"{task}|N{N}|tuned_alpha"] = d, t, ar
        print(f"   [{task}] N={N:2d}: default sigma50={d:.2e}  tuned sigma50={t:.2e}  tuned alpha50={ar:.2e}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open("digital_tune.json", "w"))
print("DONE", flush=True)
