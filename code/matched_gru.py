"""Matched comparison: bilinear digital designs vs trained GRUs; SAME programs, SAME token distribution, SAME eval sequences, SAME noise protocols."""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp, optax
warnings.filterwarnings("ignore")
import digital_compiler as dc
from zeros_empirical import majority_ab, dyck_ab
from bilinear_lib import build_bil, run_bil, noisy
from scaling import s50
V = 6; SIG = [float(x) for x in np.logspace(-3, 0.7, 18)]; NS = 6
PROBS = {"majority": [0.3, 0.3, 0.1, 0.1, 0.1, 0.1], "dyck": [0.35, 0.2, 0.15, 0.15, 0.1, 0.05]}
def gru_init(key, H, C):
    ks = iter(jax.random.split(key, 20)); n = lambda s, sc: sc * jax.random.normal(next(ks), s); p = {}
    for g in "zrh": p["W" + g] = n((V, H), V ** -0.5); p["U" + g] = n((H, H), H ** -0.5); p["b" + g] = jnp.zeros(H)
    p["Wo"] = n((H, C), H ** -0.5); p["bo"] = jnp.zeros(C); return p
def gru_fwd(p, X):
    H = p["Uz"].shape[0]; xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); h0 = jnp.zeros((X.shape[0], H))
    def step(h, x):
        z = jax.nn.sigmoid(x @ p["Wz"] + h @ p["Uz"] + p["bz"]); r = jax.nn.sigmoid(x @ p["Wr"] + h @ p["Ur"] + p["br"]); hh = jnp.tanh(x @ p["Wh"] + (r * h) @ p["Uh"] + p["bh"])
        h2 = (1 - z) * h + z * hh; return h2, h2 @ p["Wo"] + p["bo"]
    _, ys = jax.lax.scan(step, h0, xs); return jnp.swapaxes(ys, 0, 1)
def train_gru(spec, task, N, H, seed, steps=2500):
    p = gru_init(jax.random.PRNGKey(seed), H, 2); opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.cosine_decay_schedule(3e-3, steps, 0.02))); st = opt.init(p)
    @jax.jit
    def step(p, st, X, y):
        l, g = jax.value_and_grad(lambda p: optax.softmax_cross_entropy_with_integer_labels(gru_fwd(p, X), y).mean())(p); u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, l
    rng = np.random.default_rng(seed)
    for i in range(steps):
        L = [N // 2, N - 1][i % 2]; X = rng.choice(V, size=(128, L), p=PROBS[task]); y = np.array([dc.reference(spec, list(r)) for r in X]); p, st, l = step(p, st, jnp.array(X), jnp.array(y))
    return p
import os
res = json.load(open('matched_gru.json')) if os.path.exists('matched_gru.json') else {}; t0 = time.time()
for task, mk in [("majority", majority_ab), ("dyck", dyck_ab)]:
    for N in [16, 32]:
        if all(f'{task}|N{N}|{nm}' in res for nm in ['bilinear (compiled)', 'GRU32 seed0', 'GRU32 seed1', 'GRU64 seed0', 'GRU64 seed1']): continue
        spec = mk(N, 0, 1); rng = np.random.default_rng(123); X = rng.choice(V, size=(200, N - 1), p=PROBS[task]); ref = np.array([dc.reference(spec, list(r)) for r in X]); Xj = jnp.array(X)
        systems = {"bilinear (compiled)": (build_bil(spec), jax.jit(lambda Q: run_bil(Q, Xj)))}
        for H in [32, 64]:
            for seed in range(2):
                p = train_gru(spec, task, N, H, seed); systems[f"GRU{H} seed{seed}"] = (p, jax.jit(lambda Q: jnp.argmax(gru_fwd(Q, Xj), -1)))
        for name, (p, f) in systems.items():
            acc = lambda Q: float((np.asarray(f(Q)) == ref).all(1).mean()); clean = acc(p)
            a = s50({str(s): [acc(noisy(p, s, jax.random.PRNGKey(k))) for k in range(NS)] for s in SIG}); r_ = s50({str(s): [acc(noisy(p, s, jax.random.PRNGKey(k), True)) for k in range(NS)] for s in SIG})
            res[f"{task}|N{N}|{name}"] = {"clean": clean, "sigma50": a, "alpha50": r_, "params": int(sum(np.prod(v.shape) for v in p.values()))}
            print(f"{task:8s} N={N:2d} {name:20s}: clean={clean:.3f} params={res[f'{task}|N{N}|{name}']['params']:6d} sigma50={a:.2e} alpha50={r_:.2e} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open("matched_gru.json", "w"))
print("DONE", flush=True)
