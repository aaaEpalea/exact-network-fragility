"""Trained GRU baselines on prefix-hist and majority-so-far; same noise protocols as the other designs."""
import json, sys, time, warnings
import numpy as np, jax, jax.numpy as jnp, optax
warnings.filterwarnings("ignore")
from scaling import s50
V = 6; HID = int(sys.argv[1]) if len(sys.argv) > 1 else 128
PROBS = [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]
ABS = [float(x) for x in np.logspace(-4, 0, 13)]; REL = [float(x) for x in np.logspace(-3, 0.3, 14)]; NSEEDS = 4

def init(key, C):
    ks = iter(jax.random.split(key, 20)); n = lambda s, sc: sc * jax.random.normal(next(ks), s)
    p = {}
    for g in "zrh": p["W" + g] = n((V, HID), V ** -0.5); p["U" + g] = n((HID, HID), HID ** -0.5); p["b" + g] = jnp.zeros(HID)
    p["Wo"] = n((HID, C), HID ** -0.5); p["bo"] = jnp.zeros(C); return p

def forward(p, X):
    xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); h0 = jnp.zeros((X.shape[0], HID))
    def step(h, x):
        z = jax.nn.sigmoid(x @ p["Wz"] + h @ p["Uz"] + p["bz"]); r = jax.nn.sigmoid(x @ p["Wr"] + h @ p["Ur"] + p["br"])
        hh = jnp.tanh(x @ p["Wh"] + (r * h) @ p["Uh"] + p["bh"]); h2 = (1 - z) * h + z * hh
        return h2, h2 @ p["Wo"] + p["bo"]
    _, ys = jax.lax.scan(step, h0, xs); return jnp.swapaxes(ys, 0, 1)

def labels(task, X):  # X in 0..V-1
    if task == "prefix_hist":
        return np.array([[int((row[:i + 1] == row[i]).sum()) for i in range(len(row))] for row in X])
    o = np.cumsum(X == 0, 1); t = np.cumsum(X == 1, 1); return (o > t).astype(int)

def sample(task, rng, B, L):
    return rng.integers(0, V, size=(B, L)) if task == "prefix_hist" else rng.choice(V, size=(B, L), p=PROBS)

def noisy(P, s, key, mode):
    lv, tr = jax.tree_util.tree_flatten(P); ks = jax.random.split(key, len(lv))
    f = (lambda a, k: a + s * jax.random.normal(k, a.shape)) if mode == "abs" else (lambda a, k: a + s * jnp.sqrt(jnp.mean(a ** 2)) * jax.random.normal(k, a.shape))
    return jax.tree_util.tree_unflatten(tr, [f(a, k) for a, k in zip(lv, ks)])

res = {}; t0 = time.time()
for task in ["prefix_hist", "majority"]:
    for N in [8, 16, 32]:
        C = N + 1 if task == "prefix_hist" else 2
        for seed in range(2):
            p = init(jax.random.PRNGKey(seed), C); steps = 3000
            opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.cosine_decay_schedule(3e-3, steps, 0.02))); st = opt.init(p)
            @jax.jit
            def step(p, st, X, y):
                def loss(p): return optax.softmax_cross_entropy_with_integer_labels(forward(p, X), y).mean()
                l, g = jax.value_and_grad(loss)(p); u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, l
            rng = np.random.default_rng(seed); lens = sorted({max(2, N // 4), max(2, N // 2), max(2, 3 * N // 4), N - 1})
            for i in range(steps):
                L = lens[i % len(lens)]; X = sample(task, rng, 128, L)
                p, st, l = step(p, st, jnp.array(X), jnp.array(labels(task, X)))
            Xe = sample(task, np.random.default_rng(99), 200, N - 1); ye = labels(task, Xe); Xj = jnp.array(Xe)
            f = jax.jit(lambda P: jnp.argmax(forward(P, Xj), -1))
            acc = lambda P: float((np.asarray(f(P)) == ye).all(1).mean())
            clean = acc(p); npar = int(sum(np.prod(a.shape) for a in p.values()))
            out = {"clean": clean, "params": npar}
            for mode, grid in [("abs", ABS), ("rel", REL)]:
                out[mode] = s50({str(s): [acc(noisy(p, s, jax.random.PRNGKey(k), mode)) for k in range(NSEEDS)] for s in grid})
            res[f"{task}|{N}|{seed}"] = out
            print(f"GRU{HID} {task:11s} N={N:2d} seed{seed}: clean={clean:.3f} params={npar} sigma50_abs={out['abs']:.2e} alpha50_rel={out['rel']:.2e} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open(f"trained_gru_{HID}.json", "w"))
print("DONE", flush=True)
