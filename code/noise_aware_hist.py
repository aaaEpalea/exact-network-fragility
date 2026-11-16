"""Trained transformer fix attempt #1: noise-aware training (weight-noise injection) on hist; does the N-scaling of robustness change?"""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp, optax
warnings.filterwarnings("ignore")
import trained_baseline as tb
V = 6; ALPH = [float(x) for x in np.logspace(-3, 0, 13)]; SEEDS = 4
def noisy(P, a, key):
    lv, tr = jax.tree_util.tree_flatten(P); ks = jax.random.split(key, len(lv))
    return jax.tree_util.tree_unflatten(tr, [x + a * jax.lax.stop_gradient(jnp.sqrt(jnp.mean(x ** 2))) * jax.random.normal(k, x.shape) for x, k in zip(lv, ks)])
def a50(curve):
    xs = np.array([float(k) for k in curve]); ys = np.array([np.mean(v) for v in curve.values()])
    if ys[0] < 0.5: return float("nan")
    for i in range(1, len(xs)):
        if ys[i] < 0.5:
            f = (ys[i-1]-0.5)/(ys[i-1]-ys[i]); return float(np.exp(np.log(xs[i-1])+f*(np.log(xs[i])-np.log(xs[i-1]))))
    return float("inf")
res = {}; t0 = time.time()
for N in [16, 32]:
    tb.MAX_POS, tb.N_OUT = N + 1, N + 1
    rng_e = np.random.default_rng(5); ev = {}
    for L in range(2, N):
        x = rng_e.integers(0, V, size=(30, L)); ev[L] = (jnp.array(np.concatenate([np.full((30, 1), tb.BOS_ID), x], 1)), jnp.array((x[:, :, None] == x[:, None, :]).sum(-1)))
    fwd = jax.jit(lambda p, t: jnp.argmax(tb.forward(p, t, 4, True)[:, 1:], -1))
    def acc(p):
        ok = tot = 0
        for L, (t, y) in ev.items(): ok += int((np.asarray(fwd(p, t)) == np.asarray(y)).all(1).sum()); tot += len(y)
        return ok / tot
    p = tb.init(jax.random.PRNGKey(0), 1, 64, 4, 128, True); steps = 3500
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, 2e-3, 300, steps, 1e-5), weight_decay=0.01); st = opt.init(p)
    @jax.jit
    def step(p, st, toks, y, key):
        k1, k2 = jax.random.split(key); a = jnp.exp(jax.random.uniform(k1, (), minval=np.log(0.01), maxval=np.log(0.3)))
        def loss(p): return optax.softmax_cross_entropy_with_integer_labels(tb.forward(noisy(p, a, k2), toks, 4, True)[:, 1:], y).mean()
        l, g = jax.value_and_grad(loss)(p); u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, l
    rng = np.random.default_rng(0); key = jax.random.PRNGKey(1)
    for i in range(steps):
        L = int(rng.integers(2, N)); x = rng.integers(0, V, size=(256, L))
        toks = jnp.array(np.concatenate([np.full((256, 1), tb.BOS_ID), x], 1)); y = jnp.array((x[:, :, None] == x[:, None, :]).sum(-1))
        key, sk = jax.random.split(key); p, st, l = step(p, st, toks, y, sk)
    clean = acc(p)
    def noisy_eval(p_, a, key): 
        lv, tr = jax.tree_util.tree_flatten(p_); ks = jax.random.split(key, len(lv))
        return jax.tree_util.tree_unflatten(tr, [x + a * jnp.sqrt(jnp.mean(x ** 2)) * jax.random.normal(k, x.shape) for x, k in zip(lv, ks)])
    curve = {str(a): [acc(noisy_eval(p, a, jax.random.PRNGKey(k))) for k in range(SEEDS)] for a in ALPH}
    res[N] = {"clean": clean, "alpha50": a50(curve)}
    print(f"noise-aware trained hist N={N:2d}: clean={clean:.3f} alpha50={a50(curve):.2e} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open("noise_aware_hist.json", "w"))
print("DONE", flush=True)
