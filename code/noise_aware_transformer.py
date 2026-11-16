"""Can ordinary noise-aware training flatten the length-decay of counting robustness in a TRAINED transformer? (hist, bidirectional)"""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp, optax
warnings.filterwarnings("ignore")
import trained_baseline as tb
V = 6; ALPH = [float(x) for x in np.logspace(-3, 0, 13)]
def noisy(P, a, key):
    lv, tr = jax.tree_util.tree_flatten(P); ks = jax.random.split(key, len(lv))
    return jax.tree_util.tree_unflatten(tr, [x + a * jnp.sqrt(jnp.mean(x ** 2)) * jax.random.normal(k, x.shape) for x, k in zip(lv, ks)])
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
    p = tb.init(jax.random.PRNGKey(0), 1, 64, 4, 128, True)
    def make_step(opt, noise):
        @jax.jit
        def step(p, st, toks, y, key):
            k1, k2 = jax.random.split(key); alpha = jnp.exp(jax.random.uniform(k1, (), minval=np.log(0.003), maxval=np.log(0.12)))
            def loss(p):
                ce = lambda q: optax.softmax_cross_entropy_with_integer_labels(tb.forward(q, toks, 4, True)[:, 1:], y).mean()
                return ce(p) + (ce(noisy(p, alpha, k2)) if noise else 0.0)
            l, g = jax.value_and_grad(loss)(p); u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, l
        return step
    def batch(rng):
        L = int(rng.integers(2, N)); x = rng.integers(0, V, size=(256, L))
        return jnp.array(np.concatenate([np.full((256, 1), tb.BOS_ID), x], 1)), jnp.array((x[:, :, None] == x[:, None, :]).sum(-1))
    rng = np.random.default_rng(0); key = jax.random.PRNGKey(3)
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, 2e-3, 300, 3500, 1e-5), weight_decay=0.01); st = opt.init(p); step = make_step(opt, False)
    for i in range(3500):
        toks, y = batch(rng); key, sk = jax.random.split(key); p, st, l = step(p, st, toks, y, sk)
    c0 = acc(p); a0 = a50({str(al): [acc(noisy(p, al, jax.random.PRNGKey(k))) for k in range(4)] for al in ALPH})
    print(f"  stage 1 (clean-trained) hist N={N:2d}: clean={c0:.3f} alpha50={a0:.2e} ({time.time()-t0:.0f}s)", flush=True)
    opt = optax.adamw(optax.cosine_decay_schedule(5e-4, 1800, 0.05), weight_decay=0.0); st = opt.init(p); step = make_step(opt, True)
    for i in range(1800):
        toks, y = batch(rng); key, sk = jax.random.split(key); p, st, l = step(p, st, toks, y, sk)
    clean = acc(p); a = a50({str(al): [acc(noisy(p, al, jax.random.PRNGKey(k))) for k in range(4)] for al in ALPH})
    res[N] = {"clean_before": c0, "alpha50_before": a0, "clean": clean, "alpha50": a}
    print(f"noise-aware fine-tuned  hist N={N:2d}: clean={clean:.3f} alpha50={a:.2e}  (before: {a0:.2e}; gain {a/a0:.1f}x) ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open("noise_aware_transformer.json", "w"))
print("DONE", flush=True)
