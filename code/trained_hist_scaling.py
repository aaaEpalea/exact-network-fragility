"""Do ordinary TRAINED transformers also lose noise tolerance on counting as N grows?  (hist, full-sequence)"""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp, optax
warnings.filterwarnings("ignore")
import trained_baseline as tb
from tracr.compiler import compiling, lib
V = 6; ALPH = [float(x) for x in np.logspace(-3, 0, 13)]; SEEDS = 4
def noisy(P, a, key, mode="rms"):
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
for N in [8, 16, 32]:
    tb.MAX_POS, tb.N_OUT = N + 1, N + 1
    rng_e = np.random.default_rng(5)
    ev = {}
    for L in range(2, N):
        x = rng_e.integers(0, V, size=(30, L)); ev[L] = (jnp.array(np.concatenate([np.full((30, 1), tb.BOS_ID), x], 1)), jnp.array((x[:, :, None] == x[:, None, :]).sum(-1)))
    fwd = jax.jit(lambda p, t: jnp.argmax(tb.forward(p, t, 4, True)[:, 1:], -1))
    def acc(p):
        ok = tot = 0
        for L, (t, y) in ev.items(): ok += int((np.asarray(fwd(p, t)) == np.asarray(y)).all(1).sum()); tot += len(y)
        return ok / tot
    res[N] = {"trained": [], "compiled": None}
    for seed in range(2):
        p = tb.init(jax.random.PRNGKey(seed), 1, 64, 4, 128, True)
        opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, 2e-3, 300, 3500, 1e-5), weight_decay=0.01); st = opt.init(p)
        @jax.jit
        def step(p, st, toks, y):
            def loss(p): return optax.softmax_cross_entropy_with_integer_labels(tb.forward(p, toks, 4, True)[:, 1:], y).mean()
            l, g = jax.value_and_grad(loss)(p); u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, l
        rng = np.random.default_rng(seed)
        for i in range(3500):
            L = int(rng.integers(2, N)); x = rng.integers(0, V, size=(256, L))
            toks = jnp.array(np.concatenate([np.full((256, 1), tb.BOS_ID), x], 1)); y = jnp.array((x[:, :, None] == x[:, None, :]).sum(-1))
            p, st, l = step(p, st, toks, y)
        clean = acc(p)
        curve = {str(a): [acc(noisy(p, a, jax.random.PRNGKey(k))) for k in range(SEEDS)] for a in ALPH}
        res[N]["trained"].append({"clean": clean, "alpha50": a50(curve)})
        print(f"trained hist N={N:2d} seed{seed}: clean={clean:.3f} alpha50={a50(curve):.2e} ({time.time()-t0:.0f}s)", flush=True)
    # compiled comparison, same protocol, same eval distribution
    S = max(100, int(1.5 * N * N)); m = compiling.compile_rasp_to_model(lib.make_hist(), vocab=set(range(1, V + 1)), max_seq_len=N, compiler_bos="BOS", mlp_exactness=S)
    rng = np.random.default_rng(5); B = {}
    for L in range(2, N):
        xs = [list(map(int, rng.integers(1, V + 1, size=L))) for _ in range(30)]; B[L] = (jnp.array([m.input_encoder.encode(["BOS"] + x) for x in xs]), [[x.count(v) for v in x] for x in xs])
    cf = jax.jit(lambda p, t: m.forward(p, t).unembedded_output)
    def cacc(p):
        ok = tot = 0
        for L, (t, e) in B.items():
            out = np.asarray(cf(p, t))[:, 1:]
            for row, ee in zip(out, e): ok += bool(np.allclose(m.output_encoder.decode(row.tolist()), ee, atol=0.5)); tot += 1
        return ok / tot
    P0 = jax.tree_util.tree_map(jnp.asarray, m.params)
    curve = {str(a): [cacc(noisy(P0, a, jax.random.PRNGKey(k))) for k in range(SEEDS)] for a in ALPH}
    res[N]["compiled"] = {"clean": cacc(P0), "alpha50": a50(curve)}
    print(f"compiled hist N={N:2d}: clean={res[N]['compiled']['clean']:.3f} alpha50={a50(curve):.2e} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open("trained_hist_scaling.json", "w"))
print("DONE", flush=True)
