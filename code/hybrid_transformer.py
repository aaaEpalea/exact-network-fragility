"""Hybrid: trained transformer + frozen compiled digital counters (fed into the input embeddings). Task: hist (full-sequence)."""
import json, sys, time, warnings
import numpy as np, jax, jax.numpy as jnp, optax
warnings.filterwarnings("ignore")
import trained_baseline as tb
import digital_compiler as dc
V = 6; ALPH = [float(x) for x in np.logspace(-3, 0, 10)]

def a50(curve):
    xs = np.array([float(k) for k in curve]); ys = np.array([np.mean(v) for v in curve.values()])
    if ys[0] < 0.5: return float("nan")
    for i in range(1, len(xs)):
        if ys[i] < 0.5:
            f = (ys[i-1]-0.5)/(ys[i-1]-ys[i]); return float(np.exp(np.log(xs[i-1])+f*(np.log(xs[i])-np.log(xs[i-1]))))
    return float("inf")

def forward_h(p, toks, feat, heads=4):
    B, T = toks.shape; x = p["tok"][toks] + p["pos"][:T][None]
    if feat is not None: x = x + (feat @ p["Wf"])[:, None, :]
    d = x.shape[-1]; hd = d // heads
    for L in p["layers"]:
        h = tb.ln(x, L["ln1"], L["ln1b"])
        q = (h @ L["wq"]).reshape(B, T, heads, hd); k = (h @ L["wk"]).reshape(B, T, heads, hd); v = (h @ L["wv"]).reshape(B, T, heads, hd)
        a = jax.nn.softmax(jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(hd), -1)
        x = x + jnp.einsum("bhqk,bkhd->bqhd", a, v).reshape(B, T, d) @ L["wo"]
        h = tb.ln(x, L["ln2"], L["ln2b"]); x = x + jax.nn.gelu(h @ L["w1"] + L["b1"]) @ L["w2"] + L["b2"]
    return tb.ln(x, p["lnf"], p["lnfb"]) @ p["out"]

def analytic_feat(X, N):          # one-hot count per token value, flattened (V*(N+1)) -- equals the compiled module's clean final state
    B = X.shape[0]; cnt = np.stack([(X == v).sum(1) for v in range(V)], 1); f = np.zeros((B, V * (N + 1)), np.float32)
    for v in range(V): f[np.arange(B), v * (N + 1) + cnt[:, v]] = 1.0
    return f

def final_state(dp, meta, X):     # run the ACTUAL compiled digital network, return its final state
    xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); s = jnp.broadcast_to(dp["s0"], (X.shape[0], dp["s0"].shape[0]))
    def f(s, x): s2, _ = dc.step(dp, meta, s, x); return s2, None
    s, _ = jax.lax.scan(f, s, xs); return s

def rel_noise(P, a, key):
    lv, tr = jax.tree_util.tree_flatten(P); ks = jax.random.split(key, len(lv))
    return jax.tree_util.tree_unflatten(tr, [x + a * jnp.sqrt(jnp.mean(x ** 2)) * jax.random.normal(k, x.shape) for x, k in zip(lv, ks)])

def run(N, hybrid, steps=3000, seed=0):
    tb.MAX_POS, tb.N_OUT = N + 1, N + 1; t0 = time.time()
    lens_eval = list(range(2, N)) if N <= 16 else sorted({4, 8, N // 2, 3 * N // 4, N - 1}); nseq = 30 if N <= 32 else 16; nseeds = 4 if N <= 32 else 3
    rng_e = np.random.default_rng(5); ev = {}
    for L in lens_eval:
        x = rng_e.integers(0, V, size=(nseq, L)); ev[L] = (x, jnp.array(np.concatenate([np.full((nseq, 1), tb.BOS_ID), x], 1)), jnp.array((x[:, :, None] == x[:, None, :]).sum(-1)))
    p = tb.init(jax.random.PRNGKey(seed), 1, 64, 4, 128, True)
    if hybrid: p["Wf"] = 0.05 * jax.random.normal(jax.random.PRNGKey(seed + 1), (V * (N + 1), 64))
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, 2e-3, 300, steps, 1e-5), weight_decay=0.01); st = opt.init(p)
    @jax.jit
    def step(p, st, toks, feat, y):
        def loss(p): return optax.softmax_cross_entropy_with_integer_labels(forward_h(p, toks, feat if hybrid else None)[:, 1:], y).mean()
        l, g = jax.value_and_grad(loss)(p); u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, l
    rng = np.random.default_rng(seed)
    for i in range(steps):
        L = int(rng.integers(2, N)); x = rng.integers(0, V, size=(256, L))
        toks = jnp.array(np.concatenate([np.full((256, 1), tb.BOS_ID), x], 1)); y = jnp.array((x[:, :, None] == x[:, None, :]).sum(-1))
        p, st, l = step(p, st, toks, jnp.array(analytic_feat(x, N)) if hybrid else jnp.zeros((256, 1)), y)
    if hybrid:
        spec = dc.prefix_hist(N); dp, meta = dc.compile_spec(spec); fs = jax.jit(lambda d, X: final_state(d, meta, X))
    fwd = jax.jit(lambda p_, t, f: jnp.argmax(forward_h(p_, t, f if hybrid else None)[:, 1:], -1))
    def acc(p_, dp_=None):
        ok = tot = 0
        for L, (x, t, y) in ev.items():
            f = fs(dp_, jnp.array(x)) if hybrid else jnp.zeros((len(x), 1))
            ok += int((np.asarray(fwd(p_, t, f)) == np.asarray(y)).all(1).sum()); tot += len(y)
        return ok / tot
    clean = acc(p, dp if hybrid else None); curve = {}
    for a in ALPH:
        accs = []
        for k in range(nseeds):
            key = jax.random.PRNGKey(k); pk = rel_noise(p, a, key); dk = rel_noise(dp, a, jax.random.fold_in(key, 7)) if hybrid else None
            accs.append(acc(pk, dk))
        curve[str(a)] = accs
    out = {"clean": clean, "alpha50": a50(curve), "params_trainable": int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(p)))}
    print(f"N={N:2d} {'HYBRID  ' if hybrid else 'BASELINE'}: clean={clean:.3f} alpha50={out['alpha50']:.2e} ({time.time()-t0:.0f}s)", flush=True); return out

if __name__ == "__main__":
    res = {}
    for N in [int(a) for a in sys.argv[1:]] or [16, 32]:
        for hybrid in [False, True]:
            res[f"N{N}|{'hybrid' if hybrid else 'baseline'}"] = run(N, hybrid); json.dump(res, open("hybrid_transformer.json", "w"))
    print("DONE", flush=True)
