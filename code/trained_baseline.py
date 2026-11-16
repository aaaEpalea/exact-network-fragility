"""Trained-transformer baselines for the same tasks as the Tracr-compiled models.

Tasks: reverse, sort_unique, hist. Non-causal encoder transformer with a BOS token,
matching the compiled models' setup. Reports exact-sequence accuracy on the SAME
eval sequences used for the compiled models.
"""
import pickle, sys, time, warnings
import numpy as np
import jax, jax.numpy as jnp, optax
warnings.filterwarnings("ignore")

V, N_OUT, BOS_ID, MAX_POS = 6, 9, 6, 9   # tokens 1..6 -> ids 0..5, BOS -> 6; hist counts 0..8


def init(key, n_layers, d, heads, mlp, use_ln=True):
    ks = iter(jax.random.split(key, 100))
    nrm = lambda shape, s: s * jax.random.normal(next(ks), shape)
    p = {"tok": nrm((V + 1, d), 0.5), "pos": nrm((MAX_POS, d), 0.5), "out": nrm((d, N_OUT), d ** -0.5),
         "layers": []}
    for _ in range(n_layers):
        L = {"wq": nrm((d, d), d ** -0.5), "wk": nrm((d, d), d ** -0.5), "wv": nrm((d, d), d ** -0.5),
             "wo": nrm((d, d), d ** -0.5), "w1": nrm((d, mlp), d ** -0.5), "b1": jnp.zeros(mlp),
             "w2": nrm((mlp, d), mlp ** -0.5), "b2": jnp.zeros(d)}
        if use_ln:
            L.update({"ln1": jnp.ones(d), "ln1b": jnp.zeros(d), "ln2": jnp.ones(d), "ln2b": jnp.zeros(d)})
        p["layers"].append(L)
    if use_ln:
        p["lnf"], p["lnfb"] = jnp.ones(d), jnp.zeros(d)
    return p


def ln(x, g, b):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * g + b


def forward(p, toks, heads, use_ln=True):
    B, T = toks.shape
    x = p["tok"][toks] + p["pos"][:T][None]
    d = x.shape[-1]; hd = d // heads
    for L in p["layers"]:
        h = ln(x, L["ln1"], L["ln1b"]) if use_ln else x
        q = (h @ L["wq"]).reshape(B, T, heads, hd); k = (h @ L["wk"]).reshape(B, T, heads, hd)
        v = (h @ L["wv"]).reshape(B, T, heads, hd)
        a = jax.nn.softmax(jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(hd), -1)
        x = x + jnp.einsum("bhqk,bkhd->bqhd", a, v).reshape(B, T, d) @ L["wo"]
        h = ln(x, L["ln2"], L["ln2b"]) if use_ln else x
        x = x + jax.nn.gelu(h @ L["w1"] + L["b1"]) @ L["w2"] + L["b2"]
    if use_ln:
        x = ln(x, p["lnf"], p["lnfb"])
    return x @ p["out"]


def make_batch(task, rng, B, L):
    if task == "sort_unique":
        x = np.stack([rng.choice(V, size=L, replace=False) for _ in range(B)]) if L <= V else None
    else:
        x = rng.integers(0, V, size=(B, L))
    if task == "reverse":
        y = x[:, ::-1]
    elif task == "sort_unique":
        y = np.sort(x, 1)
    elif task == "hist":
        y = (x[:, :, None] == x[:, None, :]).sum(-1)
    toks = np.concatenate([np.full((B, 1), BOS_ID), x], 1)
    return jnp.array(toks), jnp.array(y)


def eval_sets(task):
    """Same eval sequences + expected outputs as the compiled-model harness (class-index form)."""
    from fragility import sample_inputs, build_programs
    prog, kind, atol, samp = build_programs()[task]
    sets = {}
    for L, xs in sample_inputs(samp).items():
        toks = jnp.array([[BOS_ID] + [x - 1 for x in xi] for xi in xs])
        exp = np.array([[(e - 1 if task != "hist" else int(round(e))) for e in prog(xi)] for xi in xs])
        sets[L] = (toks, jnp.array(exp))
    return sets


def exact_acc(p, sets, heads, use_ln=True):
    ok = tot = 0
    for L, (toks, exp) in sets.items():
        pred = jnp.argmax(forward(p, toks, heads, use_ln)[:, 1:], -1)
        ok += int((pred == exp).all(-1).sum()); tot += toks.shape[0]
    return ok / tot


def train(task, n_layers, d=64, heads=4, mlp=128, steps=6000, seed=0, use_ln=True, wd=0.01, log=True):
    rng = np.random.default_rng(seed)
    p = init(jax.random.PRNGKey(seed), n_layers, d, heads, mlp, use_ln)
    sched = optax.warmup_cosine_decay_schedule(0.0, 2e-3, 300, steps, 1e-5)
    opt = optax.adamw(sched, weight_decay=wd); st = opt.init(p)
    lens = [3, 4, 5, 6, 7] if task != "sort_unique" else [3, 4, 5, 6]

    @jax.jit
    def step(p, st, toks, y):
        def loss(p):
            lg = forward(p, toks, heads, use_ln)[:, 1:]
            return optax.softmax_cross_entropy_with_integer_labels(lg, y).mean()
        l, g = jax.value_and_grad(loss)(p)
        u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, l

    sets = eval_sets(task); t0 = time.time()
    for i in range(steps):
        toks, y = make_batch(task, rng, 256, lens[i % len(lens)])
        p, st, l = step(p, st, toks, y)
        if log and (i % 1500 == 0 or i == steps - 1):
            print(f"  {task} seed{seed} step {i} loss {float(l):.4f} acc {exact_acc(p, sets, heads, use_ln):.3f} ({time.time()-t0:.0f}s)", flush=True)
    return p, exact_acc(p, sets, heads, use_ln)


if __name__ == "__main__":
    cfg = {"hist": (1, 1500), "sort_unique": (2, 3000), "reverse": (4, 2500)}   # (layers matching compiled, steps)
    out = {}
    for task, (nl, nsteps) in cfg.items():
        for seed in range(3):
            p, acc = train(task, nl, seed=seed, steps=nsteps)
            out[(task, seed)] = (jax.tree_util.tree_map(np.asarray, p), acc)
            print(f"== {task} seed {seed}: exact acc {acc:.3f}", flush=True)
            pickle.dump(out, open("trained.pkl", "wb"))
