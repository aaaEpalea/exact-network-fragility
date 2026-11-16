import json, pickle, sys, time, warnings
warnings.filterwarnings("ignore")
import optax
from fragility import *
from hardening import *
from exhaustive import verify

SEEDS = 5
ABS = [float(x) for x in np.logspace(-5, -0.5, 14)]
ALPHAS = [float(x) for x in np.logspace(-4, 0, 13)]
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
BETA = 8.0

def s50(curve):
    xs = np.array([float(k) for k in curve]); ys = np.array([np.mean(v) for v in curve.values()])
    if ys[0] < 0.5: return float("nan")
    for i in range(1, len(xs)):
        if ys[i] < 0.5:
            f = (ys[i-1]-0.5)/(ys[i-1]-ys[i]); return float(np.exp(np.log(xs[i-1])+f*(np.log(xs[i])-np.log(xs[i-1]))))
    return float("inf")

def sampler(name, rng, B, L):
    if name == "sort_unique":
        return np.stack([rng.choice(np.arange(1, 7), size=L, replace=False) for _ in range(B)])
    return rng.integers(1, 7, size=(B, L))

res = {}; t0 = time.time()
for name in ["hist", "sort_unique", "reverse"]:
    prog, kind, atol, samp = build_programs()[name]
    T = Task(name, prog, kind, atol, samp); net = CompiledNet(T)
    enc = T.model.input_encoder
    e = enc.encode(["BOS"] + list(range(1, 7))); bos = e[0]; lut = np.zeros(7, np.int32)
    for v in range(1, 7): lut[v] = e[v]
    fwd = jax.jit(net.predict)
    res[name] = {}
    for vname, kw in {"from_base": dict(), "from_balance+c=0.25": dict(balance=True, c=0.25)}.items():
        hp0 = net.harden(T.params, **kw)
        params = jax.tree_util.tree_map(jnp.asarray, hp0["base"])
        labels_fn = jax.jit(lambda toks: net.predict({"base": jax.tree_util.tree_map(jnp.asarray, net.harden(T.params)["base"]), "snaps": []}, toks))
        sched = optax.cosine_decay_schedule(5e-4, STEPS, 0.1)
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched)); st = opt.init(params)
        lens = list(range(1, 8)) if name != "sort_unique" else list(range(1, 7))

        @jax.jit
        def step(params, st, toks, labels, key):
            k1, k2 = jax.random.split(key)
            sigma = jnp.exp(jax.random.uniform(k1, (), minval=np.log(1e-4), maxval=np.log(3e-2)))
            def loss(p):
                lg_n = net.forward(perturb({"base": p, "snaps": []}, sigma, k2), toks)[:, 1:] * BETA
                lg_c = net.forward({"base": p, "snaps": []}, toks)[:, 1:] * BETA
                ce = lambda lg: optax.softmax_cross_entropy_with_integer_labels(lg, labels).mean()
                return ce(lg_n) + ce(lg_c)
            l, g = jax.value_and_grad(loss)(params)
            u, st = opt.update(g, st, params); return optax.apply_updates(params, u), st, l

        rng = np.random.default_rng(123); key = jax.random.PRNGKey(7)
        for i in range(STEPS):
            L = lens[i % len(lens)]
            x = sampler(name, rng, 256, L)
            toks = jnp.array(np.concatenate([np.full((256, 1), bos, np.int32), lut[x]], 1))
            labels = labels_fn(toks)[:, 1:]
            key, sk = jax.random.split(key)
            params, st, l = step(params, st, toks, labels, sk)
            if i % 1000 == 0: print(f"  {name} {vname} step {i} loss {float(l):.4f} ({time.time()-t0:.0f}s)", flush=True)
        hp = {"base": params, "snaps": []}
        ok, tot = verify(net, hp, name)
        r = {"exhaustive_ok": ok, "exhaustive_total": tot,
             "nonzero_frac": float(np.mean(np.concatenate([(np.asarray(a) != 0).ravel() for a in jax.tree_util.tree_leaves(params)])))}
        r["abs"] = {str(s): [accuracy_hp(net, perturb(hp, s, jax.random.PRNGKey(k), "abs"), fwd) for k in range(SEEDS)] for s in ABS}
        r["rms"] = {str(a): [accuracy_hp(net, perturb(hp, a, jax.random.PRNGKey(k), "rms"), fwd) for k in range(SEEDS)] for a in ALPHAS}
        res[name][vname] = r
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), open(f"hardened_{name}_{vname}.pkl", "wb"))
        print(f"== {name} {vname}: exhaustive {ok}/{tot}; sigma50={s50(r['abs']):.2e}; alpha50={s50(r['rms']):.2e}; nz={r['nonzero_frac']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open("noise_train.json", "w"))
print("ALL DONE", flush=True)
