"""Compiled (Tracr) vs trained transformers under scale-free weight noise.

Protocol A ("rms-additive"): every tensor W gets W + alpha * rms(W) * N(0,1)
Protocol B ("multiplicative"): every tensor W gets W * (1 + alpha * N(0,1))
alpha is dimensionless, so models with different weight scales are comparable.
"""
import json, pickle, warnings
import numpy as np
import jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
from fragility import Task, build_programs, flat_items
import trained_baseline as tb

ALPHAS = [float(x) for x in np.logspace(-4, 0, 13)]
SEEDS = 6
TASKS = ["hist", "sort_unique", "reverse"]


def noisy(params, alpha, key, mode):
    leaves, tree = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    out = []
    for a, k in zip(leaves, keys):
        a = jnp.asarray(a)
        n = jax.random.normal(k, a.shape)
        if mode == "rms":
            out.append(a + alpha * jnp.sqrt(jnp.mean(a ** 2)) * n)
        else:
            out.append(a * (1 + alpha * n))
    return jax.tree_util.tree_unflatten(tree, out)


def alpha50(curve):
    xs = np.array(list(curve.keys()), dtype=float); ys = np.array([np.mean(v) for v in curve.values()])
    if ys[0] < 0.5: return float("nan")
    for i in range(1, len(xs)):
        if ys[i] < 0.5:
            f = (ys[i - 1] - 0.5) / (ys[i - 1] - ys[i])
            return float(np.exp(np.log(xs[i - 1]) + f * (np.log(xs[i]) - np.log(xs[i - 1]))))
    return float("inf")


if __name__ == "__main__":
    trained = pickle.load(open("trained.pkl", "rb"))
    res = {"compiled": {}, "trained": {}}
    for task in TASKS:
        prog, kind, atol, samp = build_programs()[task]
        T = Task(task, prog, kind, atol, samp)
        res["compiled"][task] = {m: {str(a): [T.accuracy(noisy(T.params, a, jax.random.PRNGKey(k), m)) for k in range(SEEDS)]
                                     for a in ALPHAS} for m in ["rms", "mult"]}
        sets = tb.eval_sets(task)
        fwd = jax.jit(lambda p, t: jnp.argmax(tb.forward(p, t, 4, True)[:, 1:], -1))
        def acc(p):
            ok = tot = 0
            for L, (toks, exp) in sets.items():
                ok += int((fwd(p, toks) == exp).all(-1).sum()); tot += toks.shape[0]
            return ok / tot
        res["trained"][task] = {}
        for (t, seed), (p, base) in trained.items():
            if t != task: continue
            p = jax.tree_util.tree_map(jnp.asarray, p)
            res["trained"][task][seed] = {"baseline": base, **{m: {str(a): [acc(noisy(p, a, jax.random.PRNGKey(100 + k), m)) for k in range(SEEDS)]
                                                                for a in ALPHAS} for m in ["rms", "mult"]}}
        print(task, "done", flush=True)
        json.dump(res, open("compare.json", "w"))
    # summary
    print(f"\n{'task':12s} {'mode':5s} {'compiled a50':>13s} {'trained a50 (per seed)':>40s}")
    for task in TASKS:
        for m in ["rms", "mult"]:
            c = alpha50(res["compiled"][task][m])
            tr = [alpha50(v[m]) for v in res["trained"][task].values()]
            print(f"{task:12s} {m:5s} {c:13.2e}   " + "  ".join(f"{x:9.2e}" for x in tr))
