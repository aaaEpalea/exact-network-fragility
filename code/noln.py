import json, warnings, numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
import trained_baseline as tb
from compare_noise import noisy, alpha50, ALPHAS, SEEDS
res = {}
for task, nl, steps in [("hist", 1, 1500), ("sort_unique", 2, 3000)]:
    sets = tb.eval_sets(task)
    fwd = jax.jit(lambda p, t: jnp.argmax(tb.forward(p, t, 4, False)[:, 1:], -1))
    def acc(p):
        ok = tot = 0
        for L, (toks, exp) in sets.items():
            ok += int((fwd(p, toks) == exp).all(-1).sum()); tot += toks.shape[0]
        return ok / tot
    res[task] = {}
    for seed in range(2):
        p, base = tb.train(task, nl, steps=steps, seed=seed, use_ln=False, log=False)
        r = {"baseline": base}
        for m in ["rms", "mult"]:
            r[m] = {str(a): [acc(noisy(p, a, jax.random.PRNGKey(100 + k), m)) for k in range(SEEDS)] for a in ALPHAS}
        res[task][seed] = r
        print(task, seed, "base", base, "a50 rms", f"{alpha50(r['rms']):.2e}", "mult", f"{alpha50(r['mult']):.2e}", flush=True)
        json.dump(res, open("noln.json", "w"))
print("DONE", flush=True)
