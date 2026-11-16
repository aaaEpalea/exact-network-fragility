import json, time
from fragility import *
SEEDS = 5
SIG = [float(x) for x in np.logspace(-4, 0, 13)]
BITS = [16, 12, 10, 8, 6, 5, 4, 3, 2]
res = {}
t0 = time.time()
for name, (prog, kind, atol, samp) in build_programs().items():
    T = Task(name, prog, kind, atol, samp)
    r = {"baseline": T.accuracy(T.params)}
    r["gauss_abs"] = {str(s): [T.accuracy(add_gaussian(T.params, s, jax.random.PRNGKey(k))) for k in range(SEEDS)] for s in SIG}
    r["gauss_rel"] = {str(s): [T.accuracy(add_relative(T.params, s, jax.random.PRNGKey(k))) for k in range(SEEDS)] for s in SIG}
    r["quant"] = {str(b): T.accuracy(quantize(T.params, b)) for b in BITS}
    r["dtype"] = {d: T.accuracy(cast_dtype(T.params, getattr(jnp, d))) for d in ["float16", "bfloat16"]}
    r["group"] = {g: {str(s): [T.accuracy(add_gaussian_group(T.params, s, jax.random.PRNGKey(k), g)) for k in range(SEEDS)]
                      for s in [0.01, 0.03, 0.1, 0.3]} for g in GROUPS}
    res[name] = r
    print(name, "done", round(time.time() - t0), "s", flush=True)
    json.dump(res, open("results.json", "w"))
print("ALL DONE", flush=True)
