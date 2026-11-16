import json, time
from fragility import *
SEEDS = 6
ABS = [float(x) for x in np.logspace(-5, -1, 13)]
REL = [float(x) for x in np.logspace(-4, 0, 13)]
GSIG = [float(x) for x in np.logspace(-4, -1, 7)]
res = {}; t0 = time.time()
for name, (prog, kind, atol, samp) in build_programs().items():
    T = Task(name, prog, kind, atol, samp)
    ev = lambda p: T.accuracy(p)
    r = {"baseline": ev(T.params),
         "layers": T.model.model_config.num_layers,
         "d_model": int(T.params["pos_embed"]["embeddings"].shape[1]),
         "params": int(sum(np.asarray(a).size for _, _, a in flat_items(T.params))),
         "nonzero_frac": float(np.mean(np.concatenate([(np.asarray(a) != 0).ravel() for _, _, a in flat_items(T.params)])))}
    r["abs"] = {str(s): [ev(add_gaussian(T.params, s, jax.random.PRNGKey(k))) for k in range(SEEDS)] for s in ABS}
    r["rel"] = {str(s): [ev(add_relative(T.params, s, jax.random.PRNGKey(k))) for k in range(SEEDS)] for s in REL}
    r["zero_only"] = {str(s): [ev(add_gaussian_masked(T.params, s, jax.random.PRNGKey(k), "zero")) for k in range(SEEDS)] for s in ABS}
    r["nonzero_only"] = {str(s): [ev(add_gaussian_masked(T.params, s, jax.random.PRNGKey(k), "nonzero")) for k in range(SEEDS)] for s in ABS}
    r["group"] = {g: {str(s): [ev(add_gaussian_group(T.params, s, jax.random.PRNGKey(k), g)) for k in range(SEEDS)] for s in GSIG} for g in GROUPS}
    res[name] = r
    print(name, "done", round(time.time() - t0), "s", flush=True)
    json.dump(res, open("results2.json", "w"))
print("ALL DONE", flush=True)
