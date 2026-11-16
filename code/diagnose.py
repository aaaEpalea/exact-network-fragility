"""Which component's noise tolerance decays with N? `length` program, noise applied to one tensor group at a time."""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
from tracr.compiler import compiling, lib
from scaling import s50
VOC = [1, 2, 3, 4, 5, 6]
SIG = [float(x) for x in np.logspace(-5, -0.5, 14)]; SEEDS = 3
GROUPS = {"embeddings": lambda m: "embed" in m, "attn_query": lambda m: m.endswith("attn/query"),
          "attn_key": lambda m: m.endswith("attn/key"), "attn_value": lambda m: m.endswith("attn/value"),
          "attn_out": lambda m: m.endswith("attn/linear"), "mlp_in": lambda m: m.endswith("mlp/linear_1"),
          "mlp_out": lambda m: m.endswith("mlp/linear_2"), "ALL": lambda m: True}
res = {}; t0 = time.time()
for N in [8, 16, 32, 48]:
    S = max(100, int(1.2 * N * N))
    m = compiling.compile_rasp_to_model(lib.make_length(), vocab=set(VOC), max_seq_len=N, compiler_bos="BOS", mlp_exactness=S)
    rng = np.random.default_rng(0)
    batches = {L: jnp.array([m.input_encoder.encode(["BOS"] + list(map(int, rng.choice(VOC, size=L)))) for _ in range(20)]) for L in range(1, N)}
    fwd = jax.jit(lambda p, t: m.forward(p, t).unembedded_output)
    def acc(params):
        ok = tot = 0
        for L, toks in batches.items():
            out = np.asarray(fwd(params, toks))[:, 1:]; ok += int((np.abs(out - L) < 0.5).all(1).sum()); tot += len(out)
        return ok / tot
    res[N] = {"d": len(m.residual_labels)}
    for g, sel in GROUPS.items():
        def noisy(s, key):
            out = {}; ks = iter(jax.random.split(key, 100))
            for mod, d in m.params.items():
                out[mod] = {n: (jnp.asarray(a) + s * jax.random.normal(next(ks), a.shape) if sel(mod) else jnp.asarray(a)) for n, a in d.items()}
            return out
        curve = {str(s): [acc(noisy(s, jax.random.PRNGKey(k))) for k in range(SEEDS)] for s in SIG}
        res[N][g] = s50(curve)
        print(f"N={N:2d} d={res[N]['d']:3d} {g:11s} sigma50={res[N][g]:.2e} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open("diagnose.json", "w"))
print("DONE", flush=True)
