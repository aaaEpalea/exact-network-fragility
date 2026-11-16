import json, warnings, time
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
from tracr.rasp import rasp
from tracr.compiler import compiling, lib
from scaling import s50, VOC
SIG = [float(x) for x in np.logspace(-5, -1, 13)]; SEEDS = 4
res = {}; t0 = time.time()
for N in [8, 12, 16, 24]:
    for S in [100, 300, 1000, 3000]:
        m = compiling.compile_rasp_to_model(lib.make_length(), vocab=set(VOC), max_seq_len=N, compiler_bos="BOS", mlp_exactness=S)
        rng = np.random.default_rng(0)
        batches = {L: jnp.array([m.input_encoder.encode(["BOS"] + list(map(int, rng.choice(VOC, size=L)))) for _ in range(30)]) for L in range(1, N)}
        fwd = jax.jit(lambda p, t: m.forward(p, t).unembedded_output)
        def acc(params):
            ok = tot = 0
            for L, toks in batches.items():
                out = np.asarray(fwd(params, toks))[:, 1:]
                ok += int((np.abs(out - L) < 0.5).all(1).sum()); tot += len(out)
            return ok / tot
        def scaled(c):
            return {mod: {n: (jnp.asarray(a) * c if mod.endswith("attn/query") and n in ("w", "b") else jnp.asarray(a)) for n, a in d.items()} for mod, d in m.params.items()}
        def noisy(p, s, key):
            leaves, tree = jax.tree_util.tree_flatten(p); ks = jax.random.split(key, len(leaves))
            return jax.tree_util.tree_unflatten(tree, [a + s * jax.random.normal(k, a.shape) for a, k in zip(leaves, ks)])
        for c in [1.0, 0.25]:
            p = scaled(c); clean = acc(p); r = {"clean": clean}
            if clean > 0.999:
                r["abs"] = {str(s): [acc(noisy(p, s, jax.random.PRNGKey(k))) for k in range(SEEDS)] for s in SIG}; r["s50"] = s50(r["abs"])
            res.setdefault(str(N), {}).setdefault(str(S), {})[str(c)] = r
            print(f"N={N:2d} S={S:4d} c={c}: clean={clean:.3f} s50={r.get('s50', float('nan')):.2e} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open("scaling2.json", "w"))
print("DONE", flush=True)
