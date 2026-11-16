"""Does compiled counting robustness degrade with max sequence length N?"""
import json, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
from tracr.rasp import rasp
from tracr.compiler import compiling, lib

VOC = [1, 2, 3, 4, 5, 6]
SIG = [float(x) for x in np.logspace(-5, -1, 13)]
SEEDS = 5

def s50(curve):
    xs = np.array([float(k) for k in curve]); ys = np.array([np.mean(v) for v in curve.values()])
    if ys[0] < 0.5: return float("nan")
    for i in range(1, len(xs)):
        if ys[i] < 0.5:
            f = (ys[i-1]-0.5)/(ys[i-1]-ys[i]); return float(np.exp(np.log(xs[i-1])+f*(np.log(xs[i])-np.log(xs[i-1]))))
    return float("inf")

if __name__ == '__main__':
    res = {}
    for N in [4, 6, 8, 12, 16, 24]:
        m = compiling.compile_rasp_to_model(lib.make_length(), vocab=set(VOC), max_seq_len=N, compiler_bos="BOS")
        rng = np.random.default_rng(0)
        batches = {}
        for L in range(1, N):
            xs = [list(map(int, rng.choice(VOC, size=L))) for _ in range(30)]
            batches[L] = jnp.array([m.input_encoder.encode(["BOS"] + x) for x in xs])
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
        res[N] = {}
        for c in [1.0, 0.25]:
            p = scaled(c); clean = acc(p)
            r = {"clean": clean}
            if clean > 0.999:
                r["abs"] = {str(s): [acc(noisy(p, s, jax.random.PRNGKey(k))) for k in range(SEEDS)] for s in SIG}
                r["s50"] = s50(r["abs"])
            res[N][str(c)] = r
            print(f"N={N:2d} c={c}: clean={clean:.3f}  sigma50={r.get('s50', float('nan')):.2e}", flush=True)
    json.dump(res, open("scaling.json", "w"))
