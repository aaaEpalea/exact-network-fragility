"""Width-matched baseline: r independent compiled Tracr copies (independent noise), vote by summing output activations."""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
from tracr.compiler import compiling, lib
from scaling import s50
from digital_prefix_hist import prefix_hist, SIG, SEEDS, BATCH, V
res = {}; t0 = time.time()
for N, Rs in [(16, [1, 4, 18]), (32, [1, 4, 23])]:
    prog = lib.make_hist(); S = max(100, int(1.5 * N * N))
    m = compiling.compile_rasp_to_model(prog, vocab=set(range(1, V + 1)), max_seq_len=N, compiler_bos="BOS", mlp_exactness=S, causal=True)
    labels = m.residual_labels; v2i = dict(m.output_encoder.encoding_map); i2v = {i: v for v, i in v2i.items()}
    out_dims = [labels.index(f"{prog.label}:{i2v[i]}") for i in range(len(i2v))]
    rng = np.random.default_rng(0); L = N - 1
    xs = [list(map(int, rng.choice(range(1, V + 1), size=L))) for _ in range(BATCH)]
    toks = jnp.array([m.input_encoder.encode(["BOS"] + x) for x in xs]); exp = np.array([[v2i[c] for c in prefix_hist(x)] for x in xs])
    f = jax.jit(lambda p, t: m.forward(p, t).transformer_output.output[..., out_dims])
    P0 = jax.tree_util.tree_map(jnp.asarray, m.params); npar = sum(int(np.prod(a.shape)) for a in jax.tree_util.tree_leaves(P0))
    def noisy(p, s, key):
        lv, tr = jax.tree_util.tree_flatten(p); ks = jax.random.split(key, len(lv))
        return jax.tree_util.tree_unflatten(tr, [a + s * jax.random.normal(k, a.shape) for a, k in zip(lv, ks)])
    def acc(r, s, k):
        logits = sum(f(noisy(P0, s, jax.random.fold_in(jax.random.PRNGKey(k), i)), toks) for i in range(r))
        return float((np.asarray(jnp.argmax(logits, -1))[:, 1:] == exp).all(1).mean())
    clean = acc(1, 0.0, 0)
    for r in Rs:
        v = s50({str(s): [acc(r, s, k) for k in range(SEEDS)] for s in SIG})
        res[f"N{N}|r{r}"] = {"s50": v, "params": r * npar, "clean": clean}
        print(f"Tracr prefix-hist N={N:2d} copies r={r:2d} params={r*npar:7d} sigma50={v:.2e} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open("width_matched.json", "w"))
print("DONE", flush=True)
