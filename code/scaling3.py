"""Scaling of clean exactness + noise tolerance with max length N for several Tracr programs."""
import json, time, warnings, itertools
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
from tracr.rasp import rasp
from tracr.compiler import compiling, lib
from scaling import s50
VOC = [1, 2, 3, 4, 5, 6]
SIG = [float(x) for x in np.logspace(-5, -1, 13)]; SEEDS = 4

def prog_for(name):
    if name == "reverse": return lib.make_reverse(rasp.tokens)
    if name == "hist": return lib.make_hist()
    if name == "sort_unique": return lib.make_sort_unique(rasp.tokens, rasp.tokens)
    if name == "frac_prevs": return lib.make_frac_prevs(rasp.tokens == 1)

def expected(name, x):
    if name == "reverse": return list(x[::-1])
    if name == "hist": return [x.count(v) for v in x]
    if name == "sort_unique": return sorted(x)
    if name == "frac_prevs":
        return [sum(1 for t in x[:i + 1] if t == 1) / (i + 1) for i in range(len(x))]

res = {}; t0 = time.time()
for name in ["reverse", "hist", "sort_unique", "frac_prevs"]:
    for N in [6, 8, 12, 16]:
        if name == "sort_unique" and N > 8: continue          # needs unique tokens; vocab=6 caps length
        S = max(100, int(1.5 * N * N))
        m = compiling.compile_rasp_to_model(prog_for(name), vocab=set(VOC), max_seq_len=N, compiler_bos="BOS", mlp_exactness=S)
        rng = np.random.default_rng(0); batches = {}
        for L in range(2, N):
            if name == "sort_unique":
                if L > 6: continue
                xs = [list(map(int, rng.choice(VOC, size=L, replace=False))) for _ in range(30)]
            elif name == "frac_prevs":
                xs = [list(map(int, rng.choice(VOC, size=L, p=[.4, .12, .12, .12, .12, .12]))) for _ in range(30)]
            else:
                xs = [list(map(int, rng.choice(VOC, size=L))) for _ in range(30)]
            batches[L] = (jnp.array([m.input_encoder.encode(["BOS"] + x) for x in xs]), [expected(name, x) for x in xs])
        fwd = jax.jit(lambda p, t: m.forward(p, t).unembedded_output)
        def acc(params):
            ok = tot = 0
            for L, (toks, exp) in batches.items():
                out = np.asarray(fwd(params, toks))[:, 1:]
                for row, e in zip(out, exp):
                    got = m.output_encoder.decode(row.tolist()) if m.output_encoder is not None else row
                    got = np.asarray(got, dtype=float) if m.output_encoder is None else np.asarray(got, dtype=float)
                    ok += bool(np.allclose(got, np.asarray(e, dtype=float), atol=(0.05 if name == "frac_prevs" else 1e-6 if name in ("reverse", "sort_unique") else 0.5))); tot += 1
            return ok / tot
        def scaled(c):
            return {mod: {n: (jnp.asarray(a) * c if mod.endswith("attn/query") and n in ("w", "b") else jnp.asarray(a)) for n, a in d.items()} for mod, d in m.params.items()}
        def noisy(p, s, key):
            leaves, tree = jax.tree_util.tree_flatten(p); ks = jax.random.split(key, len(leaves))
            return jax.tree_util.tree_unflatten(tree, [a + s * jax.random.normal(k, a.shape) for a, k in zip(leaves, ks)])
        for c in [1.0, 0.25]:
            p = scaled(c); clean = acc(p); r = {"clean": clean, "S": S}
            if clean > 0.999:
                r["abs"] = {str(s): [acc(noisy(p, s, jax.random.PRNGKey(k))) for k in range(SEEDS)] for s in SIG}; r["s50"] = s50(r["abs"])
            res.setdefault(name, {}).setdefault(str(N), {})[str(c)] = r
            print(f"{name:11s} N={N:2d} S={S:4d} c={c}: clean={clean:.3f} s50={r.get('s50', float('nan')):.2e} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open("scaling3.json", "w"))
print("DONE", flush=True)
