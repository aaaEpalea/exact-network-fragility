"""Control: are programs WITHOUT any counting also fragile?"""
import json, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
from tracr.rasp import rasp
from tracr.compiler import compiling, lib
from scaling import s50
VOC = [1, 2, 3, 4, 5, 6]; SIG = [float(x) for x in np.logspace(-5, -0.5, 14)]; SEEDS = 4
def p_map(): return rasp.Map(lambda x: x % 3, rasp.tokens)                      # pure per-token lookup (no attention)
def p_shift():                                                                   # hard attention to previous position (no counting)
    sel = rasp.Select(rasp.indices, rasp.indices, lambda k, q: k == q - 1)
    return rasp.Aggregate(sel, rasp.tokens, default=0)
exp = {"map": lambda x: [t % 3 for t in x], "shift_prev": lambda x: [0] + list(x[:-1]),
       "length": lambda x: [len(x)] * len(x), "hist": lambda x: [x.count(v) for v in x], "reverse": lambda x: list(x[::-1])}
progs = {"map": p_map, "shift_prev": p_shift, "length": lib.make_length, "hist": lib.make_hist, "reverse": lambda: lib.make_reverse(rasp.tokens)}
res = {}
for N in [8, 16]:
    for name, mk in progs.items():
        S = max(100, int(1.5 * N * N))
        try: m = compiling.compile_rasp_to_model(mk(), vocab=set(VOC), max_seq_len=N, compiler_bos="BOS", mlp_exactness=S)
        except Exception as e: print(name, N, "compile failed", repr(e)[:100]); continue
        rng = np.random.default_rng(0); B = {}
        for L in range(2, N):
            xs = [list(map(int, rng.choice(VOC, size=L))) for _ in range(30)]
            B[L] = (jnp.array([m.input_encoder.encode(["BOS"] + x) for x in xs]), [exp[name](x) for x in xs])
        fwd = jax.jit(lambda p, t: m.forward(p, t).unembedded_output)
        def acc(p):
            ok = tot = 0
            for L, (toks, e) in B.items():
                out = np.asarray(fwd(p, toks))[:, 1:]
                for row, ee in zip(out, e):
                    got = m.output_encoder.decode(row.tolist()) if m.output_encoder is not None else row
                    ok += bool(np.allclose(np.asarray(got, float), np.asarray(ee, float), atol=0.5)); tot += 1
            return ok / tot
        def noisy(p, s, key):
            lv, tr = jax.tree_util.tree_flatten(p); ks = jax.random.split(key, len(lv))
            return jax.tree_util.tree_unflatten(tr, [jnp.asarray(a) + s * jax.random.normal(k, a.shape) for a, k in zip(lv, ks)])
        clean = acc(m.params)
        if clean < 0.999: print(f"{name:10s} N={N:2d} NOT EXACT clean={clean:.2f}"); continue
        v = s50({str(s): [acc(noisy(m.params, s, jax.random.PRNGKey(k))) for k in range(SEEDS)] for s in SIG})
        res[f"{name}|{N}"] = v; print(f"{name:10s} N={N:2d} uses_counting={'yes' if name in ('length','hist','reverse') else 'no ':3s} sigma50={v:.2e}", flush=True)
json.dump(res, open("control.json", "w"))
