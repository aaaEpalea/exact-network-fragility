"""Trace fragility to counting: static analysis of each RASP program + noise tolerance."""
import json, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
from tracr.rasp import rasp
from tracr.compiler import compiling, lib
from tracr.transformer import encoder as E
from scaling import s50

VOC = [1, 2, 3, 4, 5, 6]; PAR = ["(", ")"]
SIG = [float(x) for x in np.logspace(-5, -0.5, 14)]; SEEDS = 4

def make(name, N):
    t = rasp.tokens
    if name == "map": return rasp.Map(lambda x: x % 3, t), VOC, "free"
    if name == "mean_token": return rasp.Aggregate(rasp.Select(t, t, lambda k, q: True), rasp.numerical(rasp.Map(lambda x: x, t))), VOC, "free"
    if name == "frac_prevs": return lib.make_frac_prevs(t == 1), VOC, "ones"
    if name == "length": return lib.make_length(), VOC, "free"
    if name == "hist": return lib.make_hist(), VOC, "free"
    if name == "reverse": return lib.make_reverse(t), VOC, "free"
    if name == "sort_unique": return lib.make_sort_unique(t, t), VOC, "unique"
    if name == "sort": return lib.make_sort(t, t, max_seq_len=N, min_key=1), VOC, "free"
    if name == "sort_freq": return lib.make_sort_freq(N), VOC, "free"
    if name == "pair_balance": return lib.make_pair_balance(t, "(", ")"), PAR, "paren"

def node_types(expr):
    seen, kinds = set(), set()
    def walk(e):
        if id(e) in seen: return
        seen.add(id(e)); kinds.add(type(e).__name__)
        for c in e.children: walk(c)
    walk(expr); return kinds

def sample(kind, L, rng, B=20):
    out = []
    for _ in range(B):
        if kind == "unique": out.append(list(map(int, rng.choice(VOC, size=min(L, 6), replace=False))))
        elif kind == "paren": out.append([PAR[i] for i in rng.integers(0, 2, size=L)])
        elif kind == "ones": out.append(list(map(int, rng.choice(VOC, size=L, p=[.4, .12, .12, .12, .12, .12]))))
        else: out.append(list(map(int, rng.choice(VOC, size=L))))
    return out

res = {}
for N in [8, 16]:
    for name in ["map", "mean_token", "frac_prevs", "pair_balance", "length", "hist", "reverse", "sort_unique", "sort", "sort_freq"]:
        if name == "sort_unique" and N > 8: continue
        try:
            prog, vocab, kind = make(name, N)
            kinds = node_types(prog)
            S = max(100, int(1.5 * N * N))
            m = compiling.compile_rasp_to_model(prog, vocab=set(vocab), max_seq_len=N, compiler_bos="BOS", mlp_exactness=S)
        except Exception as e:
            print(f"{name:12s} N={N:2d} compile failed: {repr(e)[:90]}", flush=True); continue
        rng = np.random.default_rng(0); B = {}
        for L in range(2, N):
            xs = sample(kind, min(L, 6) if kind == "unique" else L, rng)
            B[L] = (jnp.array([m.input_encoder.encode(["BOS"] + x) for x in xs]), [prog(x) for x in xs])
        cat = isinstance(m.output_encoder, E.CategoricalEncoder)
        fwd = jax.jit(lambda p, t: m.forward(p, t).unembedded_output)
        def acc(p):
            ok = tot = 0
            for L, (toks, e) in B.items():
                out = np.asarray(fwd(p, toks))[:, 1:]
                for row, ee in zip(out, e):
                    got = m.output_encoder.decode(row.tolist()) if m.output_encoder is not None else row
                    try:
                        good = (list(got) == list(ee)) if cat else bool(np.allclose(np.asarray(got, float), np.asarray(ee, float), atol=0.05))
                    except Exception: good = False
                    ok += good; tot += 1
            return ok / tot
        clean = acc(m.params)
        feats = {"counting(SelectorWidth)": "SelectorWidth" in kinds, "averaging(Aggregate)": "Aggregate" in kinds}
        if clean < 0.999:
            print(f"{name:12s} N={N:2d} NOT EXACT clean={clean:.2f}  {feats}", flush=True); res[f"{name}|{N}"] = {"clean": clean, **feats}; continue
        def noisy(p, s, key):
            lv, tr = jax.tree_util.tree_flatten(p); ks = jax.random.split(key, len(lv))
            return jax.tree_util.tree_unflatten(tr, [jnp.asarray(a) + s * jax.random.normal(k, a.shape) for a, k in zip(lv, ks)])
        v = s50({str(s): [acc(noisy(m.params, s, jax.random.PRNGKey(k))) for k in range(SEEDS)] for s in SIG})
        res[f"{name}|{N}"] = {"clean": clean, "s50": v, **feats}
        print(f"{name:12s} N={N:2d} sigma50={v:.2e}  counting={feats['counting(SelectorWidth)']!s:5s} averaging={feats['averaging(Aggregate)']!s:5s}", flush=True)
        json.dump(res, open("audit_counting.json", "w"))
print("DONE", flush=True)
