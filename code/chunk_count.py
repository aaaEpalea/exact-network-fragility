"""Chunked counting: count in windows of g positions (fine resolution), then sum exact integers."""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
from tracr.rasp import rasp
from tracr.compiler import compiling, lib
from scaling import s50
VOC = [1, 2, 3, 4, 5, 6]
SIG = [float(x) for x in np.logspace(-5, -1, 13)]; SEEDS = 4

def chunked_length(N, g):
    ws = []
    for lo in range(0, N, g):
        sel = rasp.Select(rasp.indices, rasp.tokens, lambda k, q, lo=lo: lo <= k < lo + g)   # queries=tokens only so the node exists
        ws.append(rasp.numerical(rasp.SelectorWidth(sel)))
    while len(ws) > 1:                       # balanced tree of exact integer additions
        nxt = []
        for i in range(0, len(ws) - 1, 2):
            nxt.append(rasp.numerical(rasp.LinearSequenceMap(ws[i], ws[i + 1], 1, 1)))
        if len(ws) % 2: nxt.append(ws[-1])
        ws = nxt
    return ws[0]

def evaluate(prog, N, S, tag):
    m = compiling.compile_rasp_to_model(prog, vocab=set(VOC), max_seq_len=N, compiler_bos="BOS", mlp_exactness=S)
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
    out = {"layers": m.model_config.num_layers, "d": len(m.residual_labels)}
    for c in [1.0, 0.25]:
        p = scaled(c); clean = acc(p); r = {"clean": clean}
        if clean > 0.999:
            r["abs"] = {str(s): [acc(noisy(p, s, jax.random.PRNGKey(k))) for k in range(SEEDS)] for s in SIG}; r["s50"] = s50(r["abs"])
        out[str(c)] = r
        print(f"{tag:22s} N={N:2d} S={S:4d} c={c}: layers={out['layers']} d={out['d']} clean={clean:.3f} s50={r.get('s50', float('nan')):.2e}", flush=True)
    return out

if __name__ == "__main__":
    res = {}
    for N in [8, 12, 16, 24]:
        res[N] = {}
        for g in [4, 8]:
            if g >= N: continue
            try:
                res[N][f"chunk{g}"] = evaluate(chunked_length(N, g), N, 100 if g <= 8 else 300, f"chunked g={g}")
            except Exception as e:
                print(f"chunked g={g} N={N} FAILED: {repr(e)[:150]}", flush=True)
        json.dump(res, open("chunk_count.json", "w"))
    print("DONE", flush=True)
