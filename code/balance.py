"""Sensitivity-balanced compilation.

Optimises ONLY exact symmetries of a compiled (no-LayerNorm) transformer:
  * per-residual-dim scale s_j   (writers' columns * s_j, readers' rows / s_j)
  * per-layer q/k balance r      (Wq/r, Wk*r)
  * per-layer v/out balance u    (Wv/u, Wo*u)
  * per-layer MLP hidden scale h (W1/h, b1/h, W2*h)
so the network function is unchanged, while noise sensitivity changes.
Objective: expected squared error of the output under injected Gaussian weight noise.
"""
import json, sys, time, warnings
import numpy as np, jax, jax.numpy as jnp, optax
warnings.filterwarnings("ignore")
from tracr.compiler import compiling, lib
from hardening import CompiledNet
from scaling import s50
from chunk_count import chunked_length

VOC = [1, 2, 3, 4, 5, 6]


class Shim:  # minimal Task-like object for CompiledNet
    pass


def build(prog, N, S):
    m = compiling.compile_rasp_to_model(prog, vocab=set(VOC), max_seq_len=N, compiler_bos="BOS", mlp_exactness=S)
    sh = Shim(); sh.model, sh.prog = m, prog
    return m, CompiledNet(sh)


def dim_groups(m, net):
    """Group residual dims by first writer: 0=embeddings, then (attn_l, mlp_l) per layer. Output dims fixed (-1)."""
    P = {k: {n: np.asarray(a) for n, a in d.items()} for k, d in m.params.items()}
    d = net.d; gid = -np.ones(d, int); assigned = np.zeros(d, bool)
    emb = (np.abs(P["token_embed"]["embeddings"]).sum(0) + np.abs(P["pos_embed"]["embeddings"]).sum(0)) > 0
    gid[emb] = 0; assigned[emb] = True; g = 1
    for l in range(net.L):
        for part in ["attn/linear", "mlp/linear_2"]:
            w = P[f"transformer/layer_{l}/{part}"]["w"]
            wr = (np.abs(w).sum(0) > 0) & ~assigned
            gid[wr] = g; assigned[wr] = True; g += 1
    gid[~assigned] = 0
    gid[net.out_dims] = -1
    return gid, g


def n_theta(net, G):
    return G + 3 * net.L


def apply_sym(params, theta, gid, G, L):
    th = 3.0 * jnp.tanh(theta / 3.0)                         # log-scales clipped to [-3, 3]
    s_dim = jnp.where(gid >= 0, jnp.exp(th[jnp.clip(gid, 0)]), 1.0)
    P = {k: dict(v) for k, v in params.items()}
    P["token_embed"]["embeddings"] = params["token_embed"]["embeddings"] * s_dim[None]
    P["pos_embed"]["embeddings"] = params["pos_embed"]["embeddings"] * s_dim[None]
    for l in range(L):
        r, u, h = jnp.exp(th[G + 3 * l]), jnp.exp(th[G + 3 * l + 1]), jnp.exp(th[G + 3 * l + 2])
        pre = f"transformer/layer_{l}/"
        q, k, v, o = (dict(P[pre + "attn/" + n]) for n in ["query", "key", "value", "linear"])
        q["w"], k["w"], v["w"] = q["w"] / s_dim[:, None] / r, k["w"] / s_dim[:, None] * r, v["w"] / s_dim[:, None] / u
        q["b"], k["b"], v["b"] = q["b"] / r, k["b"] * r, v["b"] / u
        o["w"], o["b"] = o["w"] * u * s_dim[None], o["b"] * s_dim
        m1, m2 = dict(P[pre + "mlp/linear_1"]), dict(P[pre + "mlp/linear_2"])
        m1["w"], m1["b"] = m1["w"] / s_dim[:, None] / h, m1["b"] / h
        m2["w"], m2["b"] = m2["w"] * h * s_dim[None], m2["b"] * s_dim
        for n, dct in zip(["query", "key", "value", "linear"], [q, k, v, o]): P[pre + "attn/" + n] = dct
        P[pre + "mlp/linear_1"], P[pre + "mlp/linear_2"] = m1, m2
    return P


def noisy(P, sigma, key):
    leaves, tree = jax.tree_util.tree_flatten(P); ks = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(tree, [a + sigma * jax.random.normal(k, a.shape) for a, k in zip(leaves, ks)])


def run(tag, prog, N, S, steps=500, seed=0, c=1.0):
    t0 = time.time()
    m, net = build(prog, N, S)
    params = jax.tree_util.tree_map(jnp.asarray, m.params)
    if c != 1.0:
        params = {mod: {n: (a * c if mod.endswith("attn/query") else a) for n, a in d.items()} for mod, d in params.items()}
    gid_np, G = dim_groups(m, net); gid = jnp.array(gid_np)
    enc = lambda L, B, rng: jnp.array([m.input_encoder.encode(["BOS"] + list(map(int, rng.choice(VOC, size=L)))) for _ in range(B)])
    raw = lambda P, toks: net.forward({"base": P, "snaps": []}, toks)
    numeric = net.out_numeric
    fwd_out = (lambda P, toks: raw(P, toks).squeeze(-1)) if numeric else (lambda P, toks: jnp.argmax(raw(P, toks), -1).astype(jnp.float32))
    # eval batches (all lengths)
    rng_e = np.random.default_rng(1)
    ev = {L: enc(L, 24, rng_e) for L in range(1, N)}
    evf = jax.jit(fwd_out)
    def acc(P):
        ok = tot = 0
        for L, toks in ev.items():
            out = np.asarray(evf(P, toks))[:, 1:]; ok += int((np.abs(out - L) < 0.5).all(1).sum()); tot += len(out)
        return ok / tot
    SIG = [float(x) for x in np.logspace(-5, -1.5, 12)]
    def sweep(P, seeds=4):
        return s50({str(s): [acc(noisy(P, s, jax.random.PRNGKey(k))) for k in range(seeds)] for s in SIG})
    base_acc, base_s50 = acc(params), None
    base_s50 = sweep(params)
    print(f"[{tag}] N={N} d={net.d} L={net.L} groups={G} theta={n_theta(net, G)} clean={base_acc:.3f} sigma50(base)={base_s50:.2e} ({time.time()-t0:.0f}s)", flush=True)

    # --- optimise symmetry scalings under noise ---
    lens = sorted(set([2, max(2, N // 8), N // 4, N // 2, 3 * N // 4, N - 1]))
    rng = np.random.default_rng(seed + 10)
    train = {L: enc(L, 48, rng) for L in lens}
    theta = jnp.zeros(n_theta(net, G))
    opt = optax.adam(optax.cosine_decay_schedule(0.08, steps, 0.05)); st = opt.init(theta)
    lo, hi = np.log(0.5 * base_s50), np.log(8 * base_s50)

    @jax.jit
    def step(theta, st, toks, L, key):
        k1, k2 = jax.random.split(key)
        sigma = jnp.exp(jax.random.uniform(k1, (), minval=lo, maxval=hi))
        def loss(th):
            P = noisy(apply_sym(params, th, gid, G, net.L), sigma, k2)
            if numeric:
                err = (fwd_out(P, toks)[:, 1:] - L) ** 2
                return jnp.minimum(err, 4.0).mean()
            lg = raw(P, toks)[:, 1:] * 8.0
            return optax.softmax_cross_entropy_with_integer_labels(lg, jnp.full(lg.shape[:2], L, jnp.int32)).mean()
        l, g = jax.value_and_grad(loss)(theta)
        u, st = opt.update(g, st, theta); return optax.apply_updates(theta, u), st, l

    key = jax.random.PRNGKey(seed)
    for i in range(steps):
        L = lens[i % len(lens)]; key, sk = jax.random.split(key)
        theta, st, l = step(theta, st, train[L], jnp.float32(L), sk)
        if i % 100 == 0: print(f"   [{tag}] step {i} loss {float(l):.4f} ({time.time()-t0:.0f}s)", flush=True)
    P_new = apply_sym(params, theta, gid, G, net.L)
    # exactness / equivalence check
    rawf = jax.jit(raw)
    diff = max(float(jnp.abs(rawf(P_new, ev[L]) - rawf(params, ev[L])).max()) for L in ev)
    new_s50 = sweep(P_new, seeds=6)
    out = {"tag": tag, "N": N, "S": S, "clean_before": base_acc, "clean_after": acc(P_new), "max_output_diff": diff,
           "s50_before": base_s50, "s50_after": new_s50, "gain": new_s50 / base_s50, "theta": np.asarray(3 * np.tanh(theta / 3)).tolist(),
           "log_scale_range": [float(np.min(3 * np.tanh(theta / 3))), float(np.max(3 * np.tanh(theta / 3)))]}
    print(f"== [{tag}] clean {base_acc:.3f}->{out['clean_after']:.3f}; max|out diff|={diff:.2e}; sigma50 {base_s50:.2e} -> {new_s50:.2e}  ({out['gain']:.1f}x) ({time.time()-t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    res = {}
    if which in ("both", "chunk"):
        res["chunk4"] = run("chunk g=4", chunked_length(N, 4), N, 100)
        json.dump(res, open(f"balance_{N}_chunk.json", "w"))
    if which in ("both", "plain"):
        res["plain"] = run("plain", lib.make_length(), N, int(1.2 * N * N))
        json.dump(res, open(f"balance_{N}_plain.json", "w"))
    print("DONE", flush=True)
