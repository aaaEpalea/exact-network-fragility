"""Sensitivity balancing generalised to reverse / hist / sort_unique / frac_prevs at larger N."""
import json, sys, time, warnings
import numpy as np, jax, jax.numpy as jnp, optax
warnings.filterwarnings("ignore")
from tracr.rasp import rasp
from tracr.compiler import compiling, lib
from hardening import CompiledNet
from scaling import s50
from balance import Shim, dim_groups, n_theta, apply_sym, noisy

VOC = [1, 2, 3, 4, 5, 6]


def make_prog(name):
    return {"reverse": lambda: lib.make_reverse(rasp.tokens), "hist": lib.make_hist,
            "sort_unique": lambda: lib.make_sort_unique(rasp.tokens, rasp.tokens),
            "frac_prevs": lambda: lib.make_frac_prevs(rasp.tokens == 1)}[name]()


def expected(name, x):
    if name == "reverse": return list(x[::-1])
    if name == "hist": return [x.count(v) for v in x]
    if name == "sort_unique": return sorted(x)
    if name == "frac_prevs": return [sum(1 for t in x[:i + 1] if t == 1) / (i + 1) for i in range(len(x))]


def sample(name, L, B, rng):
    out = []
    for _ in range(B):
        if name == "sort_unique": x = rng.choice(VOC, size=L, replace=False)
        elif name == "frac_prevs": x = rng.choice(VOC, size=L, p=[.4, .12, .12, .12, .12, .12])
        else: x = rng.choice(VOC, size=L)
        out.append([int(v) for v in x])
    return out


def run(name, N, c=1.0, steps=500, seed=0, S=None):
    t0 = time.time(); tag = f"{name} N={N} c={c}"
    S = S or max(100, int(1.5 * N * N))
    prog = make_prog(name)
    m = compiling.compile_rasp_to_model(prog, vocab=set(VOC), max_seq_len=N, compiler_bos="BOS", mlp_exactness=S)
    sh = Shim(); sh.model, sh.prog = m, prog; net = CompiledNet(sh)
    from tracr.transformer import encoder as _E
    categorical = isinstance(m.output_encoder, _E.CategoricalEncoder)
    if categorical:   # order output dims by the compiler's own class index, not by label string order
        v2i = dict(m.output_encoder.encoding_map)            # value -> class index (compiler's own order)
        i2v = {i: v for v, i in v2i.items()}
        net.out_dims = [m.residual_labels.index(f"{prog.label}:{i2v[i]}") for i in range(len(i2v))]
    params = jax.tree_util.tree_map(jnp.asarray, m.params)
    if c != 1.0:
        params = {mod: {n: (a * c if mod.endswith("attn/query") else a) for n, a in d.items()} for mod, d in params.items()}
    gid_np, G = dim_groups(m, net); gid = jnp.array(gid_np)
    maxL = 6 if name == "sort_unique" else N - 1
    def batch(L, B, rng):
        xs = sample(name, L, B, rng)
        toks = jnp.array([m.input_encoder.encode(["BOS"] + x) for x in xs])
        ys = [expected(name, x) for x in xs]
        y = jnp.array([[v2i[v] for v in row] for row in ys]) if categorical else jnp.array(ys, dtype=jnp.float32)
        return toks, y
    raw = lambda P, toks: net.forward({"base": P, "snaps": []}, toks)
    pred = (lambda P, toks: jnp.argmax(raw(P, toks), -1)) if categorical else (lambda P, toks: raw(P, toks).squeeze(-1))
    rng_e = np.random.default_rng(1)
    ev = {L: batch(L, 24, rng_e) for L in range(2, maxL + 1)}
    predj = jax.jit(pred)
    def acc(P):
        ok = tot = 0
        for L, (toks, y) in ev.items():
            p = np.asarray(predj(P, toks))[:, 1:]; yy = np.asarray(y)
            good = (p == yy).all(1) if categorical else (np.abs(p - yy) < 0.05).all(1)
            ok += int(good.sum()); tot += len(good)
        return ok / tot
    SIG = [float(x) for x in np.logspace(-5, -1.5, 12)]
    sweep = lambda P, seeds=4: s50({str(s): [acc(noisy(P, s, jax.random.PRNGKey(k))) for k in range(seeds)] for s in SIG})
    base_acc = acc(params)
    if base_acc < 0.999:
        print(f"== [{tag}] NOT EXACT at zero noise (clean={base_acc:.3f}); skipped", flush=True)
        return {"tag": tag, "clean_before": base_acc, "skipped": True}
    base_s50 = sweep(params)
    print(f"[{tag}] d={net.d} L={net.L} theta={n_theta(net, G)} clean=1.000 sigma50(base)={base_s50:.2e} ({time.time()-t0:.0f}s)", flush=True)

    lens = sorted(set([2, max(2, maxL // 4), maxL // 2, 3 * maxL // 4, maxL]))
    rng = np.random.default_rng(seed + 10)
    train = {L: batch(L, 48, rng) for L in lens}
    theta = jnp.zeros(n_theta(net, G))
    opt = optax.adam(optax.cosine_decay_schedule(0.08, steps, 0.05)); st = opt.init(theta)
    lo, hi = np.log(0.5 * base_s50), np.log(8 * base_s50)

    @jax.jit
    def step(theta, st, toks, y, key):
        k1, k2 = jax.random.split(key)
        sigma = jnp.exp(jax.random.uniform(k1, (), minval=lo, maxval=hi))
        def loss(th):
            P = noisy(apply_sym(params, th, gid, G, net.L), sigma, k2)
            out = raw(P, toks)[:, 1:]
            if categorical: return optax.softmax_cross_entropy_with_integer_labels(out * 8.0, y).mean()
            return jnp.minimum((out.squeeze(-1) - y) ** 2, 4.0).mean()
        l, g = jax.value_and_grad(loss)(theta)
        u, st = opt.update(g, st, theta); return optax.apply_updates(theta, u), st, l

    key = jax.random.PRNGKey(seed)
    for i in range(steps):
        L = lens[i % len(lens)]; key, sk = jax.random.split(key)
        toks, y = train[L]; theta, st, l = step(theta, st, toks, y, sk)
    P_new = apply_sym(params, theta, gid, G, net.L)
    new_s50 = sweep(P_new, seeds=6); clean_after = acc(P_new)
    out = {"tag": tag, "name": name, "N": N, "c": c, "S": S, "clean_after": clean_after, "s50_before": base_s50,
           "s50_after": new_s50, "gain": new_s50 / base_s50}
    print(f"== [{tag}] clean 1.000->{clean_after:.3f}; sigma50 {base_s50:.2e} -> {new_s50:.2e} ({out['gain']:.1f}x) ({time.time()-t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    res = {}
    cfgs = [("reverse", 12), ("reverse", 16), ("hist", 16), ("hist", 24), ("frac_prevs", 16), ("sort_unique", 8)]
    for name, N in cfgs:
        for c in [1.0, 0.25]:
            try:
                res[f"{name}|{N}|{c}"] = run(name, N, c)
            except Exception as e:
                print(f"== [{name} N={N} c={c}] FAILED {repr(e)[:160]}", flush=True)
            json.dump(res, open("balance2.json", "w"))
    print("DONE", flush=True)
