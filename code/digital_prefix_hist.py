"""Prefix histogram: y_t = number of times token x_t has appeared in x_1..x_t (inclusive).

Design A (Tracr, causal transformer): lib.make_hist() compiled with causal=True.
Design B (digital RNN): one one-hot counter per token value; the token one-hot gates which counter
shifts; a restoring 'snap' (clip via two ReLUs) after each step; readout selects the current token's counter.
Noise: Gaussian N(0, sigma^2) added to EVERY parameter entry (zeros included), reused at every step.
"""
import json, sys, time, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
from tracr.rasp import rasp
from tracr.compiler import compiling, lib
from scaling import s50

V = 6; VOC = list(range(1, V + 1))
SIG = [float(x) for x in np.logspace(-5, -0.3, 15)]; SEEDS = 4; BATCH = 24


def prefix_hist(x):
    return [x[:i + 1].count(x[i]) for i in range(len(x))]


# ---------------------------------------------------------------- digital RNN
def build_digital(N, snap=True):
    n = N + 1; D = V * n
    P = np.zeros((n, n)); P[np.arange(1, n), np.arange(n - 1)] = 1.0; P[n - 1, n - 1] = 1.0
    I = np.eye(D)
    G = np.kron(np.eye(V), np.ones((n, 1)))                      # D x V: token v -> its counter's slots
    p = {"MP": np.kron(np.eye(V), P), "G1": G, "b1": -np.ones(D),
         "I1": I, "G2": -G, "b2": np.zeros(D),
         "C1": I, "C2": I,
         "S1": 2 * I, "bs1": -0.5 * np.ones(D), "S2": 2 * I, "bs2": -1.5 * np.ones(D), "C3": I, "C4": -I,
         "I3": I, "G3": G, "b3": -np.ones(D), "R": np.tile(np.eye(n), (1, V)),
         "s0": np.tile(np.eye(n)[0], V)}
    return {k: jnp.asarray(v, jnp.float32) for k, v in p.items()}


def run_digital(p, X, snap=True):
    """X: (B, L) int tokens in 0..V-1. Returns predicted counts (B, L)."""
    B, L = X.shape
    xs = jax.nn.one_hot(X, V)                                     # (B, L, V)
    s = jnp.broadcast_to(p["s0"], (B, p["s0"].shape[0]))
    def step(s, x):
        u1 = jax.nn.relu(s @ p["MP"].T + x @ p["G1"].T + p["b1"])
        u2 = jax.nn.relu(s @ p["I1"].T + x @ p["G2"].T + p["b2"])
        z = u1 @ p["C1"].T + u2 @ p["C2"].T
        if snap:
            a1 = jax.nn.relu(z @ p["S1"].T + p["bs1"]); a2 = jax.nn.relu(z @ p["S2"].T + p["bs2"])
            s2 = a1 @ p["C3"].T + a2 @ p["C4"].T
        else:
            s2 = z
        r = jax.nn.relu(s2 @ p["I3"].T + x @ p["G3"].T + p["b3"])
        return s2, jnp.argmax(r @ p["R"].T, -1)
    _, ys = jax.lax.scan(step, s, jnp.swapaxes(xs, 0, 1))
    return jnp.swapaxes(ys, 0, 1)


def noisy_tree(P, sigma, key):
    ks = dict(zip(P.keys(), jax.random.split(key, len(P))))
    return {k: v + sigma * jax.random.normal(ks[k], v.shape) for k, v in P.items()}


def eval_digital(N, snap, rng):
    p = build_digital(N)
    L = N - 1
    X = np.stack([rng.integers(0, V, size=L) for _ in range(BATCH)])
    exp = np.array([prefix_hist([int(t) + 1 for t in row]) for row in X])
    f = jax.jit(lambda P, X: run_digital(P, X, snap))
    clean = float((np.asarray(f(p, jnp.array(X))) == exp).all(1).mean())
    curve = {}
    for s in SIG:
        accs = []
        for k in range(SEEDS):
            P = noisy_tree(p, s, jax.random.PRNGKey(k))
            accs.append(float((np.asarray(f(P, jnp.array(X))) == exp).all(1).mean()))
        curve[str(s)] = accs
    nparams = int(sum(np.prod(v.shape) for v in p.values()))
    return clean, s50(curve), nparams


# ---------------------------------------------------------------- Tracr causal
def eval_tracr(N, c=1.0):
    S = max(100, int(1.5 * N * N))
    m = compiling.compile_rasp_to_model(lib.make_hist(), vocab=set(VOC), max_seq_len=N, compiler_bos="BOS",
                                        mlp_exactness=S, causal=True)
    rng = np.random.default_rng(0); L = N - 1
    xs = [list(map(int, rng.choice(VOC, size=L))) for _ in range(BATCH)]
    toks = jnp.array([m.input_encoder.encode(["BOS"] + x) for x in xs]); exp = [prefix_hist(x) for x in xs]
    fwd = jax.jit(lambda p, t: m.forward(p, t).unembedded_output)
    def acc(p):
        out = np.asarray(fwd(p, toks))[:, 1:]; ok = 0
        for row, e in zip(out, exp):
            got = m.output_encoder.decode(row.tolist()); ok += bool(np.allclose(np.asarray(got, float), np.asarray(e, float), atol=0.5))
        return ok / len(xs)
    P0 = {mod: {n: (jnp.asarray(a) * c if (mod.endswith("attn/query")) else jnp.asarray(a)) for n, a in d.items()} for mod, d in m.params.items()}
    clean = acc(P0)
    if clean < 0.999: return clean, float("nan"), sum(int(np.prod(np.shape(a))) for d in m.params.values() for a in d.values())
    def noisy(p, s, key):
        lv, tr = jax.tree_util.tree_flatten(p); ks = jax.random.split(key, len(lv))
        return jax.tree_util.tree_unflatten(tr, [a + s * jax.random.normal(k, a.shape) for a, k in zip(lv, ks)])
    v = s50({str(s): [acc(noisy(P0, s, jax.random.PRNGKey(k))) for k in range(SEEDS)] for s in SIG})
    return clean, v, sum(int(np.prod(np.shape(a))) for d in m.params.values() for a in d.values())


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}; rng = np.random.default_rng(0); t0 = time.time()
    # exhaustive exactness certificate for the digital design at small N (N=6 -> sequences of length 5)
    import itertools
    pN = build_digital(6); allx = np.array(list(itertools.product(range(V), repeat=5)))
    exp = np.array([prefix_hist([int(t) + 1 for t in r]) for r in allx])
    got = np.asarray(jax.jit(lambda P, X: run_digital(P, X, True))(pN, jnp.array(allx)))
    print(f"digital exhaustive check N=6: {(got == exp).all(1).sum()}/{len(allx)} sequences exact", flush=True)
    if which in ("all", "digital"):
        for N in [8, 16, 32, 64, 128]:
            for snap in [True, False]:
                clean, v, npar = eval_digital(N, snap, rng)
                res[f"digital_{'snap' if snap else 'nosnap'}|{N}"] = {"clean": clean, "s50": v, "params": npar}
                print(f"digital {'snap  ' if snap else 'nosnap'} N={N:3d} clean={clean:.2f} sigma50={v:.2e} params={npar} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open("digital_prefix_hist.json", "w"))
    if which in ("all", "tracr"):
        for N in [8, 16, 32]:
            for c in [1.0, 0.25]:
                clean, v, npar = eval_tracr(N, c)
                res[f"tracr_c{c}|{N}"] = {"clean": clean, "s50": v, "params": npar}
                print(f"tracr   causal c={c} N={N:3d} clean={clean:.2f} sigma50={v:.2e} params={npar} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open("digital_prefix_hist.json", "w"))
    print("DONE", flush=True)
