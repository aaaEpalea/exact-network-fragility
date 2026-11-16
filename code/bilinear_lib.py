"""Second-order (bilinear) digital designs: single/multi-component FSM compile, and weight-tied template composition."""
import numpy as np, jax, jax.numpy as jnp
import digital_compiler as dc
V = 6

def build_bil(spec):
    ns = [c.n for c in spec.comps]; off = np.concatenate([[0], np.cumsum(ns)]); D = int(off[-1]); W = np.zeros((V, D, D), np.float32)
    for ci, c in enumerate(spec.comps):
        for v in range(V):
            for j in range(c.n): W[v, off[ci] + int(c.delta[v, j]), off[ci] + j] = 1.0
    h = dc.heads_of(spec)[0]; T = np.zeros((h["n_out"], D), np.float32)
    for i in range(spec.comps[h["comp"]].n): T[h["table"][i], off[h["comp"]] + i] = 1.0
    s0 = np.concatenate([np.eye(c.n)[c.init] for c in spec.comps]).astype(np.float32)
    return {"W": jnp.array(W), "T": jnp.array(T), "s0": jnp.array(s0)}

def run_bil(p, X):
    xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); s = jnp.broadcast_to(p["s0"], (X.shape[0], p["s0"].shape[0]))
    def f(s, x):
        z = jnp.einsum("bv,vij,bj->bi", x, p["W"], s); s2 = jnp.clip(2.0 * z - 0.5, 0.0, 1.0); return s2, jnp.argmax(s2 @ p["T"].T, -1)
    _, ys = jax.lax.scan(f, s, xs); return jnp.swapaxes(ys, 0, 1)

def build_tied(N, pairs):
    n = 2 * N + 1; Pu = np.zeros((n, n)); Pu[np.arange(1, n), np.arange(n - 1)] = 1; Pu[n - 1, n - 1] = 1; Pd = np.zeros((n, n)); Pd[np.arange(n - 1), np.arange(1, n)] = 1; Pd[0, 0] = 1
    G = np.zeros((len(pairs), 3, V))
    for j, (a, b) in enumerate(pairs):
        for t in range(V): G[j, 0 if t == a else 1 if t == b else 2, t] = 1
    T = np.zeros((2, n)); T[0, :N + 1] = 1; T[1, N + 1:] = 1
    return {"P": jnp.array(np.stack([Pu, Pd, np.eye(n)]), jnp.float32), "G": jnp.array(G, jnp.float32), "T": jnp.array(T, jnp.float32), "s0": jnp.array(np.eye(n)[N], jnp.float32)}

def run_tied(p, X, k):
    xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); s = jnp.broadcast_to(p["s0"], (X.shape[0], k, p["s0"].shape[0]))
    def f(s, x):
        g = jnp.einsum("krv,bv->bkr", p["G"], x); z = jnp.einsum("bkr,rij,bkj->bki", g, p["P"], s); s2 = jnp.clip(2.0 * z - 0.5, 0.0, 1.0)
        return s2, jnp.argmax(s2 @ p["T"].T, -1)
    _, ys = jax.lax.scan(f, s, xs); return jnp.swapaxes(ys, 0, 1)

def noisy(P, s, key, rel=False):
    return {k: v + s * (jnp.sqrt(jnp.mean(v ** 2)) if rel else 1.0) * jax.random.normal(kk, v.shape) for (k, v), kk in zip(P.items(), jax.random.split(key, len(P)))}
