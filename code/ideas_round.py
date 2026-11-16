"""Idea 1: second-order (bilinear) digital design.  Idea 2: weight-tied (shared-capacity) composition of template programs."""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
import digital_compiler as dc
from zeros_empirical import majority_ab, dyck_ab
from scaling import s50
V = 6; SIG = [float(x) for x in np.logspace(-3, 0.5, 18)]; NS = 6
def noisy(P, s, key, rel=False):
    ks = jax.random.split(key, len(P)); return {k: v + s * (jnp.sqrt(jnp.mean(v ** 2)) if rel else 1.0) * jax.random.normal(kk, v.shape) for (k, v), kk in zip(P.items(), ks)}
res = {}; t0 = time.time(); rng = np.random.default_rng(0)

# ---------------- Idea 1: bilinear / second-order design ----------------
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
prev = json.load(open("clip_activation.json"))
for mk, name in [(majority_ab, "maj(0>1)"), (dyck_ab, "dyck(0,1)")]:
    for N in [16, 32, 64]:
        spec = mk(N, 0, 1); p = build_bil(spec); X = rng.choice(V, size=(24, N - 1)); ref = np.array([dc.reference(spec, list(r)) for r in X]); Xj = jnp.array(X)
        f = jax.jit(lambda Q: run_bil(Q, Xj)); acc = lambda Q: float((np.asarray(f(Q)) == ref).all(1).mean()); clean = acc(p)
        a = s50({str(s): [acc(noisy(p, s, jax.random.PRNGKey(k))) for k in range(NS)] for s in SIG}); r_ = s50({str(s): [acc(noisy(p, s, jax.random.PRNGKey(k), True)) for k in range(NS)] for s in SIG})
        key = f"{name}|N{N}"; old = prev.get(key); res[f"bilinear|{key}"] = {"clean": clean, "sigma50": a, "alpha50": r_, "params": int(sum(np.prod(v.shape) for v in p.values()))}
        print(f"BILINEAR {name:10s} N={N:2d}: clean={clean:.2f} params={res[f'bilinear|{key}']['params']:7d}  sigma50={a:.2e}  alpha50={r_:.2e}" + (f"   (clip-activation design: sigma50 {old['abs']['clip_activation']:.2e}, alpha50 {old['rel']['clip_activation']:.2e})" if old else ""), flush=True)
        json.dump(res, open("ideas_round.json", "w"))

# ---------------- Idea 2: weight-tied composition ----------------
def build_tied(N, pairs):
    n = 2 * N + 1; Pu = np.zeros((n, n)); Pu[np.arange(1, n), np.arange(n - 1)] = 1; Pu[n - 1, n - 1] = 1; Pd = np.zeros((n, n)); Pd[np.arange(n - 1), np.arange(1, n)] = 1; Pd[0, 0] = 1
    G = np.zeros((len(pairs), 3, V))
    for j, (a, b) in enumerate(pairs):
        for t in range(V): G[j, 0 if t == a else 1 if t == b else 2, t] = 1
    T = np.zeros((2, n)); T[0, :N + 1] = 1; T[1, N + 1:] = 1; s0 = np.eye(n)[N]
    return {"P": jnp.array(np.stack([Pu, Pd, np.eye(n)]), jnp.float32), "G": jnp.array(G, jnp.float32), "bias": -jnp.ones((3, n)), "T": jnp.array(T, jnp.float32), "s0": jnp.array(s0, jnp.float32)}
def run_tied(p, X, k):
    xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); s = jnp.broadcast_to(p["s0"], (X.shape[0], k, p["s0"].shape[0]))
    def f(s, x):
        Ps = jnp.einsum("rij,bkj->bkri", p["P"], s); g = jnp.einsum("krv,bv->bkr", p["G"], x)
        z = jax.nn.relu(Ps + g[..., None] + p["bias"][None, None]).sum(2); s2 = jnp.clip(2.0 * z - 0.5, 0.0, 1.0)
        return s2, jnp.argmax(s2 @ p["T"].T, -1)
    _, ys = jax.lax.scan(f, s, xs); return jnp.swapaxes(ys, 0, 1)                      # (B, L, k)
N = 16; allpairs = [(a, b) for a in range(V) for b in range(V) if a != b]
for k in [1, 2, 4, 8, 15, 30]:
    pairs = allpairs[:k]; p = build_tied(N, pairs); specs = [majority_ab(N, a, b) for a, b in pairs]
    X = rng.choice(V, size=(48, N - 1)); ref = np.stack([np.array([dc.reference(sp, list(r)) for r in X]) for sp in specs], -1); Xj = jnp.array(X)
    f = jax.jit(lambda Q: run_tied(Q, Xj, k)); out = lambda Q: np.asarray(f(Q))
    clean = float((out(p) == ref).all((1, 2)).mean())
    def curves(mode):
        mean_c, joint_c = {}, {}
        for s in SIG:
            v = []
            for i in range(NS):
                q = noisy(p, s, jax.random.PRNGKey(i)); o = out(q) == ref; v.append((float(o.all(1).mean()), float(o.all((1, 2)).mean())))
            mean_c[str(s)] = [a for a, _ in v]; joint_c[str(s)] = [b for _, b in v]
        return s50(mean_c), s50(joint_c)
    ph, jt = curves("all"); params = int(sum(np.prod(v.shape) for v in p.values()))
    res[f"tied|k{k}"] = {"clean": clean, "per_head_s50": ph, "joint_s50": jt, "params": params}
    print(f"TIED k={k:2d}: clean={clean:.2f} params={params:6d}  per-head sigma50={ph:.2e}  joint sigma50={jt:.2e}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open("ideas_round.json", "w"))
print("DONE", flush=True)
