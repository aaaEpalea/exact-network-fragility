"""Compose two TRAINED GRUs (majority, dyck-valid-prefix): weight averaging vs block-diagonal stacking, plus noise tolerance."""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp, optax
warnings.filterwarnings("ignore")
import digital_compiler as dc
from digital_compiler import FSM, Spec, sat
def majority_ab(N, a, b):
    n = 2 * N + 1; delta = np.array([[sat(n, i, 1) if t == a else sat(n, i, -1) if t == b else i for i in range(n)] for t in range(6)])
    return Spec(6, [FSM(n, N, delta)], {"type": "table", "comp": 0, "table": [int(i > N) for i in range(n)], "n_out": 2}, f"maj({a}>{b})")
def dyck_ab(N, a, b):
    n = N + 2; dead = N + 1; rows = []
    for t in range(6):
        rows.append([dead if i == dead else (min(i + 1, N) if t == a else (i - 1 if i > 0 else dead) if t == b else i) for i in range(n)])
    return Spec(6, [FSM(n, 0, np.array(rows))], {"type": "table", "comp": 0, "table": [1] * (N + 1) + [0], "n_out": 2}, f"dyck({a},{b})")
from scaling import s50
V, N, HID = 6, 16, 64
specs = {"majority": majority_ab(N, 0, 1), "dyck": dyck_ab(N, 0, 1)}
def labels(spec, X): return np.array([dc.reference(spec, list(r)) for r in X])
def init(key, C=2, H=HID):
    ks = iter(jax.random.split(key, 20)); n = lambda s, sc: sc * jax.random.normal(next(ks), s); p = {}
    for g in "zrh": p["W" + g] = n((V, H), V ** -0.5); p["U" + g] = n((H, H), H ** -0.5); p["b" + g] = jnp.zeros(H)
    p["Wo"] = n((H, C), H ** -0.5); p["bo"] = jnp.zeros(C); return p
def fwd(p, X, head=("Wo", "bo")):
    H = p["Uz"].shape[0]; xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); h0 = jnp.zeros((X.shape[0], H))
    def step(h, x):
        z = jax.nn.sigmoid(x @ p["Wz"] + h @ p["Uz"] + p["bz"]); r = jax.nn.sigmoid(x @ p["Wr"] + h @ p["Ur"] + p["br"])
        hh = jnp.tanh(x @ p["Wh"] + (r * h) @ p["Uh"] + p["bh"]); h2 = (1 - z) * h + z * hh
        return h2, [h2 @ p[a] + p[b] for a, b in [head]][0]
    _, ys = jax.lax.scan(step, h0, xs); return jnp.swapaxes(ys, 0, 1)
def train(spec, seed):
    p = init(jax.random.PRNGKey(seed)); steps = 2500
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.cosine_decay_schedule(3e-3, steps, 0.02))); st = opt.init(p)
    @jax.jit
    def step(p, st, X, y):
        def loss(p): return optax.softmax_cross_entropy_with_integer_labels(fwd(p, X), y).mean()
        l, g = jax.value_and_grad(loss)(p); u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, l
    rng = np.random.default_rng(seed)
    for i in range(steps):
        L = [8, N - 1][i % 2]; X = rng.integers(0, V, size=(128, L)); p, st, l = step(p, st, jnp.array(X), jnp.array(labels(spec, X)))
    return p
rng = np.random.default_rng(7); Xe = rng.integers(0, V, size=(200, N - 1)); Xj = jnp.array(Xe); Y = {k: labels(s, Xe) for k, s in specs.items()}
acc = lambda P, task, head=("Wo", "bo"): float((np.asarray(jnp.argmax(fwd(P, Xj, head), -1)) == Y[task]).all(1).mean())
t0 = time.time(); P = {k: train(s, i) for i, (k, s) in enumerate(specs.items())}
print({k: f"{acc(P[k], k):.3f}" for k in P}, f"({time.time()-t0:.0f}s)", flush=True)
# (i) weight averaging
avg = {n: (P["majority"][n] + P["dyck"][n]) / 2 for n in P["majority"]}
print("weight-average merge: majority acc %.3f, dyck acc %.3f" % (acc(avg, "majority"), acc(avg, "dyck")), flush=True)
# (ii) block-diagonal stacking, two heads
def blockdiag(a, b): 
    z1, z2 = jnp.zeros((a.shape[0], b.shape[1])), jnp.zeros((b.shape[0], a.shape[1])); return jnp.block([[a, z1], [z2, b]])
bd = {}
for g in "zrh": bd["W" + g] = jnp.concatenate([P["majority"]["W" + g], P["dyck"]["W" + g]], 1); bd["U" + g] = blockdiag(P["majority"]["U" + g], P["dyck"]["U" + g]); bd["b" + g] = jnp.concatenate([P["majority"]["b" + g], P["dyck"]["b" + g]])
bd["Wo_maj"] = jnp.concatenate([P["majority"]["Wo"], jnp.zeros((HID, 2))], 0); bd["bo_maj"] = P["majority"]["bo"]
bd["Wo_dyc"] = jnp.concatenate([jnp.zeros((HID, 2)), P["dyck"]["Wo"]], 0); bd["bo_dyc"] = P["dyck"]["bo"]
both = lambda P_: (acc(P_, "majority", ("Wo_maj", "bo_maj")), acc(P_, "dyck", ("Wo_dyc", "bo_dyc")))
print("block-diagonal composite: majority acc %.3f, dyck acc %.3f" % both(bd), flush=True)
# noise tolerance (relative, RMS protocol): components vs composite (both heads must be exact)
def noisy(P_, a, key):
    lv, tr = jax.tree_util.tree_flatten(P_); ks = jax.random.split(key, len(lv))
    return jax.tree_util.tree_unflatten(tr, [x + a * jnp.sqrt(jnp.mean(x ** 2)) * jax.random.normal(k, x.shape) for x, k in zip(lv, ks)])
ALPH = [float(x) for x in np.logspace(-2, 0.5, 13)]
def a50(f): return s50({str(a): [f(noisy_p(a, k)) for k in range(4)] for a in ALPH})
out = {}
for name, P_, fn in [("majority alone", P["majority"], lambda q: acc(q, "majority")), ("dyck alone", P["dyck"], lambda q: acc(q, "dyck")),
                     ("composite (both heads exact)", bd, lambda q: min(*both(q)))]:
    noisy_p = lambda a, k, P_=P_: noisy(P_, a, jax.random.PRNGKey(k)); out[name] = a50(fn); print(f"alpha50 {name:30s} = {out[name]:.2e}", flush=True)
json.dump(out, open("gru_merge.json", "w")); print("DONE", flush=True)
