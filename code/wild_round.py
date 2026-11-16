"""Wild ideas round: (1) unit-level replication of the bilinear state, (2) superposition-code capacity for state sharing, (3) noise-injection training of bilinear weights (re-verified exact)."""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp, optax
warnings.filterwarnings("ignore")
import digital_compiler as dc
from zeros_empirical import majority_ab, dyck_ab
from bilinear_lib import build_bil, run_bil, noisy
from scaling import s50
V = 6; SIG = [float(x) for x in np.logspace(-3, 0.7, 18)]; NS = 6; res = {}; t0 = time.time(); rng = np.random.default_rng(0)

# ---------- (1) unit-level replication: r copies of every state unit, all weights 1 (unscaled), activation rescaled ----------
def replicate(p, r):
    return {"W": jnp.array(np.stack([np.kron(np.asarray(w), np.ones((r, r))) for w in p["W"]]), jnp.float32),
            "T": jnp.array(np.kron(np.asarray(p["T"]), np.ones((1, r))), jnp.float32), "s0": jnp.array(np.kron(np.asarray(p["s0"]), np.ones(r)), jnp.float32)}
def run_rep(p, X, r):
    xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); s = jnp.broadcast_to(p["s0"], (X.shape[0], p["s0"].shape[0]))
    def f(s, x):
        z = jnp.einsum("bv,vij,bj->bi", x, p["W"], s); s2 = jnp.clip(2.0 * z / r - 0.5, 0.0, 1.0); return s2, jnp.argmax(s2 @ p["T"].T, -1)
    _, ys = jax.lax.scan(f, s, xs); return jnp.swapaxes(ys, 0, 1)
N = 16; spec = majority_ab(N, 0, 1); X = rng.choice(V, size=(48, N - 1)); ref = np.array([dc.reference(spec, list(r)) for r in X]); Xj = jnp.array(X)
base = build_bil(spec)
for r in [1, 2, 4, 8]:
    p = replicate(base, r); f = jax.jit(lambda Q: run_rep(Q, Xj, r)); acc = lambda Q: float((np.asarray(f(Q)) == ref).all(1).mean()); clean = acc(p)
    a = s50({str(s): [acc(noisy(p, s, jax.random.PRNGKey(k))) for k in range(NS)] for s in SIG}); params = int(sum(np.prod(v.shape) for v in p.values()))
    res[f"replication|r{r}"] = {"clean": clean, "sigma50": a, "params": params}
    print(f"REPLICATION r={r}: clean={clean:.2f} params={params:7d} sigma50={a:.2e}  (sqrt(r) prediction vs r=1: {res['replication|r1']['sigma50']*np.sqrt(r):.2e})  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open("wild_round.json", "w"))

# ---------- (2) superposition capacity: random unit codes, two active codes (one per program), decode by threshold 0.5 ----------
nA, nB = 33, 18
for d in [24, 48, 96, 192, 384, 768, 1536]:
    ok_all = []
    for seed in range(3):
        U = np.random.default_rng(seed).standard_normal((d, nA + nB)); U /= np.linalg.norm(U, axis=0)
        good = 0; tot = 0
        for a in range(nA):
            for b in range(nB):
                s = U[:, a] + U[:, nA + b]; c = U.T @ s; dec = c > 0.5; truth = np.zeros(nA + nB, bool); truth[a] = truth[nA + b] = True
                good += bool((dec == truth).all()); tot += 1
        ok_all.append(good / tot)
    res[f"superposition|d{d}"] = {"exact_decode_fraction": float(np.mean(ok_all))}
    print(f"SUPERPOSITION state dim d={d:5d} (vs {nA+nB} one-hot dims): fraction of (stateA,stateB) pairs decoded exactly = {np.mean(ok_all):.3f}", flush=True)
json.dump(res, open("wild_round.json", "w"))

# ---------- (3) noise-injection training of the bilinear weights (all one-step configs), then re-verify exactness ----------
cfgs = list(dc._configs(spec)); S = jnp.array(np.stack([c[1] for c in cfgs]), jnp.float32); Xo = jnp.array(np.stack([np.eye(V)[c[0]] for c in cfgs]), jnp.float32)
ES = jnp.array(np.stack([c[2] for c in cfgs]), jnp.float32); EY = jnp.array([c[3][0] for c in cfgs], jnp.int32)
P = {"W": base["W"], "T": base["T"]}; opt = optax.adam(optax.cosine_decay_schedule(3e-3, 3000, 0.05)); st = opt.init(P)
@jax.jit
def step(P, st, key):
    k1, k2, k3, k4 = jax.random.split(key, 4); sg = jnp.exp(jax.random.uniform(k1, (), minval=np.log(0.03), maxval=np.log(0.4)))
    def loss(P):
        W = P["W"] + sg * jax.random.normal(k2, P["W"].shape); T = P["T"] + sg * jax.random.normal(k3, P["T"].shape)
        eta = 0.08 * jax.random.uniform(k4, S.shape, minval=-1, maxval=1); z = jnp.einsum("mv,vij,mj->mi", Xo, W, S + eta); s2 = jnp.clip(2 * z - 0.5, 0, 1)
        return jnp.mean((s2 - ES) ** 2) + optax.softmax_cross_entropy_with_integer_labels(8.0 * (s2 @ T.T), EY).mean()
    l, g = jax.value_and_grad(loss)(P); u, st = opt.update(g, st, P); return optax.apply_updates(P, u), st, l
key = jax.random.PRNGKey(0)
for i in range(3000):
    key, sk = jax.random.split(key); P, st, l = step(P, st, sk)
Pt = {**P, "s0": base["s0"]}
z = jnp.einsum("mv,vij,mj->mi", Xo, Pt["W"], S); s2 = jnp.clip(2 * z - 0.5, 0, 1); cfg_exact = float((jnp.abs(s2 - ES).max(-1) < 1e-3).mean())
f = jax.jit(lambda Q: run_bil(Q, Xj)); acc = lambda Q: float((np.asarray(f(Q)) == ref).all(1).mean())
Lr = rng.choice(V, size=(64, 400)); refl = np.array([dc.reference(spec, list(r)) for r in Lr]); long_acc = float((np.asarray(jax.jit(lambda Q: run_bil(Q, jnp.array(Lr)))(Pt)) == refl).all(1).mean())
out = {"loss": float(l), "all_configs_exact": cfg_exact, "clean": acc(Pt), "length400_acc": long_acc,
       "sigma50_before": s50({str(s): [acc(noisy(base, s, jax.random.PRNGKey(k))) for k in range(NS)] for s in SIG}), "sigma50_after": s50({str(s): [acc(noisy(Pt, s, jax.random.PRNGKey(k))) for k in range(NS)] for s in SIG}),
       "alpha50_before": s50({str(s): [acc(noisy(base, s, jax.random.PRNGKey(k), True)) for k in range(NS)] for s in SIG}), "alpha50_after": s50({str(s): [acc(noisy(Pt, s, jax.random.PRNGKey(k), True)) for k in range(NS)] for s in SIG})}
res["noise_trained_bilinear"] = out
print(f"NOISE-TRAINED BILINEAR: all one-step configs exact={cfg_exact:.3f} clean={out['clean']:.2f} length-400 acc={long_acc:.2f} | sigma50 {out['sigma50_before']:.2e} -> {out['sigma50_after']:.2e} | alpha50 {out['alpha50_before']:.2e} -> {out['alpha50_after']:.2e}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open("wild_round.json", "w")); print("DONE", flush=True)
