"""A: weight-tied composition (k template programs, shared role matrices) -- certificates + function-level attacks.
Designs: 'bil'  = tied-bilinear  z_i = sum_r g_r (P_r s)_i                         (no hidden gates, no biases)
         'relu' = ReLU-gated     z_i = sum_r relu((P_r s)_i + g_r + bias_ri)       (the ORIGINAL tied design, bias = -1)
k in {1,2,4,8,15,30}; N in {16,32}.  Per-head vs joint:
   ANY  : attack objective = softmin over ALL heads (smallest eps at which at least one head can be made wrong = joint exactness lost)
   HEAD : attack objective = one designated head only (head k-1); smallest eps at which THAT head can be broken
   ALL  : at a ladder of multiples of the certified radius, number of heads that one shared perturbation breaks simultaneously (attack maximises the summed per-head violation)
Bisection: doubling ladder then 5 bisection steps between the largest multiple with no break found and the smallest with a break (certificate -> first failure interval)."""
import json, time, sys, os, numpy as np, jax, jax.numpy as jnp
import warnings; warnings.filterwarnings("ignore")
from fw_lib import V, majority_ab, PAIRS, ref_run
from cert_tied import cert_radius_tied
OUT = "../results/A_tied.json"; t0 = time.time()
def build(design, N, pairs):
    n = 2 * N + 1; Pu = np.zeros((n, n)); Pu[np.arange(1, n), np.arange(n - 1)] = 1; Pu[n - 1, n - 1] = 1; Pd = np.zeros((n, n)); Pd[np.arange(n - 1), np.arange(1, n)] = 1; Pd[0, 0] = 1
    G = np.zeros((len(pairs), 3, V))
    for j, (a, b) in enumerate(pairs):
        for t in range(V): G[j, 0 if t == a else 1 if t == b else 2, t] = 1
    T = np.zeros((2, n)); T[0, :N + 1] = 1; T[1, N + 1:] = 1
    p = {"P": np.stack([Pu, Pd, np.eye(n)]), "G": G, "T": T, "s0": np.eye(n)[N]}
    if design == "relu": p["bias"] = -np.ones((3, n))
    return {k: jnp.asarray(v, jnp.float32) for k, v in p.items()}
def run_logits(design, p, X, k):
    xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); s = jnp.broadcast_to(p["s0"], (X.shape[0], k, p["s0"].shape[0]))
    def f(s, x):
        g = jnp.einsum("krv,bv->bkr", p["G"], x)
        if design == "bil": z = jnp.einsum("bkr,rij,bkj->bki", g, p["P"], s)
        else: z = jax.nn.relu(jnp.einsum("rij,bkj->bkri", p["P"], s) + g[..., None] + p["bias"][None, None]).sum(2)
        s2 = jnp.clip(2.0 * z - 0.5, 0.0, 1.0); return s2, s2 @ p["T"].T
    _, ys = jax.lax.scan(f, s, xs); return jnp.swapaxes(ys, 0, 1)             # (B,L,k,2)
def attack_data(N, k, B=40, seed=0):
    pairs = PAIRS[:k]; rng = np.random.default_rng(seed); L = 3 * N; X = rng.choice(V, size=(B, L))
    for j, (a, b) in enumerate(pairs[:min(k, 6)]): X[2 * j % B] = a; X[(2 * j + 1) % B] = b                 # constant up / down sequences
    for q in range(10): v = rng.integers(V); m = rng.random(L) < 0.85; X[B - 1 - q] = np.where(m, v, rng.integers(V, size=L))
    ref = np.stack([np.array([[int(x) for x in ref_run(majority_ab(N, a, b), X[i:i + 1])[0]] for i in range(B)]) for a, b in pairs], -1)    # (B,L,k)
    return pairs, X, ref
def make_attack(design, N, k, protect, head=None, mode="any"):
    big = (N == 32 and k >= 15); pairs, X, ref = attack_data(N, k, B=(20 if big else 40)); p = build(design, N, pairs); Xj = jnp.array(X); Rj = jnp.array(ref)
    masks = {kk: ((p[kk] != 0).astype(jnp.float32) if protect else jnp.ones_like(p[kk])) for kk in p}
    if design == "relu": masks["bias"] = jnp.ones_like(p["bias"])      # biases are non-zero parameters: always perturbed
    def margins(d):
        q = {kk: p[kk] + d[kk] for kk in p}; lg = run_logits(design, q, Xj, k); corr = jnp.take_along_axis(lg, Rj[..., None], -1)[..., 0]
        oth = jnp.where(jax.nn.one_hot(Rj, 2) > 0, -1e9, lg).max(-1); return corr - oth                          # (B,L,k)
    def obj(d):
        m = margins(d)
        if mode == "any": return -0.05 * jax.nn.logsumexp(-m.reshape(-1) / 0.05)
        if mode == "head": return -0.05 * jax.nn.logsumexp(-m[..., head].reshape(-1) / 0.05)
        per = -0.05 * jax.nn.logsumexp(-m.reshape(m.shape[0] * m.shape[1], k) / 0.05, axis=0)                      # softmin per head; "all": maximise damage to every head
        return per.mean() - 0.5 * jnp.mean(jnp.maximum(per, 0))
    g = jax.jit(jax.value_and_grad(obj)); mj = jax.jit(margins); rng = np.random.default_rng(0)
    def attack(eps, steps=(30 if big else 60), restarts=(1 if big else 2)):      # reduced-strength attack for N=32,k>=15 (compute); its break multiple is an upper bound
        best = 1e9; heads_broken = 0
        for r in range(restarts):
            d = {kk: eps * masks[kk] * jnp.asarray(rng.choice([-1., 1.], size=p[kk].shape), jnp.float32) for kk in p}
            for t in range(steps):
                _, gr = g(d); gr = jax.tree_util.tree_map(jnp.nan_to_num, gr); d = {kk: jnp.clip(d[kk] - eps / 6 * jnp.sign(gr[kk]), -eps, eps) * masks[kk] for kk in p}
            m = np.asarray(mj(d)); best = min(best, float(np.nanmin(m) if mode != "head" else np.nanmin(m[..., head]))); heads_broken = max(heads_broken, int((m.reshape(-1, k).min(0) <= 0).sum()))
        return best, heads_broken
    return attack
def first_break(att, e0, grow=1.5, tmax=40, bis=5):
    m0, _ = att(e0)
    if m0 <= 0: return 1.0, m0, 1.0
    lo, hi, t = 1.0, None, 1.0
    while t < tmax:
        t *= grow; m, _ = att(t * e0)
        if m <= 0: hi = t; break
        lo = t
    if hi is None: return None, m0, lo
    for _ in range(bis):
        mid = float(np.sqrt(lo * hi)); m, _ = att(mid * e0)
        if m <= 0: hi = mid
        else: lo = mid
    return hi, m0, lo
def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    plan = [("bil", 16, False), ("relu", 16, False), ("bil", 16, True), ("relu", 16, True), ("bil", 32, False), ("relu", 32, False)]
    cert = {}
    for design, N, protect in plan:
        key = f"cert|{design}|N{N}|{'protected' if protect else 'dense'}"
        if key not in res:
            e, d = cert_radius_tied(design, N, protect); res[key] = {"eps_cert": e, "tube": d}; json.dump(res, open(OUT, "w"), indent=1)
            print(f"CERT {design} N={N} {'zeros-protected' if protect else 'dense':15s}: certified eps* = {e:.4f} (tube {d})  ({time.time()-t0:.0f}s)", flush=True)
        cert[(design, N, protect)] = res[key]["eps_cert"]
    for design, N, protect in plan:
        e0 = cert[(design, N, protect)]
        for k in [1, 2, 4, 8, 15, 30]:
            if N == 32 and k not in (1, 4, 15, 30): continue
            key = f"attack|{design}|N{N}|{'protected' if protect else 'dense'}|k{k}"
            if key in res: continue
            out = {"eps_cert": e0}
            for mode, head in [("any", None), ("head", k - 1)]:
                att = make_attack(design, N, k, protect, head, mode); tb, m0, lo = first_break(att, e0)
                out[mode] = {"first_break_multiple": tb, "largest_unbroken_multiple": lo, "margin_at_cert": m0}
            att = make_attack(design, N, k, protect, None, "all")
            out["all_ladder"] = {str(m): att(m * e0)[1] for m in (1.0, 1.5, 2.0, 3.0, 5.0)}
            res[key] = out; json.dump(res, open(OUT, "w"), indent=1)
            f = lambda d: "none<40x" if d["first_break_multiple"] is None else f"{d['first_break_multiple']:.2f}x (unbroken up to {d['largest_unbroken_multiple']:.2f}x)"
            print(f"ATTACK {design:4s} N={N:2d} {'prot ' if protect else 'dense'} k={k:2d} cert={e0:.4f}: ANY first break {f(out['any'])} | HEAD(k-1) {f(out['head'])} | heads broken at 1/1.5/2/3/5x: " + "/".join(str(v) for v in out["all_ladder"].values()) + f"  ({time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)
if __name__ == "__main__": main()
