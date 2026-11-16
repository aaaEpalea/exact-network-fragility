"""(A) Nonlinear error-correcting state codes: sparse constant-weight codewords + Willshaw associative transitions.
   (B) Shared-memory universal interpreter: several DIFFERENT programs in ONE associative memory, selected by a program code in the key."""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
import digital_compiler as dc
from digital_compiler import FSM, Spec
from zeros_empirical import majority_ab, dyck_ab
from bilinear_lib import build_bil
from scaling import s50
V = 6; SIG = [float(x) for x in np.logspace(-3, 0.3, 15)]; NS = 4

def make_codes(n, m, w, rng, max_overlap=1, tries=6000):
    codes = []
    for _ in range(tries):
        c = np.zeros(m, bool); c[rng.choice(m, w, replace=False)] = True
        if all((c & d).sum() <= max_overlap for d in codes): codes.append(c)
        if len(codes) == n: return np.array(codes, np.float32)
    return None
act = lambda z, thr: np.clip(2.0 * (z - (thr - 1)) - 0.5, 0.0, 1.0)

# ------------------------------------------------------------------ (A) single program
def spec_arrays(spec):
    c = spec.comps[0]; h = spec.readout; return np.array(c.delta), np.array(h["table"]), c.init, h["n_out"]
def build_single(spec, m, w, rng):
    delta, table, init, K = spec_arrays(spec); n = delta.shape[1]; C = make_codes(n, m, w, rng)
    if C is None: return None
    M = np.zeros((V, m, m), np.float32); T = np.zeros((K, m), np.float32)
    for v in range(V):
        for i in range(n): M[v] = np.maximum(M[v], np.outer(C[delta[v, i]], C[i]))
    for i in range(n): T[table[i]] = np.maximum(T[table[i]], C[i])
    for v in range(V):                                                   # exhaustive one-step verification (=> exact for any length, noise-free)
        for i in range(n):
            s2 = act(M[v] @ C[i], w)
            if not np.array_equal(s2, C[delta[v, i]]): return None
            y = T @ C[delta[v, i]]; tr = table[delta[v, i]]
            if y[tr] - np.delete(y, tr).max() < 1: return None
    return {"M": jnp.array(M), "T": jnp.array(T), "s0": jnp.array(C[init])}
def run_w(p, X, w):
    xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); s = jnp.broadcast_to(p["s0"], (X.shape[0], p["s0"].shape[0]))
    def f(s, x):
        z = jnp.einsum("bv,vij,bj->bi", x, p["M"], s); s2 = jnp.clip(2.0 * (z - (w - 1)) - 0.5, 0.0, 1.0); return s2, jnp.argmax(s2 @ p["T"].T, -1)
    _, ys = jax.lax.scan(f, s, xs); return jnp.swapaxes(ys, 0, 1)
def noisy_keys(P, s, key, keys, rel=False):
    out = dict(P); ks = jax.random.split(key, len(keys))
    for k, kk in zip(keys, ks): out[k] = P[k] + s * (jnp.sqrt(jnp.mean(P[k] ** 2)) if rel else 1.0) * jax.random.normal(kk, P[k].shape)
    return out
def tolerance(f, ref, p, keys, rng_seed=0):
    acc = lambda Q: float((np.asarray(f(Q)) == ref).all(1).mean())
    a = s50({str(s): [acc(noisy_keys(p, s, jax.random.PRNGKey(k), keys)) for k in range(NS)] for s in SIG})
    r = s50({str(s): [acc(noisy_keys(p, s, jax.random.PRNGKey(k), keys, True)) for k in range(NS)] for s in SIG}); return acc(p), a, r

if __name__ == "__main__":
    res = {}; t0 = time.time(); rng = np.random.default_rng(0)
    for fam, mk in [("majority", majority_ab), ("dyck", dyck_ab)]:
        for N in [16, 32, 64]:
            spec = mk(N, 0, 1); n = len(spec.comps[0].delta[0]); X = rng.choice(V, size=(24, N - 1)); ref = np.array([dc.reference(spec, list(r)) for r in X]); Xj = jnp.array(X)
            pb = build_bil(spec); fb = jax.jit(lambda Q: run_bil_(Q, Xj)) if False else None
            from bilinear_lib import run_bil
            fb = jax.jit(lambda Q: run_bil(Q, Xj)); cb, ab, rb = tolerance(fb, ref, pb, ["W", "T", "s0"]); pn_b = int(sum(np.prod(v.shape) for v in pb.values()))
            found = None
            for w in [3, 4]:
                for m in [16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128]:
                    for seed in range(5):
                        p = build_single(spec, m, w, np.random.default_rng(100 * seed + m))
                        if p is not None: found = (w, m, p); break
                    if found: break
                if found: break
            if not found: print(f"WILLSHAW {fam} N={N}: no exact code found up to m=128", flush=True); continue
            w, m, p = found; f = jax.jit(lambda Q: run_w(Q, Xj, w)); c, a, r = tolerance(f, ref, p, ["M", "T", "s0"]); pn = int(sum(np.prod(v.shape) for v in p.values()))
            res[f"single|{fam}|N{N}"] = {"states_n": n, "code_len_m": m, "code_weight_w": w, "params_willshaw": pn, "params_onehot_bilinear": pn_b, "param_ratio": pn_b / pn, "clean": c,
                                        "sigma50_willshaw": a, "alpha50_willshaw": r, "sigma50_onehot": ab, "alpha50_onehot": rb}
            print(f"(A) {fam:8s} N={N:2d} n={n:3d}: code m={m:3d} w={w}  params willshaw {pn:6d} vs one-hot bilinear {pn_b:6d} ({pn_b/pn:.1f}x)  clean={c:.2f} | sigma50 willshaw {a:.2e} vs one-hot {ab:.2e} | alpha50 {r:.2e} vs {rb:.2e}  ({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open("willshaw_fsm.json", "w"))

    # ---------------- (B) shared-memory universal interpreter ----------------
    def mod_spec(mm, a):
        delta = np.array([[(i + 1) % mm if t == a else i for i in range(mm)] for t in range(V)]); return Spec(V, [FSM(mm, 0, delta)], {"type": "table", "comp": 0, "table": list(range(mm)), "n_out": mm}, f"mod{mm}(tok{a})")
    def rand_spec(n, K, seed):
        r = np.random.default_rng(seed); delta = r.integers(0, n, size=(V, n)); table = r.integers(0, K, size=n); table[:K] = np.arange(K)
        return Spec(V, [FSM(n, 0, delta)], {"type": "table", "comp": 0, "table": list(table), "n_out": K}, f"random_fsm{seed}")
    pool = [majority_ab(23, 0, 1), dyck_ab(46, 0, 1), mod_spec(7, 2), rand_spec(48, 4, 1), majority_ab(23, 2, 3), dyck_ab(46, 2, 3), mod_spec(5, 4), rand_spec(48, 4, 2)]
    Kmax = 7
    def build_shared(specs, ms, mp, ws, wp, seed):
        r = np.random.default_rng(seed); P = len(specs); nmax = max(sp.comps[0].n for sp in specs); Cs = make_codes(nmax, ms, ws, r); Cp = make_codes(P, mp, wp, r)
        if Cs is None or Cp is None: return None
        thr = ws + wp; M = np.zeros((V, ms, ms + mp), np.float32); R = np.zeros((Kmax, ms + mp), np.float32); info = []
        for p, sp in enumerate(specs):
            delta, table, init, K = spec_arrays(sp); n = delta.shape[1]; g = Cp[p]; info.append((delta, table, init, K, n))
            for i in range(n):
                key = np.concatenate([Cs[i], g])
                for v in range(V): M[v] = np.maximum(M[v], np.outer(Cs[delta[v, i]], key))
                R[table[i]] = np.maximum(R[table[i]], key)
        for p, (delta, table, init, K, n) in enumerate(info):                       # exhaustive one-step verification of EVERY program
            g = Cp[p]
            for i in range(n):
                for v in range(V):
                    s2 = act(M[v] @ np.concatenate([Cs[i], g]), thr)
                    if not np.array_equal(s2, Cs[delta[v, i]]): return None
                    y = R @ np.concatenate([s2, g]); tr = table[delta[v, i]]
                    if y[tr] - np.delete(y, tr).max() < 1: return None
        return {"M": jnp.array(M), "R": jnp.array(R)}, Cs, Cp, info, thr
    def run_shared(p, X, s0, g, thr):
        xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); s = jnp.broadcast_to(s0, (X.shape[0], s0.shape[0])); gg = jnp.broadcast_to(g, (X.shape[0], g.shape[0]))
        def f(s, x):
            key = jnp.concatenate([s, gg], -1); z = jnp.einsum("bv,vij,bj->bi", x, p["M"], key); s2 = jnp.clip(2.0 * (z - (thr - 1)) - 0.5, 0.0, 1.0)
            return s2, jnp.argmax(jnp.concatenate([s2, gg], -1) @ p["R"].T, -1)
        _, ys = jax.lax.scan(f, s, xs); return jnp.swapaxes(ys, 0, 1)
    L = 40; Xs = rng.choice(V, size=(24, L)); Xj = jnp.array(Xs)
    for P in [2, 4, 8]:
        specs = pool[:P]; mp = {2: 8, 4: 12, 8: 16}[P]; found = None
        for ms in [48, 64, 80, 96, 128, 160, 192, 256]:
            for seed in range(4):
                out = build_shared(specs, ms, mp, 3, 3, seed)
                if out is not None: found = (ms, out); break
            if found: break
        sep_params = int(sum(np.prod(v.shape) for sp in specs for v in build_bil(sp).values()))
        if not found: print(f"(B) P={P}: no exact shared memory found up to ms=256 (separate one-hot tables: {sep_params} params)", flush=True); res[f"shared|P{P}"] = {"found": False, "separate_params": sep_params}; continue
        ms, (p, Cs, Cp, info, thr) = found; params = int(sum(np.prod(v.shape) for v in p.values()))
        refs = [np.array([dc.reference(sp, list(r)) for r in Xs]) for sp in specs]
        fns = [jax.jit(lambda Q, s0=jnp.array(Cs[info[q][2]]), g=jnp.array(Cp[q]): run_shared(Q, Xj, s0, g, thr)) for q in range(P)]
        def shared_acc(Q): return float(np.mean([(np.asarray(fns[q](Q)) == refs[q]).all(1).mean() for q in range(P)]))
        sepP = [build_bil(sp) for sp in specs]; sepf = [jax.jit(lambda Q, sp=sp: __import__("bilinear_lib").run_bil(Q, Xj)) for sp in specs]
        def sep_acc(Qs): return float(np.mean([(np.asarray(sepf[q](Qs[q])) == refs[q]).all(1).mean() for q in range(P)]))
        sh = s50({str(s): [shared_acc(noisy_keys(p, s, jax.random.PRNGKey(k), ["M", "R"])) for k in range(NS)] for s in SIG})
        sp_ = s50({str(s): [sep_acc([noisy_keys(sepP[q], s, jax.random.PRNGKey(100 * q + k), ["W", "T"]) for q in range(P)]) for k in range(NS)] for s in SIG})
        res[f"shared|P{P}"] = {"found": True, "ms": ms, "mp": mp, "params_shared": params, "params_separate_onehot": sep_params, "param_ratio_separate_over_shared": sep_params / params, "clean": shared_acc(p),
                               "sigma50_shared": sh, "sigma50_separate": sp_, "programs": [sp.name for sp in specs]}
        print(f"(B) P={P}: shared memory ms={ms} mp={mp}: params {params} vs separate one-hot tables {sep_params} ({sep_params/params:.2f}x) | clean={res[f'shared|P{P}']['clean']:.2f} | sigma50 shared {sh:.2e} vs separate {sp_:.2e}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open("willshaw_fsm.json", "w"))
    print("DONE", flush=True)
