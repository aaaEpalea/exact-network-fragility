"""How much fragility comes from the zero entries? Empirical sigma50 with noise on ALL / only ZERO / only NONZERO entries."""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
import digital_compiler as dc
from digital_compiler import FSM, Spec, sat
from scaling import s50
from tracr.compiler import compiling, lib
V = 6; SIG = [float(x) for x in np.logspace(-4, 0, 17)]; NS = 4
def majority_ab(N, a, b):
    n = 2 * N + 1; d = np.array([[sat(n, i, 1) if t == a else sat(n, i, -1) if t == b else i for i in range(n)] for t in range(V)])
    return Spec(V, [FSM(n, N, d)], {"type": "table", "comp": 0, "table": [int(i > N) for i in range(n)], "n_out": 2}, f"maj({a}>{b})")
def dyck_ab(N, a, b):
    n = N + 2; dead = N + 1
    rows = [[dead if i == dead else (min(i + 1, N) if t == a else (i - 1 if i > 0 else dead) if t == b else i) for i in range(n)] for t in range(V)]
    return Spec(V, [FSM(n, 0, np.array(rows))], {"type": "table", "comp": 0, "table": [1] * (N + 1) + [0], "n_out": 2}, f"dyck({a},{b})")
def noise(P, s, key, mode, masks):
    ks = jax.random.split(key, len(P)); out = {}
    for (k, v), kk in zip(P.items(), ks):
        n = s * jax.random.normal(kk, v.shape); m = masks[k]
        out[k] = v + n * (1.0 if mode == "all" else (1 - m) if mode == "zero" else m)
    return out
def digital_s50(spec, N, rng):
    p, meta = dc.compile_spec(spec); masks = {k: (v != 0).astype(jnp.float32) for k, v in p.items()}
    X = rng.choice(V, size=(24, N - 1)); ref = np.array([dc.reference(spec, list(r)) for r in X]).reshape(24, N - 1, -1); Xj = jnp.array(X)
    f = jax.jit(lambda P: dc.run(P, meta, Xj, V)); acc = lambda P: float((np.asarray(f(P)).reshape(24, N - 1, -1) == ref).all((1, 2)).mean())
    return {m: s50({str(s): [acc(noise(p, s, jax.random.PRNGKey(k), m, masks)) for k in range(NS)] for s in SIG}) for m in ["all", "zero", "nonzero"]}
if __name__ == '__main__':
    res = {}; t0 = time.time(); rng = np.random.default_rng(0)
    for fam, mk in [("majority", lambda N: majority_ab(N, 0, 1)), ("dyck", lambda N: dyck_ab(N, 0, 1))]:
        for N in [16, 32, 64]:
            r = digital_s50(mk(N), N, rng); res[f"digital|{fam}|{N}"] = r
            print(f"digital {fam:8s} N={N:3d}: sigma50 all={r['all']:.2e}  zero-only={r['zero']:.2e}  nonzero-only={r['nonzero']:.2e}  ({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open("zeros_empirical.json", "w"))
    pool = [majority_ab(16, 0, 1), dyck_ab(16, 0, 1), majority_ab(16, 2, 3), dyck_ab(16, 2, 3), majority_ab(16, 4, 5), dyck_ab(16, 4, 5), majority_ab(16, 0, 2), dyck_ab(16, 1, 3)]
    for k in [1, 2, 4, 8]:
        spec = pool[0] if k == 1 else dc.compose_parallel(pool[:k]); r = digital_s50(spec, 16, rng); res[f"composition|k{k}"] = r
        print(f"composition k={k}: sigma50 all={r['all']:.2e}  zero-only={r['zero']:.2e}  nonzero-only={r['nonzero']:.2e}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open("zeros_empirical.json", "w"))
    for N in [8, 16, 32]:
        S = max(100, int(1.5 * N * N)); m = compiling.compile_rasp_to_model(lib.make_hist(), vocab=set(range(1, V + 1)), max_seq_len=N, compiler_bos="BOS", mlp_exactness=S)
        r2 = np.random.default_rng(0); B = {}
        for L in range(2, N):
            xs = [list(map(int, r2.integers(1, V + 1, size=L))) for _ in range(20)]; B[L] = (jnp.array([m.input_encoder.encode(["BOS"] + x) for x in xs]), [[x.count(v) for v in x] for x in xs])
        cf = jax.jit(lambda p, t: m.forward(p, t).unembedded_output)
        def cacc(p):
            ok = tot = 0
            for L, (t, e) in B.items():
                out = np.asarray(cf(p, t))[:, 1:]
                for row, ee in zip(out, e): ok += bool(np.allclose(m.output_encoder.decode(row.tolist()), ee, atol=0.5)); tot += 1
            return ok / tot
        P0 = {mod: {n: jnp.asarray(a) for n, a in d.items()} for mod, d in m.params.items()}
        flat = {f"{mod}|{n}": a for mod, d in P0.items() for n, a in d.items()}; masks = {k: (v != 0).astype(jnp.float32) for k, v in flat.items()}
        unflat = lambda F: {mod: {n: F[f"{mod}|{n}"] for n in d} for mod, d in P0.items()}
        r = {mode: s50({str(s): [cacc(unflat(noise(flat, s, jax.random.PRNGKey(k), mode, masks))) for k in range(NS)] for s in SIG}) for mode in ["all", "zero", "nonzero"]}
        res[f"tracr|hist|{N}"] = r
        print(f"TRACR hist N={N:2d}: sigma50 all={r['all']:.2e}  zero-only={r['zero']:.2e}  nonzero-only={r['nonzero']:.2e}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open("zeros_empirical.json", "w"))
    print("DONE", flush=True)
