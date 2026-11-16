"""Second program: majority-so-far.  y_t = 1 if (#tokens==1 in x_1..x_t) > (#tokens==2 in x_1..x_t) else 0.
Digital: a single SIGNED one-hot counter (diff = #1 - #2), token-gated shifts, snapping, threshold readout.
Tracr: causal transformer, SequenceMap(a>b) over two SelectorWidth counters."""
import json, sys, time, itertools, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
from tracr.rasp import rasp
from tracr.compiler import compiling
from scaling import s50
from digital_prefix_hist import noisy_tree, SIG, SEEDS, BATCH, V
PROBS = [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]

def maj_expected(x):  # x: tokens 1..6
    o = t2 = 0; out = []
    for v in x:
        o += (v == 1); t2 += (v == 2); out.append(int(o > t2))
    return out

def build_digital(N):
    n = 2 * N + 1; I = np.eye(n)
    Pu = np.zeros((n, n)); Pu[np.arange(1, n), np.arange(n - 1)] = 1; Pu[n - 1, n - 1] = 1
    Pd = np.zeros((n, n)); Pd[np.arange(n - 1), np.arange(1, n)] = 1; Pd[0, 0] = 1
    G1 = np.zeros((n, V)); G1[:, 0] = 1; G2 = np.zeros((n, V)); G2[:, 1] = 1
    pos = (np.arange(n) > N).astype(float)
    p = {"Pu": Pu, "G1a": G1, "b1": -np.ones(n), "Pd": Pd, "G2a": G2, "b2": -np.ones(n),
         "I3": I, "G3": -(G1 + G2), "b3": np.zeros(n), "C1": I, "C2": I, "C3": I,
         "S1": 2 * I, "bs1": -0.5 * np.ones(n), "S2": 2 * I, "bs2": -1.5 * np.ones(n), "C4": I, "C5": -I,
         "A": np.stack([-pos, pos]), "bA": np.array([1.0, 0.0]), "s0": np.eye(n)[N]}
    return {k: jnp.asarray(v, jnp.float32) for k, v in p.items()}

def run_digital(p, X, snap=True):
    B, L = X.shape; xs = jax.nn.one_hot(X, V); s = jnp.broadcast_to(p["s0"], (B, p["s0"].shape[0]))
    def step(s, x):
        u1 = jax.nn.relu(s @ p["Pu"].T + x @ p["G1a"].T + p["b1"]); u2 = jax.nn.relu(s @ p["Pd"].T + x @ p["G2a"].T + p["b2"])
        u3 = jax.nn.relu(s @ p["I3"].T + x @ p["G3"].T + p["b3"]); z = u1 @ p["C1"].T + u2 @ p["C2"].T + u3 @ p["C3"].T
        if snap:
            a1 = jax.nn.relu(z @ p["S1"].T + p["bs1"]); a2 = jax.nn.relu(z @ p["S2"].T + p["bs2"]); s2 = a1 @ p["C4"].T + a2 @ p["C5"].T
        else: s2 = z
        return s2, jnp.argmax(s2 @ p["A"].T + p["bA"], -1)
    _, ys = jax.lax.scan(step, s, jnp.swapaxes(xs, 0, 1)); return jnp.swapaxes(ys, 0, 1)

def eval_digital(N, snap, rng):
    p = build_digital(N); L = N - 1
    X = np.stack([rng.choice(V, size=L, p=PROBS) for _ in range(BATCH)]); exp = np.array([maj_expected([int(t) + 1 for t in r]) for r in X])
    f = jax.jit(lambda P, X: run_digital(P, X, snap)); clean = float((np.asarray(f(p, jnp.array(X))) == exp).all(1).mean())
    curve = {str(s): [float((np.asarray(f(noisy_tree(p, s, jax.random.PRNGKey(k)), jnp.array(X))) == exp).all(1).mean()) for k in range(SEEDS)] for s in SIG}
    return clean, s50(curve), int(sum(np.prod(v.shape) for v in p.values()))

def eval_tracr(N, c=1.0):
    t = rasp.tokens
    ones = rasp.SelectorWidth(rasp.Select(t, t, lambda k, q: k == 1)); twos = rasp.SelectorWidth(rasp.Select(t, t, lambda k, q: k == 2))
    prog = rasp.SequenceMap(lambda a, b: int(a > b), ones, twos)
    S = max(100, int(1.5 * N * N))
    m = compiling.compile_rasp_to_model(prog, vocab=set(range(1, V + 1)), max_seq_len=N, compiler_bos="BOS", mlp_exactness=S, causal=True)
    rng = np.random.default_rng(0); L = N - 1
    xs = [list(map(int, rng.choice(range(1, V + 1), size=L, p=PROBS))) for _ in range(BATCH)]
    toks = jnp.array([m.input_encoder.encode(["BOS"] + x) for x in xs]); exp = [maj_expected(x) for x in xs]
    fwd = jax.jit(lambda p, t: m.forward(p, t).unembedded_output)
    def acc(p):
        out = np.asarray(fwd(p, toks))[:, 1:]; ok = 0
        for row, e in zip(out, exp): ok += list(m.output_encoder.decode(row.tolist())) == list(e)
        return ok / len(xs)
    P0 = {mod: {n: (jnp.asarray(a) * c if mod.endswith("attn/query") else jnp.asarray(a)) for n, a in d.items()} for mod, d in m.params.items()}
    clean = acc(P0); npar = sum(int(np.prod(np.shape(a))) for d in m.params.values() for a in d.values())
    if clean < 0.999: return clean, float("nan"), npar
    def noisy(p, s, key):
        lv, tr = jax.tree_util.tree_flatten(p); ks = jax.random.split(key, len(lv))
        return jax.tree_util.tree_unflatten(tr, [a + s * jax.random.normal(k, a.shape) for a, k in zip(lv, ks)])
    return clean, s50({str(s): [acc(noisy(P0, s, jax.random.PRNGKey(k))) for k in range(SEEDS)] for s in SIG}), npar

if __name__ == "__main__":
    res = {}; rng = np.random.default_rng(0); t0 = time.time()
    pN = build_digital(6); allx = np.array(list(itertools.product(range(V), repeat=5)))
    exp = np.array([maj_expected([int(t) + 1 for t in r]) for r in allx]); got = np.asarray(jax.jit(lambda P, X: run_digital(P, X))(pN, jnp.array(allx)))
    print(f"digital majority exhaustive check N=6: {(got == exp).all(1).sum()}/{len(allx)} exact", flush=True)
    for N in [8, 16, 32, 64]:
        for snap in [True, False]:
            clean, v, npar = eval_digital(N, snap, rng); res[f"digital_{'snap' if snap else 'nosnap'}|{N}"] = {"clean": clean, "s50": v, "params": npar}
            print(f"digital {'snap  ' if snap else 'nosnap'} N={N:3d} clean={clean:.2f} sigma50={v:.2e} params={npar} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open("second_program.json", "w"))
    for N in [8, 16, 32]:
        for c in [1.0, 0.25]:
            try: clean, v, npar = eval_tracr(N, c)
            except Exception as e: print(f"tracr N={N} c={c} FAILED {repr(e)[:120]}", flush=True); continue
            res[f"tracr_c{c}|{N}"] = {"clean": clean, "s50": v, "params": npar}
            print(f"tracr causal c={c} N={N:3d} clean={clean:.2f} sigma50={v:.2e} params={npar} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open("second_program.json", "w"))
    print("DONE", flush=True)
