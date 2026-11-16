"""Does a DIGITAL recurrent counter (one-hot state + per-step snapping) keep its noise tolerance as N grows?
Compare with an analog-style unary chain WITHOUT snapping. Noise on EVERY parameter (incl. zeros), reused each step."""
import json, numpy as np
rng = np.random.default_rng(0)
SIG = np.logspace(-4, 0, 17); DRAWS = 24

def s50(xs, ys):
    for i in range(1, len(xs)):
        if ys[i] < 0.5:
            f = (ys[i-1]-0.5)/(ys[i-1]-ys[i]); return float(np.exp(np.log(xs[i-1])+f*(np.log(xs[i])-np.log(xs[i-1]))))
    return float("inf")

def counter_params(N):
    n = N + 1; P = np.zeros((n, n)); P[np.arange(1, n), np.arange(n - 1)] = 1.0   # shift: count i -> i+1
    P[n - 1, n - 1] = 1.0                                                         # saturate at N
    return {"M1": 2 * P, "M2": 2 * P, "b1": -0.5 * np.ones(n), "b2": -1.5 * np.ones(n), "C1": np.eye(n), "C2": -np.eye(n)}

def run_snap(p, L):
    n = p["b1"].shape[0]; s = np.zeros(n); s[0] = 1.0
    for _ in range(L):
        h1 = np.maximum(p["M1"] @ s + p["b1"], 0); h2 = np.maximum(p["M2"] @ s + p["b2"], 0)
        s = p["C1"] @ h1 + p["C2"] @ h2            # = clip(2*P s - 0.5, 0, 1)-like restoring step
    return int(np.argmax(s))

def run_nosnap(p, L):                               # analog chain: s <- P s (no restoring)
    n = p["M1"].shape[0]; s = np.zeros(n); s[0] = 1.0
    for _ in range(L): s = (p["M1"] / 2) @ s
    return int(np.argmax(s))

def sigma50(N, mode):
    base = counter_params(N); lens = list(range(1, N))
    accs = []
    for sg in SIG:
        ok = 0
        for _ in range(DRAWS):
            p = {k: v + sg * rng.standard_normal(v.shape) for k, v in base.items()}
            f = run_snap if mode == "snap" else run_nosnap
            ok += np.mean([f(p, L) == L for L in lens])
        accs.append(ok / DRAWS)
    return s50(SIG, accs), accs

res = {}
for N in [8, 16, 32, 64, 128]:
    a, _ = sigma50(N, "snap"); b, _ = sigma50(N, "nosnap")
    res[N] = {"digital_snap": a, "analog_nosnap": b}
    print(f"N={N:4d}  digital+snap sigma50={a:.2e}   no-snap chain sigma50={b:.2e}", flush=True)
json.dump(res, open("digital_counter.json", "w"))
Ns = list(res); 
for k in ["digital_snap", "analog_nosnap"]:
    print(k, "slope vs N:", round(np.polyfit(np.log(Ns), np.log([res[n][k] for n in Ns]), 1)[0], 2))
