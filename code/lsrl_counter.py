"""LSRL linear-state counter: parameter-noise tolerance vs N (analog accumulate-by-addition design)."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))   # repository root
import sys, json, warnings
sys.path.insert(0, _os.environ.get("LSRL_PATH", REPO + "/external/LSRL"))
warnings.filterwarnings("ignore")
import numpy as np, lsrl

inp = lsrl.Input(dim=1)
ctr = lsrl.LinState(inp, A=lsrl.Matrix.ones(1, 1), B=lsrl.Matrix.ones(1, 1), init_state=lsrl.Matrix.zeros(1, 1), name="Count")
loop = lsrl.ForEach(ctr); loop.simplify()
layers = loop.topological_sort()
params = []
for L in layers:
    if isinstance(L, lsrl.LinState):
        params.append(("lin", {"A": np.array(L.A.numpy(), float), "B": np.array(L.B.numpy(), float), "bias": np.array(L.bias.numpy(), float), "init": np.array(L.init_state.numpy(), float)}))
    elif isinstance(L, lsrl.Linear):
        params.append(("lin_out", {"A": np.array(L.A.numpy(), float), "b": np.array(L.b.numpy(), float)}))
    elif isinstance(L, lsrl.ReLU): params.append(("relu", {}))
print("compiled layers:", [(k, {n: v.shape for n, v in d.items()}) for k, d in params])

def run(P, L):
    s = None; out = None
    for kind, d in P:
        if kind == "lin": s = d["init"].reshape(-1)
    x_seq = np.ones((L, 1)); y = None
    for t in range(L):
        h = x_seq[t]
        for kind, d in P:
            if kind == "lin": s = d["A"] @ s + d["B"] @ h + d["bias"].reshape(-1); h = s
            elif kind == "lin_out": h = d["A"] @ h + d["b"].reshape(-1)
            elif kind == "relu": h = np.maximum(h, 0)
        y = h
    return float(y[0])

# sanity: clean model counts exactly
assert all(abs(run(params, L) - L) < 1e-9 for L in range(1, 20)), "clean counter wrong"
rng = np.random.default_rng(0); SIG = np.logspace(-8, -1, 22); DRAWS = 30
def s50(xs, ys):
    for i in range(1, len(xs)):
        if ys[i] < 0.5:
            f = (ys[i-1]-0.5)/(ys[i-1]-ys[i]); return float(np.exp(np.log(xs[i-1])+f*(np.log(xs[i])-np.log(xs[i-1]))))
    return float("inf")
res = {}
for N in [8, 16, 32, 64, 128, 256]:
    accs = []
    for sg in SIG:
        ok = 0
        for _ in range(DRAWS):
            P = [(k, {n: v + sg * rng.standard_normal(v.shape) for n, v in d.items()}) for k, d in params]
            ok += np.mean([abs(run(P, L) - L) < 0.5 for L in range(1, N)])
        accs.append(ok / DRAWS)
    res[N] = s50(SIG, accs); print(f"N={N:4d}  LSRL linear-state counter sigma50 = {res[N]:.2e}", flush=True)
Ns = list(res); print("slope vs N:", round(np.polyfit(np.log(Ns), np.log([res[n] for n in Ns]), 1)[0], 2))
json.dump(res, open("lsrl_counter.json", "w"))
