"""hist, constrained, category 3b, no feedback, contaminated run (attempt 1)"""
import numpy as np
def zeros(cfg, d_out):
    d, H, dh, dm = cfg["d_model"], cfg["n_heads"], cfg["d_head"], cfg["d_mlp"]; P = {"embed": np.zeros((7, d)), "pos_embed": np.zeros((8, d))}
    for i in range(cfg["n_layers"]):
        for n in ("wq", "wk", "wv"): P[f"l{i}.{n}"] = np.zeros((H, d, dh))
        for n in ("bq", "bk", "bv"): P[f"l{i}.{n}"] = np.zeros((H, dh))
        P[f"l{i}.wo"] = np.zeros((H, dh, d)); P[f"l{i}.bo"] = np.zeros(d); P[f"l{i}.w1"] = np.zeros((d, dm)); P[f"l{i}.b1"] = np.zeros(dm); P[f"l{i}.w2"] = np.zeros((dm, d)); P[f"l{i}.b2"] = np.zeros(d)
    P["unembed.w"] = np.zeros((d, d_out)); P["unembed.b"] = np.zeros(d_out); return P
def interp(xs, ys):
    """piecewise-linear function through the points, as c0 + sum_j w_j * relu(a - t_j)"""
    o = np.argsort(xs); x = np.array(xs, float)[o]; y = np.array(ys, float)[o]; s = (y[1:] - y[:-1]) / (x[1:] - x[:-1]); w = np.concatenate([[s[0]], s[1:] - s[:-1]]); return y[0], x[:-1], w
CONFIG = dict(n_layers=1, n_heads=1, d_model=16, d_head=8, d_mlp=32, causal=False)

def build_params():
    cfg = CONFIG; P = zeros(cfg, 1); Qs = 60.0 * np.sqrt(cfg["d_head"])
    for t in range(1, 7): P["embed"][t, t - 1] = 1.0         # dims 0..5: token one-hot
    P["embed"][0, 6] = 1.0                                   # dim 6: BOS flag
    for d_ in range(6): P["l0.wq"][0, d_, d_] = Qs           # query: token one-hot (scaled) ...
    P["l0.bq"][0, 6] = Qs                                    # ... plus constant on the BOS coordinate
    for d_ in range(7): P["l0.wk"][0, d_, d_] = 1.0          # key: token one-hot, BOS flag -> matches and BOS share the top score
    P["l0.wv"][0, 6, 0] = 1.0; P["l0.wo"][0, 0, 7] = 1.0     # value = BOS flag -> dim 7 holds 1/(count+1)
    c0, th, w = interp([1.0 / (k + 1) for k in range(1, 8)], [float(k) for k in range(1, 8)])
    for j in range(len(th)): P["l0.w1"][7, j] = 1.0; P["l0.b1"][j] = -th[j]; P["l0.w2"][j, 8] = w[j]
    P["l0.b2"][8] = c0; P["unembed.w"][8, 0] = 1.0
    return P
