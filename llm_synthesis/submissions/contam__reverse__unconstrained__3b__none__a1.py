"""reverse, unconstrained, category 3b, no feedback, contaminated run (attempt 1)"""
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
CONFIG = dict(n_layers=2, n_heads=1, d_model=18, d_head=6, d_mlp=8, causal=False)

def build_params():
    cfg = CONFIG; P = zeros(cfg, 7); Qs = 40.0 * np.sqrt(cfg["d_head"])
    for t in range(1, 7): P["embed"][t, t - 1] = 1.0         # dims 0..5 token one-hot
    P["embed"][0, 6] = 1.0                                   # dim 6 BOS flag
    for p in range(8): P["pos_embed"][p, 7] = p; P["pos_embed"][p, 8] = p * p     # dim 7: position i, dim 8: i^2
    # layer 0: uniform attention, value = BOS flag -> dim 9 = 1/(L+1); MLP turns it into M = L+1 in dim 10
    P["l0.wv"][0, 6, 0] = 1.0; P["l0.wo"][0, 0, 9] = 1.0
    c0, th, w = interp([1.0 / (k + 1) for k in range(1, 8)], [float(k + 1) for k in range(1, 8)])
    for j in range(len(th)): P["l0.w1"][9, j] = 1.0; P["l0.b1"][j] = -th[j]; P["l0.w2"][j, 10] = w[j]
    P["l0.b2"][10] = c0
    # layer 1: score(j) = 40 * (-(i + j - M)^2) + const(i)  =>  position i attends (almost) only to j = M - i = L + 1 - i
    P["l1.wq"][0, 7, 0] = -2 * Qs; P["l1.wq"][0, 10, 0] = 2 * Qs; P["l1.bq"][0, 1] = -Qs
    P["l1.wk"][0, 7, 0] = 1.0; P["l1.wk"][0, 8, 1] = 1.0
    for t in range(6): P["l1.wv"][0, t, t] = 1.0; P["l1.wo"][0, t, 12 + t] = 1.0     # copy the token one-hot found at j
    for t in range(1, 7): P["unembed.w"][12 + t - 1, t] = 10.0
    return P
