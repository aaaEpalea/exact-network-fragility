"""dyck8, unconstrained, category 3b, no feedback, contaminated run (attempt 1)"""
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
CONFIG = dict(n_layers=2, n_heads=2, d_model=8, d_head=1, d_mlp=2, causal=True)

def build_params():
    cfg = CONFIG; P = zeros(cfg, 2); Qs = 60.0 * np.sqrt(cfg["d_head"]); K = 20.0
    P["embed"][1, 0] = 1.0; P["embed"][2, 1] = 1.0; P["embed"][0, 2] = 1.0
    for h, f in ((0, 0), (1, 1)):
        P["l0.wk"][h, f, 0] = 1.0; P["l0.wk"][h, 2, 0] = 1.0; P["l0.bq"][h, 0] = Qs; P["l0.wv"][h, 2, 0] = 1.0; P["l0.wo"][h, 0, 3 + h] = 1.0
    # layer 0: bad_t = [c2 > c1] (depth negative at t) = [a1 - a2 >= 0.05] -> dim 5
    P["l0.w1"][3, 0] = K; P["l0.w1"][4, 0] = -K; P["l0.w1"][3, 1] = K; P["l0.w1"][4, 1] = -K; P["l0.b1"][1] = -1.0; P["l0.w2"][0, 5] = 1.0; P["l0.w2"][1, 5] = -1.0
    # layer 1: uniform causal average of bad over positions 0..t  (>= 1/8 if any prefix went negative)  ->  valid = relu(1 - 16 * avg)
    P["l1.wv"][0, 5, 0] = 1.0; P["l1.wo"][0, 0, 6] = 1.0; P["l1.w1"][6, 0] = -16.0; P["l1.b1"][0] = 1.0; P["l1.w2"][0, 7] = 1.0
    P["unembed.w"][7, 1] = 1.0; P["unembed.b"][0] = 0.5
    return P
