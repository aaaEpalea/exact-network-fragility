"""Authoring script of the CONTAMINATED run (category 3b): writes the 12 priority-1 submissions (6 tasks x constrained/unconstrained, no feedback).
Written before any evaluation; the harness was NOT run on these constructions before they were finalised."""
import os, textwrap
HELP = '''import numpy as np
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
'''
BODY = {}
BODY["length"] = '''
def build_params():
    cfg = CONFIG; P = zeros(cfg, 1)
    P["embed"][0, 0] = 1.0                                   # residual dim 0: BOS flag
    P["l0.wv"][0, 0, 0] = 1.0; P["l0.wo"][0, 0, 1] = 1.0     # q=k=0 -> uniform attention; value = BOS flag -> dim 1 holds a = 1/(L+1)
    c0, th, w = interp([1.0 / (k + 1) for k in range(1, 8)], [float(k) for k in range(1, 8)])   # a -> L
    for j in range(len(th)): P["l0.w1"][1, j] = 1.0; P["l0.b1"][j] = -th[j]; P["l0.w2"][j, 2] = w[j]
    P["l0.b2"][2] = c0; P["unembed.w"][2, 0] = 1.0
    return P
'''
BODY["hist"] = '''
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
'''
BODY["reverse"] = '''
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
'''
BODY["parity"] = '''
def build_params():
    cfg = CONFIG; P = zeros(cfg, 2); Qs = 60.0 * np.sqrt(cfg["d_head"])
    P["embed"][1, 0] = 1.0; P["embed"][0, 1] = 1.0           # dim 0: "token 1" flag, dim 1: BOS flag
    P["l0.wk"][0, 0, 0] = 1.0; P["l0.wk"][0, 1, 0] = 1.0; P["l0.bq"][0, 0] = Qs   # causal attention to {positions with token 1} U {BOS}
    P["l0.wv"][0, 1, 0] = 1.0; P["l0.wo"][0, 0, 2] = 1.0     # dim 2 = weight on BOS = 1/(c+1), c = number of 1s so far
    c0, th, w = interp([1.0 / (c + 1) for c in range(0, 8)], [float(c % 2) for c in range(0, 8)])      # a -> parity(c)
    for j in range(len(th)): P["l0.w1"][2, j] = 1.0; P["l0.b1"][j] = -th[j]; P["l0.w2"][j, 3] = w[j]
    P["l0.b2"][3] = c0; P["unembed.w"][3, 1] = 1.0; P["unembed.b"][0] = 0.5
    return P
'''
BODY["majority8"] = '''
def build_params():
    cfg = CONFIG; P = zeros(cfg, 2); Qs = 60.0 * np.sqrt(cfg["d_head"]); K = 20.0
    P["embed"][1, 0] = 1.0; P["embed"][2, 1] = 1.0; P["embed"][0, 2] = 1.0          # dims: token1 flag, token2 flag, BOS flag
    for h, f in ((0, 0), (1, 1)):                                                    # head 0 counts token 1, head 1 counts token 2 (BOS trick)
        P["l0.wk"][h, f, 0] = 1.0; P["l0.wk"][h, 2, 0] = 1.0; P["l0.bq"][h, 0] = Qs; P["l0.wv"][h, 2, 0] = 1.0; P["l0.wo"][h, 0, 3 + h] = 1.0
    # dims 3,4 = a1 = 1/(c1+1), a2 = 1/(c2+1);  c1 > c2  <=>  a2 - a1 >= 0.05 ;  y1 = relu(K d) - relu(K d - 1)
    P["l0.w1"][3, 0] = -K; P["l0.w1"][4, 0] = K; P["l0.w1"][3, 1] = -K; P["l0.w1"][4, 1] = K; P["l0.b1"][1] = -1.0
    P["l0.w2"][0, 5] = 1.0; P["l0.w2"][1, 5] = -1.0; P["unembed.w"][5, 1] = 1.0; P["unembed.b"][0] = 0.5
    return P
'''
BODY["dyck8"] = '''
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
'''
CONS = {"length": (1, 1, 16, 8, 32, False), "hist": (1, 1, 16, 8, 32, False), "reverse": (2, 2, 32, 16, 64, False), "parity": (2, 2, 16, 8, 64, True), "majority8": (2, 2, 32, 16, 128, True), "dyck8": (2, 2, 32, 16, 128, True)}
UNC = {"length": (1, 1, 4, 1, 6, False), "hist": (1, 1, 9, 7, 6, False), "reverse": (2, 1, 18, 6, 8, False), "parity": (1, 1, 4, 1, 7, True), "majority8": (1, 2, 6, 1, 2, True), "dyck8": (2, 2, 8, 1, 2, True)}
out = os.path.dirname(os.path.abspath(__file__))
for task in BODY:
    for cond, tab in (("constrained", CONS), ("unconstrained", UNC)):
        L, H, d, dh, dm, c = tab[task]; cfg = f"CONFIG = dict(n_layers={L}, n_heads={H}, d_model={d}, d_head={dh}, d_mlp={dm}, causal={c})\n"
        open(os.path.join(out, f"contam__{task}__{cond}__3b__none__a1.py"), "w").write(f'"""{task}, {cond}, category 3b, no feedback, contaminated run (attempt 1)"""\n' + HELP + cfg + BODY[task])
print("written")
