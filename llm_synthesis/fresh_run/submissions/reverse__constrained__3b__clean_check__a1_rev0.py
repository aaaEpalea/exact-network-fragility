import numpy as np
CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=64, causal=False)

def build_params():
    d, H, dh, dm = 32, 2, 16, 64
    f = np.float32
    P = {}
    emb = np.zeros((7, d), f)
    for t in range(7): emb[t, t] = 1
    pe = np.zeros((8, d), f)
    for p in range(8): pe[p, 7] = p; pe[p, 8] = p * p
    P["embed"], P["pos_embed"] = emb, pe
    for i in range(2):
        for n in ("wq", "wk", "wv"): P[f"l{i}.{n}"] = np.zeros((H, d, dh), f)
        for n in ("bq", "bk", "bv"): P[f"l{i}.{n}"] = np.zeros((H, dh), f)
        P[f"l{i}.wo"] = np.zeros((H, dh, d), f); P[f"l{i}.bo"] = np.zeros(d, f)
        P[f"l{i}.w1"] = np.zeros((d, dm), f); P[f"l{i}.b1"] = np.zeros(dm, f)
        P[f"l{i}.w2"] = np.zeros((dm, d), f); P[f"l{i}.b2"] = np.zeros(d, f)
    # layer 0: uniform attention averages BOS indicator -> u = 1/T in dim 9
    P["l0.wv"][0, 0, 0] = 1; P["l0.wo"][0, 0, 9] = 1
    # MLP: piecewise-linear 1/T -> T, then c = T - i in dim 11
    ts = [1.0 / T for T in range(8, 1, -1)]
    g = [float(T) for T in range(8, 1, -1)]
    slopes = [(g[k+1]-g[k])/(ts[k+1]-ts[k]) for k in range(6)]
    for k in range(6):
        d_k = slopes[k] - (slopes[k-1] if k else 0.0)
        P["l0.w1"][9, k] = 1; P["l0.b1"][k] = -ts[k]; P["l0.w2"][k, 11] = d_k
    P["l0.w1"][7, 6] = 1; P["l0.w2"][6, 11] = -1; P["l0.b2"][11] = 8.0
    # layer 1: head 0 attends to j = T - i via score S*(c^2-(j-c)^2)
    S = 20.0
    P["l1.bq"][0, 0] = 4 * S; P["l1.wq"][0, 11, 1] = 8 * S
    P["l1.wk"][0, 8, 0] = -1; P["l1.wk"][0, 7, 1] = 1
    for t in range(7):
        P["l1.wv"][0, t, t] = 1; P["l1.wo"][0, t, 12 + t] = 1
    P["unembed.w"] = np.zeros((d, 7), f); P["unembed.b"] = np.zeros(7, f)
    for t in range(7): P["unembed.w"][12 + t, t] = 1
    return P