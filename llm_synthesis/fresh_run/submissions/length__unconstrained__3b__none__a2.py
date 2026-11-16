import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=3, d_head=1, d_mlp=6, causal=False)

def build_params():
    d, H, dh, dm = 3, 1, 1, 6
    f = np.float32
    P = {}
    # dim0 = BOS indicator, dim1 = mean of that indicator (u = 1/(L+1)), dim2 = output
    embed = np.zeros((7, d), f); embed[0, 0] = 1.0
    P["embed"] = embed
    P["pos_embed"] = np.zeros((8, d), f)

    # attention: zero q/k -> all scores 0 -> exactly uniform attention over L+1 positions
    P["l0.wq"] = np.zeros((H, d, dh), f); P["l0.bq"] = np.zeros((H, dh), f)
    P["l0.wk"] = np.zeros((H, d, dh), f); P["l0.bk"] = np.zeros((H, dh), f)
    wv = np.zeros((H, d, dh), f); wv[0, 0, 0] = 1.0       # v = BOS indicator
    P["l0.wv"] = wv; P["l0.bv"] = np.zeros((H, dh), f)
    wo = np.zeros((H, dh, d), f); wo[0, 0, 1] = 1.0       # write the mean into dim1
    P["l0.wo"] = wo; P["l0.bo"] = np.zeros(d, f)

    # MLP: piecewise-linear interpolation of f(1/(L+1)) = L, nodes u_k = 1/(8-k), k=0..6, values 7-k
    u = [1.0 / (8 - k) for k in range(7)]
    val = [7.0 - k for k in range(7)]
    s = [(val[j + 1] - val[j]) / (u[j + 1] - u[j]) for j in range(6)]
    w1 = np.zeros((d, dm), f); b1 = np.zeros(dm, f); w2 = np.zeros((dm, d), f)
    for j in range(6):
        w1[1, j] = 1.0
        b1[j] = -u[j]
        w2[j, 2] = s[0] if j == 0 else s[j] - s[j - 1]
    b2 = np.zeros(d, f); b2[2] = val[0]
    P["l0.w1"], P["l0.b1"], P["l0.w2"], P["l0.b2"] = w1, b1, w2, b2

    uw = np.zeros((d, 1), f); uw[2, 0] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.zeros(1, f)
    return P