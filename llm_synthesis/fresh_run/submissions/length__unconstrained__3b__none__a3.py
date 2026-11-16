import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=3, d_head=1, d_mlp=6, causal=False)

def build_params():
    d, dh, H, dm = 3, 1, 1, 6
    f = np.float32
    P = {}
    # dim0 = BOS indicator, dim1 = mean of that indicator (=1/T), dim2 = output
    embed = np.zeros((7, d), f); embed[0, 0] = 1.0
    P["embed"] = embed
    P["pos_embed"] = np.zeros((8, d), f)

    # attention: zero q/k -> all scores 0 -> exactly uniform softmax over all T=L+1 positions
    P["l0.wq"] = np.zeros((H, d, dh), f); P["l0.bq"] = np.zeros((H, dh), f)
    P["l0.wk"] = np.zeros((H, d, dh), f); P["l0.bk"] = np.zeros((H, dh), f)
    wv = np.zeros((H, d, dh), f); wv[0, 0, 0] = 1.0     # value = BOS indicator
    P["l0.wv"] = wv; P["l0.bv"] = np.zeros((H, dh), f)
    wo = np.zeros((H, dh, d), f); wo[0, 0, 1] = 1.0     # write mean into dim1
    P["l0.wo"] = wo; P["l0.bo"] = np.zeros(d, f)

    # MLP: piecewise-linear interpolation of g(1/(L+1)) = L at the 7 possible points
    p = [1.0 / (8 - j) for j in range(7)]        # 1/8 ... 1/2  (L = 7 ... 1)
    y = [7.0 - j for j in range(7)]
    slopes = [(y[j + 1] - y[j]) / (p[j + 1] - p[j]) for j in range(6)]
    c = [slopes[0]] + [slopes[j] - slopes[j - 1] for j in range(1, 6)]
    w1 = np.zeros((d, dm), f); b1 = np.zeros(dm, f); w2 = np.zeros((dm, d), f)
    for j in range(6):
        w1[1, j] = 1.0
        b1[j] = -p[j]
        w2[j, 2] = c[j]
    P["l0.w1"] = w1; P["l0.b1"] = b1; P["l0.w2"] = w2; P["l0.b2"] = np.zeros(d, f)

    # output = 7 + sum_j c_j * relu(a - p_j), read from dim2 only
    uw = np.zeros((d, 1), f); uw[2, 0] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.array([7.0], f)
    return P