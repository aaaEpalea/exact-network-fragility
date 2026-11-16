import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=3, d_head=1, d_mlp=5, causal=False)


def build_params():
    d, dh, dm = 3, 1, 5
    f = np.float32
    P = {}
    # dim0 = BOS indicator, dim1 = mean of BOS indicator (= 1/(L+1)), dim2 = MLP output
    embed = np.zeros((7, d), f)
    embed[0, 0] = 1.0
    P["embed"] = embed
    P["pos_embed"] = np.zeros((8, d), f)

    # Attention: zero q/k -> uniform attention over all L+1 positions; value = BOS indicator
    P["l0.wq"] = np.zeros((1, d, dh), f)
    P["l0.wk"] = np.zeros((1, d, dh), f)
    P["l0.bq"] = np.zeros((1, dh), f)
    P["l0.bk"] = np.zeros((1, dh), f)
    wv = np.zeros((1, d, dh), f)
    wv[0, 0, 0] = 1.0
    P["l0.wv"] = wv
    P["l0.bv"] = np.zeros((1, dh), f)
    wo = np.zeros((1, dh, d), f)
    wo[0, 0, 1] = 1.0
    P["l0.wo"] = wo
    P["l0.bo"] = np.zeros(d, f)

    # MLP: piecewise-linear interpolation of L = 1/m - 1 through m = 1/(L+1), L=1..7
    # knots m_k = 1/(8-k), y_k = 7-k, k=0..6 ; slope s_k = -(7-k)(8-k)
    s = [-(7 - k) * (8 - k) for k in range(6)]
    m = [1.0 / (8 - k) for k in range(7)]
    w1 = np.zeros((d, dm), f)
    b1 = np.zeros(dm, f)
    w2 = np.zeros((dm, d), f)
    for j, k in enumerate(range(1, 6)):
        w1[1, j] = 1.0
        b1[j] = -m[k]
        w2[j, 2] = s[k] - s[k - 1]
    P["l0.w1"] = w1
    P["l0.b1"] = b1
    P["l0.w2"] = w2
    P["l0.b2"] = np.zeros(d, f)

    uw = np.zeros((d, 1), f)
    uw[1, 0] = s[0]
    uw[2, 0] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.array([7.0 - s[0] * m[0]], f)
    return P