import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=9, d_head=7, d_mlp=6, causal=False)


def build_params():
    d, dh, dm = 9, 7, 6
    A = 20.0          # score for "same token"
    C = 20.0          # score for attending to BOS (equal to A, so weight on BOS = 1/(n+1))
    s = np.sqrt(dh)
    P = {}
    embed = np.zeros((7, d), np.float32)
    for t in range(1, 7):
        embed[t, t - 1] = 1.0          # one-hot of token in dims 0..5
    embed[0, 6] = 1.0                  # BOS flag in dim 6
    P["embed"] = embed
    P["pos_embed"] = np.zeros((8, d), np.float32)

    wq = np.zeros((1, d, dh), np.float32)
    wk = np.zeros((1, d, dh), np.float32)
    wv = np.zeros((1, d, dh), np.float32)
    bq = np.zeros((1, dh), np.float32)
    for i in range(6):
        wq[0, i, i] = A * s
    bq[0, 6] = C * s                   # constant query on the BOS-flag channel
    for i in range(7):
        wk[0, i, i] = 1.0
    wv[0, 6, 0] = 1.0                  # value = BOS flag
    wo = np.zeros((1, dh, d), np.float32)
    wo[0, 0, 7] = 1.0                  # w = 1/(n+1) written into dim 7
    P["l0.wq"], P["l0.wk"], P["l0.wv"] = wq, wk, wv
    P["l0.bq"], P["l0.bk"], P["l0.bv"] = bq, np.zeros((1, dh), np.float32), np.zeros((1, dh), np.float32)
    P["l0.wo"] = wo
    P["l0.bo"] = np.zeros(d, np.float32)

    # MLP: piecewise-linear map w=1/(n+1) -> n, exact at the knots
    w1 = np.zeros((d, dm), np.float32)
    b1 = np.zeros(dm, np.float32)
    w2 = np.zeros((dm, d), np.float32)
    dprev = 0.0
    for k in range(1, 7):
        tk = 1.0 / (k + 1)
        dk = (k + 1) * (k + 2)         # 1/(t_k - t_{k+1})
        w1[7, k - 1] = -1.0
        b1[k - 1] = tk
        w2[k - 1, 8] = dk - dprev
        dprev = dk
    P["l0.w1"], P["l0.b1"], P["l0.w2"] = w1, b1, w2
    P["l0.b2"] = np.zeros(d, np.float32)

    uw = np.zeros((d, 1), np.float32)
    uw[8, 0] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.ones(1, np.float32)   # n = 1 at w = 1/2
    return P