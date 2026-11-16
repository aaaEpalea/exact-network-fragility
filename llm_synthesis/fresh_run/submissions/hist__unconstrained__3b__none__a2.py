import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=9, d_head=7, d_mlp=6, causal=False)

def build_params():
    d, H, dh, dm = 9, 1, 7, 6
    P = {}
    # embeddings: token t (1..6) -> one-hot at dim t-1 ; BOS -> dim 6
    E = np.zeros((7, d), np.float32)
    for t in range(1, 7):
        E[t, t - 1] = 1.0
    E[0, 6] = 1.0
    P["embed"] = E
    P["pos_embed"] = np.zeros((8, d), np.float32)

    A = 20.0 * np.sqrt(dh)          # score gap of 20 after 1/sqrt(dh)
    wq = np.zeros((H, d, dh), np.float32)
    wk = np.zeros((H, d, dh), np.float32)
    wv = np.zeros((H, d, dh), np.float32)
    for i in range(6):
        wq[0, i, i] = A             # query = A * onehot(own token) ...
    for i in range(7):
        wk[0, i, i] = 1.0           # key = [onehot(token), isBOS]
    bq = np.zeros((H, dh), np.float32)
    bq[0, 6] = A                    # ... plus A on the BOS dim (every non-BOS query likes BOS)
    wv[0, 6, 0] = 1.0               # value = BOS indicator
    wo = np.zeros((H, dh, d), np.float32)
    wo[0, 0, 7] = 1.0               # write attention output (= weight on BOS) to dim 7
    P["l0.wq"], P["l0.wk"], P["l0.wv"] = wq, wk, wv
    P["l0.bq"] = bq
    P["l0.bk"] = np.zeros((H, dh), np.float32)
    P["l0.bv"] = np.zeros((H, dh), np.float32)
    P["l0.wo"] = wo
    P["l0.bo"] = np.zeros(d, np.float32)

    # MLP: piecewise-linear interpolation of c = 1/w - 1 through knots w = 1/8..1/2
    ws = np.array([1.0 / k for k in range(8, 1, -1)])   # 1/8 ... 1/2 (increasing)
    vals = np.array([k - 1 for k in range(8, 1, -1)], float)  # 7 ... 1
    slopes = (vals[1:] - vals[:-1]) / (ws[1:] - ws[:-1])      # 6 segments
    coef = np.concatenate([[slopes[0]], slopes[1:] - slopes[:-1]])
    w1 = np.zeros((d, dm), np.float32)
    b1 = np.zeros(dm, np.float32)
    w2 = np.zeros((dm, d), np.float32)
    for m in range(6):
        w1[7, m] = 1.0
        b1[m] = -ws[m]
        w2[m, 8] = coef[m]
    b2 = np.zeros(d, np.float32)
    b2[8] = vals[0]
    P["l0.w1"], P["l0.b1"], P["l0.w2"], P["l0.b2"] = w1, b1, w2, b2

    uw = np.zeros((d, 1), np.float32)
    uw[8, 0] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.zeros(1, np.float32)
    return P