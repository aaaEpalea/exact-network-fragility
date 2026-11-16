import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=9, d_head=6, d_mlp=6, causal=False)

def build_params():
    d, H, dh, dm = 9, 1, 6, 6
    A = 20.0
    P = {}
    emb = np.zeros((7, d), np.float32)
    for v in range(1, 7):
        emb[v, v - 1] = 1.0          # token one-hot in dims 0..5
    emb[0, 6] = 1.0                  # BOS flag in dim 6
    P["embed"] = emb
    P["pos_embed"] = np.zeros((8, d), np.float32)

    wq = np.zeros((H, d, dh), np.float32)
    wk = np.zeros((H, d, dh), np.float32)
    wv = np.zeros((H, d, dh), np.float32)
    for i in range(6):
        wq[0, i, i] = A * np.sqrt(dh)   # score = A on equal tokens (after 1/sqrt(dh))
        wk[0, i, i] = 1.0
    wk[0, 6, :] = 1.0                   # BOS key matches every query with score A
    wv[0, 6, 0] = 1.0                   # only BOS carries a value (1)
    P["l0.wq"], P["l0.wk"], P["l0.wv"] = wq, wk, wv
    P["l0.bq"] = np.zeros((H, dh), np.float32)
    P["l0.bk"] = np.zeros((H, dh), np.float32)
    P["l0.bv"] = np.zeros((H, dh), np.float32)
    wo = np.zeros((H, dh, d), np.float32)
    wo[0, 0, 7] = 1.0                   # write u = 1/(1+c) into dim 7
    P["l0.wo"] = wo
    P["l0.bo"] = np.zeros(d, np.float32)

    # MLP: piecewise-linear interpolation of c = 1/u - 1 through nodes u=1/(8-k), c=7-k
    w1 = np.zeros((d, dm), np.float32)
    b1 = np.zeros(dm, np.float32)
    w2 = np.zeros((dm, d), np.float32)
    slopes = [-(7.0 - k) * (8.0 - k) for k in range(6)]
    for k in range(6):
        w1[7, k] = 1.0
        b1[k] = -1.0 / (8.0 - k)
        w2[k, 8] = slopes[k] - (slopes[k - 1] if k > 0 else 0.0)
    P["l0.w1"], P["l0.b1"], P["l0.w2"] = w1, b1, w2
    b2 = np.zeros(d, np.float32)
    b2[8] = 7.0
    P["l0.b2"] = b2

    uw = np.zeros((d, 1), np.float32)
    uw[8, 0] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.zeros(1, np.float32)
    return P