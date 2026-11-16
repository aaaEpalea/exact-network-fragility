import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=16, d_head=8, d_mlp=32, causal=False)

def build_params():
    d, H, dh, dm = 16, 1, 8, 32
    P = {}
    P["embed"] = np.zeros((7, d), np.float32)
    P["pos_embed"] = np.zeros((8, d), np.float32)
    for t in range(1, 7):
        P["embed"][t, t - 1] = 1.0      # one-hot token identity in dims 0..5
    P["embed"][0, 6] = 1.0              # BOS flag in dim 6

    s8 = np.sqrt(8.0)
    C = 20.0
    M = C + np.log(2.0)
    wq = np.zeros((H, d, dh)); wk = np.zeros((H, d, dh)); wv = np.zeros((H, d, dh))
    bq = np.zeros((H, dh)); bk = np.zeros((H, dh)); bv = np.zeros((H, dh))
    for t in range(6):
        wq[0, t, t] = s8 * M
        wk[0, t, t] = 1.0
    bq[0, 6] = s8 * C                   # constant query component
    wk[0, 6, 6] = 1.0                   # BOS key flag
    wv[0, 6, 0] = 1.0                   # value = BOS flag
    wo = np.zeros((H, dh, d)); wo[0, 0, 8] = 1.0   # write attention output to dim 8
    P.update({"l0.wq": wq, "l0.wk": wk, "l0.wv": wv, "l0.bq": bq, "l0.bk": bk,
              "l0.bv": bv, "l0.wo": wo, "l0.bo": np.zeros(d)})

    # a(c) = 1/(2c+1); piecewise-linear decode a -> c through nodes c=1..7
    a = np.array([1.0 / (2 * c + 1) for c in range(1, 8)])
    slopes = np.array([1.0 / (a[k] - a[k + 1]) for k in range(6)])
    w = np.zeros(6)
    w[0] = slopes[0]
    for k in range(1, 6):
        w[k] = slopes[k] - slopes[k - 1]
    w1 = np.zeros((d, dm)); b1 = np.zeros(dm); w2 = np.zeros((dm, d)); b2 = np.zeros(d)
    for k in range(6):
        w1[8, k] = -1.0
        b1[k] = a[k]
        w2[k, 9] = w[k]
    b2[9] = 1.0
    P.update({"l0.w1": w1, "l0.b1": b1, "l0.w2": w2, "l0.b2": b2})

    uw = np.zeros((d, 1)); uw[9, 0] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.zeros(1)
    return {k: np.asarray(v, np.float32) for k, v in P.items()}