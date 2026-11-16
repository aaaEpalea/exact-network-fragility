import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=16, d_head=8, d_mlp=32, causal=False)

def build_params():
    d, dh, dm, H = 16, 8, 32, 1
    P = {}
    z = lambda *s: np.zeros(s, np.float64)
    embed = z(7, d)
    for t in range(7):
        embed[t, t] = 1.0            # one-hot token in dims 0..6 (dim 0 = BOS)
    P["embed"] = embed
    P["pos_embed"] = z(8, d)

    S = np.sqrt(dh)
    A = 10.0                          # effective match score
    wq, wk, wv, wo = z(H, d, dh), z(H, d, dh), z(H, d, dh), z(H, dh, d)
    bq, bk, bv = z(H, dh), z(H, dh), z(H, dh)
    for t in range(1, 7):
        wq[0, t, t - 1] = A * S       # query: scaled one-hot of own token
        wk[0, t, t - 1] = 1.0         # key: one-hot of own token
    bq[0, 6] = 1.0                    # constant query feature
    wk[0, 0, 6] = A * S               # BOS key gets effective score A
    wv[0, 0, 0] = 1.0                 # value: BOS indicator
    wo[0, 0, 7] = 1.0                 # write attention-to-BOS weight into dim 7
    P.update({"l0.wq": wq, "l0.wk": wk, "l0.wv": wv, "l0.bq": bq, "l0.bk": bk,
              "l0.bv": bv, "l0.wo": wo, "l0.bo": z(d)})

    # w ~= 1/(c+1); decode c = 1/w - 1 by piecewise-linear interpolation (exact at knots)
    us = [1.0 / (c + 1) for c in range(7, 0, -1)]   # 1/8 ... 1/2
    fs = list(range(7, 0, -1))                      # 7 ... 1
    m = [(fs[i + 1] - fs[i]) / (us[i + 1] - us[i]) for i in range(6)]
    w1, b1, w2 = z(d, dm), z(dm), z(dm, d)
    uw, ub = z(d, 1), 0.0
    uw = z(d, 1)
    uw[7, 0] = m[0]
    for j in range(1, 6):
        k = j - 1
        w1[7, k] = 1.0
        b1[k] = -us[j]
        w2[k, 8 + k] = 1.0
        uw[8 + k, 0] = m[j] - m[j - 1]
    P.update({"l0.w1": w1, "l0.b1": b1, "l0.w2": w2, "l0.b2": z(d)})
    P["unembed.w"] = uw
    P["unembed.b"] = np.array([fs[0] - m[0] * us[0]])
    return {k: np.asarray(v, np.float32) for k, v in P.items()}