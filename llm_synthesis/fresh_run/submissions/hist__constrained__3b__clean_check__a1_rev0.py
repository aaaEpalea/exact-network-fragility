import numpy as np
CONFIG = dict(n_layers=1, n_heads=1, d_model=16, d_head=8, d_mlp=32, causal=False)

def build_params():
    d, H, dh, dm = 16, 1, 8, 32
    f = np.float32
    P = {}
    P["embed"] = np.zeros((7, d), f)
    for t in range(1, 7):
        P["embed"][t, t - 1] = 1.0      # token t -> dim t-1
    P["embed"][0, 6] = 1.0              # BOS -> dim 6
    P["pos_embed"] = np.zeros((8, d), f)
    c = 12.0
    s = c * np.sqrt(dh)
    wq = np.zeros((H, d, dh), f); wk = np.zeros((H, d, dh), f); wv = np.zeros((H, d, dh), f)
    for t in range(6):
        wq[0, t, t] = s
        wk[0, t, t] = 1.0
    wk[0, 6, 6] = 1.0
    bq = np.zeros((H, dh), f); bq[0, 6] = s   # every query also matches BOS key
    wv[0, 6, 0] = 1.0                          # value = BOS indicator
    wo = np.zeros((H, dh, d), f); wo[0, 0, 7] = 1.0   # -> residual dim 7 (~ 1/(n+1))
    P.update({"l0.wq": wq, "l0.wk": wk, "l0.wv": wv, "l0.bq": bq,
              "l0.bk": np.zeros((H, dh), f), "l0.bv": np.zeros((H, dh), f),
              "l0.wo": wo, "l0.bo": np.zeros(d, f)})
    # MLP: piecewise-linear interpolation g=1/(n+1) -> n
    g = [1.0 / (8 - k) for k in range(7)]      # knots, n = 7-k
    slopes = [-1.0 / (g[k + 1] - g[k]) for k in range(6)]
    w1 = np.zeros((d, dm), f); b1 = np.zeros(dm, f); w2 = np.zeros((dm, d), f)
    for k in range(6):
        w1[7, k] = 1.0
        b1[k] = -g[k]
        w2[k, 8] = slopes[k] - (slopes[k - 1] if k > 0 else 0.0)
    P.update({"l0.w1": w1, "l0.b1": b1, "l0.w2": w2, "l0.b2": np.zeros(d, f)})
    uw = np.zeros((d, 1), f); uw[8, 0] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.array([7.0], f)
    return P