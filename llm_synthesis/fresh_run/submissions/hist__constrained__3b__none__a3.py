import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=16, d_head=8, d_mlp=32, causal=False)

def build_params():
    d, dh, H, dm = 16, 8, 1, 32
    M = 100.0                       # large logit offset (pre-scaling)
    P = {}
    P["embed"] = np.zeros((7, d), np.float32)
    P["embed"][0, 6] = 1.0                      # BOS flag in dim 6
    for t in range(1, 7):
        P["embed"][t, t - 1] = 1.0              # one-hot token in dims 0..5
    P["pos_embed"] = np.zeros((8, d), np.float32)

    wq = np.zeros((H, d, dh), np.float32); bq = np.zeros((H, dh), np.float32)
    wk = np.zeros((H, d, dh), np.float32); bk = np.zeros((H, dh), np.float32)
    wv = np.zeros((H, d, dh), np.float32); bv = np.zeros((H, dh), np.float32)
    wo = np.zeros((H, dh, d), np.float32)
    for e in range(6):
        wq[0, e, e] = M          # q[e] = M * onehot_e
        wk[0, e, e] = 1.0        # k[e] = onehot_e
    bq[0, 6] = 1.0               # q[6] = 1
    wk[0, 6, 6] = M              # k[6] = -M for real tokens, 0 for BOS
    bk[0, 6] = -M
    wv[0, 6, 0] = 1.0            # value = BOS flag
    wo[0, 0, 8] = 1.0            # attention-on-BOS written to residual dim 8
    P.update({"l0.wq": wq, "l0.bq": bq, "l0.wk": wk, "l0.bk": bk,
              "l0.wv": wv, "l0.bv": bv, "l0.wo": wo,
              "l0.bo": np.zeros(d, np.float32)})

    # MLP: piecewise-linear map a0 = 1/(1+n)  ->  n
    pts = [1.0 / (9 - j) for j in range(7)]      # 1/8, 1/7, ..., 1/2
    ys = [7 - j for j in range(7)]               # 7, 6, ..., 1
    slopes = [(ys[i + 1] - ys[i]) / (pts[i + 1] - pts[i]) for i in range(6)]
    w1 = np.zeros((d, dm), np.float32); b1 = np.zeros(dm, np.float32)
    w2 = np.zeros((dm, d), np.float32); b2 = np.zeros(d, np.float32)
    for i in range(6):
        w1[8, i] = 1.0
        b1[i] = -pts[i]
        w2[i, 9] = slopes[i] - (slopes[i - 1] if i > 0 else 0.0)
    b2[9] = 7.0
    P.update({"l0.w1": w1, "l0.b1": b1, "l0.w2": w2, "l0.b2": b2})

    uw = np.zeros((d, 1), np.float32); uw[9, 0] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.zeros(1, np.float32)
    return {k: v.astype(np.float32) for k, v in P.items()}