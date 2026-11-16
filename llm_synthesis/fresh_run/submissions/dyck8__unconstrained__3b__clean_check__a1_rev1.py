import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=128, causal=True)

# Only the first few residual dims / head 0 / first 2 MLP units are used; the rest are zero.
# Residual dims: 0 = signed token value (+1 open, -1 close, 0 otherwise)
#                1 = mean of signed values over positions 0..t  (= balance/(t+1))
#                2 = "balance is negative here" flag
#                3 = mean of flags over positions 0..t
#                4 = "some prefix went negative" (absorbing)


def build_params():
    d, dh, dm, H = 32, 16, 128, 2
    f = np.float32
    P = {}
    emb = np.zeros((7, d), f)
    emb[1, 0] = 1.0
    emb[2, 0] = -1.0
    P["embed"] = emb
    P["pos_embed"] = np.zeros((8, d), f)

    def layer(i, src, dst, w1v, b1v, w2v):
        P[f"l{i}.wq"] = np.zeros((H, d, dh), f)
        P[f"l{i}.wk"] = np.zeros((H, d, dh), f)
        P[f"l{i}.bq"] = np.zeros((H, dh), f)
        P[f"l{i}.bk"] = np.zeros((H, dh), f)
        wv = np.zeros((H, d, dh), f)
        wv[0, src, 0] = 1.0
        P[f"l{i}.wv"] = wv
        P[f"l{i}.bv"] = np.zeros((H, dh), f)
        wo = np.zeros((H, dh, d), f)
        wo[0, 0, dst] = 1.0
        P[f"l{i}.wo"] = wo
        P[f"l{i}.bo"] = np.zeros(d, f)
        w1 = np.zeros((d, dm), f)
        b1 = np.zeros(dm, f)
        w2 = np.zeros((dm, d), f)
        w1[:, :2] = w1v
        b1[:2] = b1v
        w2[:2, :] = w2v
        P[f"l{i}.w1"] = w1
        P[f"l{i}.b1"] = b1
        P[f"l{i}.w2"] = w2
        P[f"l{i}.b2"] = np.zeros(d, f)

    # layer 0: mean of signed values -> negative-balance flag = clip(-8*avg, 0, 1)
    w1 = np.zeros((d, 2), f)
    w1[1, 0] = -8.0
    w1[1, 1] = -8.0
    w2 = np.zeros((2, d), f)
    w2[0, 2] = 1.0
    w2[1, 2] = -1.0
    layer(0, 0, 1, w1, np.array([0.0, -1.0], f), w2)

    # layer 1: mean of flags -> any-flag = clip(8*avg, 0, 1)
    w1 = np.zeros((d, 2), f)
    w1[3, 0] = 8.0
    w1[3, 1] = 8.0
    w2 = np.zeros((2, d), f)
    w2[0, 4] = 1.0
    w2[1, 4] = -1.0
    layer(1, 2, 3, w1, np.array([0.0, -1.0], f), w2)

    uw = np.zeros((d, 2), f)
    uw[4, 0] = 1.0
    uw[4, 1] = -1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.array([0.0, 1.0], f)
    return P