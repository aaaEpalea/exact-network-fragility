import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=16, d_head=8, d_mlp=32, causal=False)

def build_params():
    d, dh, dm, H = 16, 8, 32, 1
    P = {}
    P["embed"] = np.zeros((7, d), np.float32)
    P["pos_embed"] = np.zeros((8, d), np.float32)
    P["embed"][0, 0] = 1.0                       # BOS flag in dim 0

    P["l0.wq"] = np.zeros((H, d, dh), np.float32)  # zero q,k -> uniform attention
    P["l0.wk"] = np.zeros((H, d, dh), np.float32)
    P["l0.wv"] = np.zeros((H, d, dh), np.float32)
    P["l0.bq"] = np.zeros((H, dh), np.float32)
    P["l0.bk"] = np.zeros((H, dh), np.float32)
    P["l0.bv"] = np.zeros((H, dh), np.float32)
    P["l0.wo"] = np.zeros((H, dh, d), np.float32)
    P["l0.bo"] = np.zeros(d, np.float32)
    P["l0.wv"][0, 0, 0] = 1.0                    # value = BOS flag
    P["l0.wo"][0, 0, 1] = 1.0                    # write mean flag f = 1/(L+1) into dim 1

    # MLP: piecewise-linear map f -> L through knots (1/k, k-1), k=2..8
    pts = [1.0 / k for k in range(8, 1, -1)]     # ascending f: 1/8 ... 1/2
    vals = [float(k - 1) for k in range(8, 1, -1)]  # 7,6,...,1
    slopes = [(vals[j + 1] - vals[j]) / (pts[j + 1] - pts[j]) for j in range(6)]
    P["l0.w1"] = np.zeros((d, dm), np.float32)
    P["l0.b1"] = np.zeros(dm, np.float32)
    P["l0.w2"] = np.zeros((dm, d), np.float32)
    P["l0.b2"] = np.zeros(d, np.float32)
    for j in range(6):
        P["l0.w1"][1, j] = 1.0
        P["l0.b1"][j] = -pts[j]
        P["l0.w2"][j, 2] = slopes[j] - (slopes[j - 1] if j > 0 else 0.0)
    P["l0.b2"][2] = vals[0]                      # g(f) = 7 + sum d_j relu(f - p_j)

    P["unembed.w"] = np.zeros((d, 1), np.float32)
    P["unembed.w"][2, 0] = 1.0
    P["unembed.b"] = np.zeros(1, np.float32)
    return P