import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=16, d_head=8, d_mlp=32, causal=False)

def build_params():
    d, dh, dm, H = 16, 8, 32, 1
    P = {}
    P["embed"] = np.zeros((7, d), np.float32)
    P["embed"][0, 0] = 1.0                      # BOS indicator in dim 0
    P["pos_embed"] = np.zeros((8, d), np.float32)

    # attention: q = k = 0 -> all scores 0 -> uniform attention (weight 1/(L+1))
    P["l0.wq"] = np.zeros((H, d, dh), np.float32)
    P["l0.wk"] = np.zeros((H, d, dh), np.float32)
    P["l0.wv"] = np.zeros((H, d, dh), np.float32)
    P["l0.wv"][0, 0, 0] = 1.0                   # value = BOS indicator
    P["l0.bq"] = np.zeros((H, dh), np.float32)
    P["l0.bk"] = np.zeros((H, dh), np.float32)
    P["l0.bv"] = np.zeros((H, dh), np.float32)
    P["l0.wo"] = np.zeros((H, dh, d), np.float32)
    P["l0.wo"][0, 0, 1] = 1.0                   # write u = 1/(L+1) into dim 1
    P["l0.bo"] = np.zeros(d, np.float32)

    # MLP: piecewise-linear map u=1/(L+1) -> L, interpolating the 7 points exactly
    w1 = np.zeros((d, dm), np.float64); b1 = np.zeros(dm, np.float64)
    w2 = np.zeros((dm, d), np.float64); b2 = np.zeros(d, np.float64)
    slopes = [-(7 - k) * (8 - k) for k in range(7)]   # slope of segment k (points L=7-k -> 6-k)
    prev = 0.0
    for k in range(6):
        a = 1.0 / (8 - k)                       # breakpoint at u of L = 7-k
        w1[1, k] = 1.0
        b1[k] = -a
        w2[k, 2] = slopes[k] - prev
        prev = slopes[k]
    b2[2] = 7.0                                  # value at u = 1/8 (L = 7)
    P["l0.w1"] = w1.astype(np.float32); P["l0.b1"] = b1.astype(np.float32)
    P["l0.w2"] = w2.astype(np.float32); P["l0.b2"] = b2.astype(np.float32)

    P["unembed.w"] = np.zeros((d, 1), np.float32)
    P["unembed.w"][2, 0] = 1.0
    P["unembed.b"] = np.zeros(1, np.float32)
    return P