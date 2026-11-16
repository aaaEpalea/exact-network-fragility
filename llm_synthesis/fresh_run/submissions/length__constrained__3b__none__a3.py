import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=16, d_head=8, d_mlp=32, causal=False)

def build_params():
    d, H, dh, dm = 16, 1, 8, 32
    P = {}
    f = lambda *s: np.zeros(s, dtype=np.float32)
    P["embed"] = f(7, d)
    P["pos_embed"] = f(8, d)
    P["embed"][0, 0] = 1.0            # BOS indicator in residual dim 0

    # attention: zero q/k -> uniform attention over all L+1 positions
    P["l0.wq"] = f(H, d, dh); P["l0.bq"] = f(H, dh)
    P["l0.wk"] = f(H, d, dh); P["l0.bk"] = f(H, dh)
    P["l0.wv"] = f(H, d, dh); P["l0.bv"] = f(H, dh)
    P["l0.wo"] = f(H, dh, d); P["l0.bo"] = f(d)
    P["l0.wv"][0, 0, 0] = 1.0         # value = BOS indicator
    P["l0.wo"][0, 0, 1] = 1.0         # write s = 1/(L+1) into residual dim 1

    # MLP: piecewise-linear map g(s) = L at s = 1/(L+1), L = 1..7
    P["l0.w1"] = f(d, dm); P["l0.b1"] = f(dm)
    P["l0.w2"] = f(dm, d); P["l0.b2"] = f(d)
    knots = [1.0 / (8 - j) for j in range(6)]          # 1/8, 1/7, ..., 1/3
    slopes = [-1.0 / (1.0 / (7 - j) - 1.0 / (8 - j)) for j in range(6)]
    prev = 0.0
    for j in range(6):
        P["l0.w1"][1, j] = 1.0
        P["l0.b1"][j] = -knots[j]
        P["l0.w2"][j, 2] = slopes[j] - prev
        prev = slopes[j]
    P["l0.b2"][2] = 7.0               # g(1/8) = 7

    P["unembed.w"] = f(d, 1); P["unembed.b"] = f(1)
    P["unembed.w"][2, 0] = 1.0
    return P