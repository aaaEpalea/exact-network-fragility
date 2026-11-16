import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=16, d_head=8, d_mlp=32, causal=False)

def build_params():
    d, dh, dm, H = 16, 8, 32, 1
    P = {}
    z = lambda *s: np.zeros(s, np.float32)

    # embeddings: BOS carries a 1 in dim 0, everything else is 0
    P["embed"] = z(7, d); P["embed"][0, 0] = 1.0
    P["pos_embed"] = z(8, d)

    # attention: zero q/k -> uniform attention over all L+1 positions
    P["l0.wq"] = z(H, d, dh); P["l0.bq"] = z(H, dh)
    P["l0.wk"] = z(H, d, dh); P["l0.bk"] = z(H, dh)
    P["l0.wv"] = z(H, d, dh); P["l0.bv"] = z(H, dh)
    P["l0.wv"][0, 0, 0] = 1.0            # value = BOS indicator
    P["l0.wo"] = z(H, dh, d)
    P["l0.wo"][0, 0, 1] = 1.0            # mean -> residual dim 1 = 1/(L+1)
    P["l0.bo"] = z(d)

    # MLP: piecewise-linear interpolation of g(1/(L+1)) = L
    ts = [1.0 / (L + 1) for L in range(7, 0, -1)]   # ascending: 1/8 ... 1/2
    ys = [float(L) for L in range(7, 0, -1)]        # 7 ... 1
    slopes = [(ys[j + 1] - ys[j]) / (ts[j + 1] - ts[j]) for j in range(6)]
    P["l0.w1"] = z(d, dm); P["l0.b1"] = z(dm)
    P["l0.w2"] = z(dm, d); P["l0.b2"] = z(d)
    for j in range(6):
        P["l0.w1"][1, j] = 1.0
        P["l0.b1"][j] = -ts[j]
        P["l0.w2"][j, 2] = slopes[j] - (slopes[j - 1] if j > 0 else 0.0)
    P["l0.b2"][2] = ys[0]                            # g(1/8) = 7

    # unembed reads dim 2
    P["unembed.w"] = z(d, 1); P["unembed.w"][2, 0] = 1.0
    P["unembed.b"] = z(1)
    return P