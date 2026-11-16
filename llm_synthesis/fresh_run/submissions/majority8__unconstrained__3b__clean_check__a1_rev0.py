import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=2, d_head=1, d_mlp=1, causal=True)


def build_params():
    f = np.float32
    d, H, dh, dm = 2, 1, 1, 1
    P = {}
    # dim 0 = signed count contribution of the token (+1 for token 1, -1 for token 2)
    embed = np.zeros((7, d), f)
    embed[1, 0] = 1.0
    embed[2, 0] = -1.0
    P["embed"] = embed
    P["pos_embed"] = np.zeros((8, d), f)
    # zero q/k -> uniform causal attention over positions 0..t
    P["l0.wq"] = np.zeros((H, d, dh), f)
    P["l0.wk"] = np.zeros((H, d, dh), f)
    P["l0.bq"] = np.zeros((H, dh), f)
    P["l0.bk"] = np.zeros((H, dh), f)
    wv = np.zeros((H, d, dh), f)
    wv[0, 0, 0] = 1.0
    P["l0.wv"] = wv
    P["l0.bv"] = np.zeros((H, dh), f)
    wo = np.zeros((H, dh, d), f)
    wo[0, 0, 1] = 1.0  # mean (c1-c2)/(t+1) written to dim 1
    P["l0.wo"] = wo
    P["l0.bo"] = np.zeros(d, f)
    P["l0.w1"] = np.zeros((d, dm), f)
    P["l0.b1"] = np.zeros(dm, f)
    P["l0.w2"] = np.zeros((dm, d), f)
    P["l0.b2"] = np.zeros(d, f)
    uw = np.zeros((d, 2), f)
    uw[1, 1] = 1.0
    P["unembed.w"] = uw
    # nonzero mean is >= 1/8 in magnitude (t+1 <= 8); threshold at 1/16
    P["unembed.b"] = np.array([0.0, -1.0 / 16], f)
    return P