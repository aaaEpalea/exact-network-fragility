import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=3, d_head=1, d_mlp=1, causal=True)

def build_params():
    f = np.float32
    d, H, dh, dm = 3, 1, 1, 1
    P = {}
    emb = np.zeros((7, d), f)
    emb[1, 0] = 1.0    # token 1 -> value +1
    emb[2, 0] = -1.0   # token 2 -> value -1
    P["embed"] = emb
    P["pos_embed"] = np.zeros((8, d), f)
    P["l0.wq"] = np.zeros((H, d, dh), f)   # zero q,k -> uniform attention over prefix
    P["l0.wk"] = np.zeros((H, d, dh), f)
    P["l0.wv"] = np.zeros((H, d, dh), f)
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.bq"] = np.zeros((H, dh), f)
    P["l0.bk"] = np.zeros((H, dh), f)
    P["l0.bv"] = np.zeros((H, dh), f)
    P["l0.wo"] = np.zeros((H, dh, d), f)
    P["l0.wo"][0, 0, 1] = 1.0              # write the prefix average to dim 1
    P["l0.bo"] = np.zeros(d, f)
    P["l0.w1"] = np.zeros((d, dm), f)
    P["l0.w1"][1, 0] = 8.0
    P["l0.b1"] = np.full(dm, -0.5, f)
    P["l0.w2"] = np.zeros((dm, d), f)
    P["l0.w2"][0, 2] = 1.0                 # write the relu output to dim 2
    P["l0.b2"] = np.zeros(d, f)
    uw = np.zeros((d, 2), f)
    uw[2, 1] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.array([0.25, 0.0], f)
    return P