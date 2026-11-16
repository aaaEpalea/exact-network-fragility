import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=2, d_head=1, d_mlp=1, causal=True)

def build_params():
    f = np.float32
    embed = np.zeros((7, 2), f)
    embed[1, 0] = 1.0    # token 1: +1
    embed[2, 0] = -1.0   # token 2: -1
    P = {}
    P["embed"] = embed
    P["pos_embed"] = np.zeros((8, 2), f)
    P["l0.wq"] = np.zeros((1, 2, 1), f)
    P["l0.wk"] = np.zeros((1, 2, 1), f)
    wv = np.zeros((1, 2, 1), f); wv[0, 0, 0] = 1.0
    P["l0.wv"] = wv
    P["l0.bq"] = np.zeros((1, 1), f)
    P["l0.bk"] = np.zeros((1, 1), f)
    P["l0.bv"] = np.zeros((1, 1), f)
    wo = np.zeros((1, 1, 2), f); wo[0, 0, 1] = 1.0
    P["l0.wo"] = wo
    P["l0.bo"] = np.zeros(2, f)
    P["l0.w1"] = np.zeros((2, 1), f)
    P["l0.b1"] = np.zeros(1, f)
    P["l0.w2"] = np.zeros((1, 2), f)
    P["l0.b2"] = np.zeros(2, f)
    uw = np.zeros((2, 2), f); uw[1, 1] = 1.0
    P["unembed.w"] = uw
    ub = np.zeros(2, f); ub[1] = -1.0 / 16
    P["unembed.b"] = ub
    return P