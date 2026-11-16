import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=3, d_head=1, d_mlp=1, causal=True)

def build_params():
    d, H, dh, dm = 3, 1, 1, 1
    f = lambda *s: np.zeros(s, dtype=np.float32)
    P = {}
    # dim0 of the embedding holds the signed vote: +1 for token 1, -1 for token 2, 0 otherwise (BOS = token 0 -> 0)
    P["embed"] = f(7, d)
    P["embed"][1, 0] = 1.0
    P["embed"][2, 0] = -1.0
    P["pos_embed"] = f(8, d)

    # Layer 0, one head: zero q/k => uniform causal attention; value = dim0, written to dim1
    P["l0.wq"] = f(H, d, dh); P["l0.bq"] = f(H, dh)
    P["l0.wk"] = f(H, d, dh); P["l0.bk"] = f(H, dh)
    P["l0.wv"] = f(H, d, dh); P["l0.wv"][0, 0, 0] = 1.0
    P["l0.bv"] = f(H, dh)
    P["l0.wo"] = f(H, dh, d); P["l0.wo"][0, 0, 1] = 1.0
    P["l0.bo"] = f(d)

    # MLP: h = relu(8*avg - 0.5), written to dim2
    P["l0.w1"] = f(d, dm); P["l0.w1"][1, 0] = 8.0
    P["l0.b1"] = f(dm);    P["l0.b1"][0] = -0.5
    P["l0.w2"] = f(dm, d); P["l0.w2"][0, 2] = 1.0
    P["l0.b2"] = f(d)

    # Output: logit0 = 0.25 (constant), logit1 = h
    P["unembed.w"] = f(d, 2); P["unembed.w"][2, 1] = 1.0
    P["unembed.b"] = np.array([0.25, 0.0], dtype=np.float32)
    return P