import numpy as np

CONFIG = dict(n_layers=2, n_heads=1, d_model=4, d_head=1, d_mlp=2, causal=True)

def build_params():
    d, dh, H, dm = 4, 1, 1, 2
    z = lambda *s: np.zeros(s, np.float32)
    P = {}
    P["embed"] = z(7, d)
    P["embed"][1, 0] = -1.0   # open  -> -1
    P["embed"][2, 0] = 1.0    # close -> +1
    P["pos_embed"] = z(8, d)

    # Layer 0: uniform causal attention averages (closes - opens) -> dim1 ; MLP -> dim2
    P["l0.wq"] = z(H, d, dh); P["l0.wk"] = z(H, d, dh); P["l0.wv"] = z(H, d, dh)
    P["l0.bq"] = z(H, dh); P["l0.bk"] = z(H, dh); P["l0.bv"] = z(H, dh)
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"] = z(H, dh, d); P["l0.wo"][0, 0, 1] = 1.0
    P["l0.bo"] = z(d)
    P["l0.w1"] = z(d, dm); P["l0.b1"] = z(dm)
    P["l0.w1"][1, 0] = 8.0
    P["l0.w1"][1, 1] = 8.0; P["l0.b1"][1] = -1.0
    P["l0.w2"] = z(dm, d)
    P["l0.w2"][0, 2] = 1.0; P["l0.w2"][1, 2] = -1.0
    P["l0.b2"] = z(d)

    # Layer 1: uniform causal attention averages the "bad" indicator (dim2) -> dim3
    P["l1.wq"] = z(H, d, dh); P["l1.wk"] = z(H, d, dh); P["l1.wv"] = z(H, d, dh)
    P["l1.bq"] = z(H, dh); P["l1.bk"] = z(H, dh); P["l1.bv"] = z(H, dh)
    P["l1.wv"][0, 2, 0] = 1.0
    P["l1.wo"] = z(H, dh, d); P["l1.wo"][0, 0, 3] = 1.0
    P["l1.bo"] = z(d)
    P["l1.w1"] = z(d, dm); P["l1.b1"] = z(dm)
    P["l1.w2"] = z(dm, d); P["l1.b2"] = z(d)

    P["unembed.w"] = z(d, 2); P["unembed.b"] = z(2)
    P["unembed.w"][3, 1] = -16.0
    P["unembed.b"][1] = 1.0
    return P