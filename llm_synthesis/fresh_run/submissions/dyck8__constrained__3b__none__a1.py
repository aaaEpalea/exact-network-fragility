import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=128, causal=True)

def build_params():
    d, H, dh, dm = 32, 2, 16, 128
    P = {}
    P["embed"] = np.zeros((7, d), np.float32)
    P["pos_embed"] = np.zeros((8, d), np.float32)
    P["embed"][1, 0] = 1.0    # open bracket  -> +1
    P["embed"][2, 0] = -1.0   # close bracket -> -1
    for i in range(2):
        P[f"l{i}.wq"] = np.zeros((H, d, dh), np.float32)
        P[f"l{i}.wk"] = np.zeros((H, d, dh), np.float32)
        P[f"l{i}.wv"] = np.zeros((H, d, dh), np.float32)
        P[f"l{i}.bq"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.bk"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.bv"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.wo"] = np.zeros((H, dh, d), np.float32)
        P[f"l{i}.bo"] = np.zeros((d,), np.float32)
        P[f"l{i}.w1"] = np.zeros((d, dm), np.float32)
        P[f"l{i}.b1"] = np.zeros((dm,), np.float32)
        P[f"l{i}.w2"] = np.zeros((dm, d), np.float32)
        P[f"l{i}.b2"] = np.zeros((d,), np.float32)
    # Layer 0: uniform causal attention (q=k=0) averages dim0 (+1/-1/0) -> dim1
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 1] = 1.0
    # Layer 0 MLP: flag = min(1, relu(-8*avg)) = relu(-8a) - relu(-8a-1) -> dim2
    P["l0.w1"][1, 0] = -8.0
    P["l0.w1"][1, 1] = -8.0
    P["l0.b1"][1] = -1.0
    P["l0.w2"][0, 2] = 1.0
    P["l0.w2"][1, 2] = -1.0
    # Layer 1: uniform causal attention averages the "bad" flag (dim2) -> dim3
    P["l1.wv"][0, 2, 0] = 1.0
    P["l1.wo"][0, 0, 3] = 1.0
    # Unembed: logit1 = 1 (valid), logit0 = 16*avg_bad
    P["unembed.w"] = np.zeros((d, 2), np.float32)
    P["unembed.b"] = np.array([0.0, 1.0], np.float32)
    P["unembed.w"][3, 0] = 16.0
    return P