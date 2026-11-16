import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=128, causal=True)

def build_params():
    d, H, dh, dm = 32, 2, 16, 128
    P = {}
    P["embed"] = np.zeros((7, d), np.float32)
    P["embed"][1, 0] = 1.0    # token 1 -> +1
    P["embed"][2, 0] = -1.0   # token 2 -> -1
    P["pos_embed"] = np.zeros((8, d), np.float32)
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
    # Layer 0, head 0: q=k=0 gives uniform causal attention; copy value dim 0 into residual dim 1
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 1] = 1.0
    # Unembed: logit0 = 0, logit1 = 16*avg - 1
    P["unembed.w"] = np.zeros((d, 2), np.float32)
    P["unembed.w"][1, 1] = 16.0
    P["unembed.b"] = np.array([0.0, -1.0], np.float32)
    return P