import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=128, causal=True)

def build_params():
    d, H, dh, dm = 32, 2, 16, 128
    P = {}
    P["embed"] = np.zeros((7, d), np.float32)
    P["embed"][1, 0] = 1.0          # '(' -> dim0
    P["embed"][2, 1] = 1.0          # ')' -> dim1
    P["pos_embed"] = np.zeros((8, d), np.float32)
    for i in range(2):
        for n in ("wq", "wk"):
            P[f"l{i}.{n}"] = np.zeros((H, d, dh), np.float32)
        for n in ("bq", "bk", "bv"):
            P[f"l{i}.{n}"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.wv"] = np.zeros((H, d, dh), np.float32)
        P[f"l{i}.wo"] = np.zeros((H, dh, d), np.float32)
        P[f"l{i}.bo"] = np.zeros(d, np.float32)
        P[f"l{i}.w1"] = np.zeros((d, dm), np.float32)
        P[f"l{i}.b1"] = np.zeros(dm, np.float32)
        P[f"l{i}.w2"] = np.zeros((dm, d), np.float32)
        P[f"l{i}.b2"] = np.zeros(d, np.float32)
    # Layer 0, head 0: uniform causal average of (is_open - is_close) -> dim2
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wv"][0, 1, 0] = -1.0
    P["l0.wo"][0, 0, 2] = 1.0
    # Layer 0 MLP: bad = clip(-8*o, 0, 1) -> dim3, where o = balance/(t+1)
    P["l0.w1"][2, 0] = -8.0
    P["l0.w1"][2, 1] = -8.0
    P["l0.b1"][1] = -1.0
    P["l0.w2"][0, 3] = 1.0
    P["l0.w2"][1, 3] = -1.0
    # Layer 1, head 0: uniform causal average of bad flags -> dim4
    P["l1.wv"][0, 3, 0] = 1.0
    P["l1.wo"][0, 0, 4] = 1.0
    # Unembed: class 0 = dead, class 1 = valid
    P["unembed.w"] = np.zeros((d, 2), np.float32)
    P["unembed.w"][4, 0] = 8.0
    P["unembed.b"] = np.array([-0.5, 0.0], np.float32)
    return P