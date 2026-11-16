import numpy as np

CONFIG = dict(n_layers=2, n_heads=1, d_model=5, d_head=1, d_mlp=2, causal=True)

def build_params():
    d, H, dh, dm = 5, 1, 1, 2
    P = {}
    emb = np.zeros((7, d), np.float32)
    emb[1, 0] = 1.0    # '(' -> +1
    emb[2, 0] = -1.0   # ')' -> -1
    P["embed"] = emb
    P["pos_embed"] = np.zeros((8, d), np.float32)

    def layer(i):
        for n in ("wq", "wk", "wv"):
            P[f"l{i}.{n}"] = np.zeros((H, d, dh), np.float32)
        for n in ("bq", "bk", "bv"):
            P[f"l{i}.{n}"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.wo"] = np.zeros((H, dh, d), np.float32)
        P[f"l{i}.bo"] = np.zeros(d, np.float32)
        P[f"l{i}.w1"] = np.zeros((d, dm), np.float32)
        P[f"l{i}.b1"] = np.zeros(dm, np.float32)
        P[f"l{i}.w2"] = np.zeros((dm, d), np.float32)
        P[f"l{i}.b2"] = np.zeros(d, np.float32)

    # Layer 0: uniform causal attention averages the +1/-1 values over positions 0..t
    # (BOS counts as 0), giving a = balance/(t+1) in dim1.
    # MLP: dim2 = [balance < 0] = relu(-8a) - relu(-8a - 1).
    layer(0)
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 1] = 1.0
    P["l0.w1"][1, 0] = -8.0
    P["l0.w1"][1, 1] = -8.0
    P["l0.b1"][:] = [0.0, -1.0]
    P["l0.w2"][0, 2] = 1.0
    P["l0.w2"][1, 2] = -1.0

    # Layer 1: uniform causal attention averages the flag in dim2 over positions 0..t.
    # The result is >= 1/8 iff some prefix has gone negative, else exactly 0.
    # MLP: dim4 = min(1, 8*avg), the "dead" indicator.
    layer(1)
    P["l1.wv"][0, 2, 0] = 1.0
    P["l1.wo"][0, 0, 3] = 1.0
    P["l1.w1"][3, 0] = 8.0
    P["l1.w1"][3, 1] = 8.0
    P["l1.b1"][:] = [0.0, -1.0]
    P["l1.w2"][0, 4] = 1.0
    P["l1.w2"][1, 4] = -1.0

    # Output: class 0 (invalid) logit = dead indicator; class 1 (valid) logit = 0.5.
    uw = np.zeros((d, 2), np.float32)
    uw[4, 0] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.array([0.0, 0.5], np.float32)
    return P