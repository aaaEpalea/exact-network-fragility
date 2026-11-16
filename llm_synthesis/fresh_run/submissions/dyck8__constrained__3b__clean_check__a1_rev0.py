import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=128, causal=True)


def build_params():
    d, H, dh, dm = 32, 2, 16, 128
    f = np.float32
    P = {}
    P["embed"] = np.zeros((7, d), f)
    P["embed"][1, 0] = 1.0    # open bracket  -> +1
    P["embed"][2, 0] = -1.0   # close bracket -> -1
    P["pos_embed"] = np.zeros((8, d), f)
    for i in range(2):
        for n in ("wq", "wk", "wv"):
            P[f"l{i}.{n}"] = np.zeros((H, d, dh), f)
        for n in ("bq", "bk", "bv"):
            P[f"l{i}.{n}"] = np.zeros((H, dh), f)
        P[f"l{i}.wo"] = np.zeros((H, dh, d), f)
        P[f"l{i}.bo"] = np.zeros((d,), f)
        P[f"l{i}.w1"] = np.zeros((d, dm), f)
        P[f"l{i}.b1"] = np.zeros((dm,), f)
        P[f"l{i}.w2"] = np.zeros((dm, d), f)
        P[f"l{i}.b2"] = np.zeros((d,), f)
    # Layer 0: zero q/k -> uniform causal attention; value = +-1 bracket signal.
    # Output m = balance/(t+1) written to residual dim 1.
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 1] = 1.0
    # MLP 0: flag = relu(-8m) - relu(-8m-1) = 1 iff balance < 0  -> dim 2
    P["l0.w1"][1, 0] = -8.0
    P["l0.w1"][1, 1] = -8.0
    P["l0.b1"][1] = -1.0
    P["l0.w2"][0, 2] = 1.0
    P["l0.w2"][1, 2] = -1.0
    # Layer 1: uniform attention averaging the violation flags -> dim 3
    P["l1.wv"][0, 2, 0] = 1.0
    P["l1.wo"][0, 0, 3] = 1.0
    # Unembed: class0 (invalid) = mean flag, class1 (valid) = 1/16 constant
    P["unembed.w"] = np.zeros((d, 2), f)
    P["unembed.w"][3, 0] = 1.0
    P["unembed.b"] = np.array([0.0, 1.0 / 16.0], f)
    return P