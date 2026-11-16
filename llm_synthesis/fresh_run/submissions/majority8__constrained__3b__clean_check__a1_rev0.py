import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=128, causal=True)


def build_params():
    d, H, dh, dm, V, P = 32, 2, 16, 128, 7, 8
    f = np.float32
    p = {}
    p["embed"] = np.zeros((V, d), f)
    p["pos_embed"] = np.zeros((P, d), f)
    # channel 0 holds the signed token indicator: +1 for token 1, -1 for token 2
    p["embed"][1, 0] = 1.0
    p["embed"][2, 0] = -1.0
    for i in range(2):
        p[f"l{i}.wq"] = np.zeros((H, d, dh), f)
        p[f"l{i}.wk"] = np.zeros((H, d, dh), f)
        p[f"l{i}.wv"] = np.zeros((H, d, dh), f)
        p[f"l{i}.bq"] = np.zeros((H, dh), f)
        p[f"l{i}.bk"] = np.zeros((H, dh), f)
        p[f"l{i}.bv"] = np.zeros((H, dh), f)
        p[f"l{i}.wo"] = np.zeros((H, dh, d), f)
        p[f"l{i}.bo"] = np.zeros((d,), f)
        p[f"l{i}.w1"] = np.zeros((d, dm), f)
        p[f"l{i}.b1"] = np.zeros((dm,), f)
        p[f"l{i}.w2"] = np.zeros((dm, d), f)
        p[f"l{i}.b2"] = np.zeros((d,), f)
    # layer 0, head 0: uniform causal attention (all scores 0) averaging channel 0
    p["l0.wv"][0, 0, 0] = 1.0
    p["l0.wo"][0, 0, 1] = 1.0          # mean = (#1 - #2)/(t+1) -> channel 1
    # layer 0 MLP: relu(8*mean - 0.5) -> channel 2
    p["l0.w1"][1, 0] = 8.0
    p["l0.b1"][0] = -0.5
    p["l0.w2"][0, 2] = 1.0
    # layer 1 is a no-op (all zeros)
    p["unembed.w"] = np.zeros((d, 2), f)
    p["unembed.b"] = np.zeros((2,), f)
    p["unembed.w"][2, 1] = 1.0
    p["unembed.b"][0] = 0.25
    return p