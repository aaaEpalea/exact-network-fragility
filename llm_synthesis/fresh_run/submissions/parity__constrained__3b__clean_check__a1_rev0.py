import numpy as np
CONFIG = dict(n_layers=2, n_heads=2, d_model=16, d_head=8, d_mlp=64, causal=True)

def build_params():
    d, H, dh, dm = 16, 2, 8, 64
    f = np.float32
    P = {}
    P["embed"] = np.zeros((7, d), f)
    P["embed"][1, 8] = 1.0                      # indicator "token == 1"
    P["pos_embed"] = np.zeros((8, d), f)
    for p in range(8):
        P["pos_embed"][p, p] = 1.0              # one-hot position
    for i in range(2):
        P[f"l{i}.wq"] = np.zeros((H, d, dh), f); P[f"l{i}.wk"] = np.zeros((H, d, dh), f)
        P[f"l{i}.wv"] = np.zeros((H, d, dh), f); P[f"l{i}.bq"] = np.zeros((H, dh), f)
        P[f"l{i}.bk"] = np.zeros((H, dh), f);    P[f"l{i}.bv"] = np.zeros((H, dh), f)
        P[f"l{i}.wo"] = np.zeros((H, dh, d), f); P[f"l{i}.bo"] = np.zeros(d, f)
        P[f"l{i}.w1"] = np.zeros((d, dm), f);    P[f"l{i}.b1"] = np.zeros(dm, f)
        P[f"l{i}.w2"] = np.zeros((dm, d), f);    P[f"l{i}.b2"] = np.zeros(d, f)
    # layer 0, head 0: uniform causal attention averaging the indicator -> dim 9 = c/(t+1)
    P["l0.wv"][0, 8, 0] = 1.0
    P["l0.wo"][0, 0, 9] = 1.0
    # layer 0 MLP: per-position zigzag piecewise-linear interpolation of parity(c) over a=c/(t+1)
    M = 100.0
    u = 0
    for t in range(1, 8):
        prev = 0.0
        for c in range(t):
            m = (t + 1) * (1.0 if c % 2 == 0 else -1.0)
            P["l0.w1"][9, u] = 1.0
            P["l0.w1"][t, u] = M
            P["l0.b1"][u] = -c / (t + 1.0) - M
            P["l0.w2"][u, 10] = m - prev
            prev = m
            u += 1
    P["unembed.w"] = np.zeros((d, 2), f)
    P["unembed.w"][10, 0] = -1.0; P["unembed.w"][10, 1] = 1.0
    P["unembed.b"] = np.array([0.5, -0.5], f)
    return P