import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=16, d_head=8, d_mlp=64, causal=True)

def build_params():
    d, H, dh, dm = 16, 2, 8, 64
    P = {}
    P["embed"] = np.zeros((7, d), np.float32)
    P["pos_embed"] = np.zeros((8, d), np.float32)
    P["embed"][1, 8] = 1.0                      # indicator: token == 1
    for p in range(8):
        P["pos_embed"][p, p] = 1.0              # one-hot position in dims 0..7
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
    # Layer 0, head 0: uniform causal attention (all scores 0) averaging the "is 1" indicator.
    P["l0.wv"][0, 8, 0] = 1.0
    P["l0.wo"][0, 0, 10] = 1.0                  # f = c/(t+1) into dim 10
    # Layer 0 MLP: parity from f, gated by one-hot position t.
    S, G = 64.0, 100.0
    u = 0
    for t in range(1, 8):
        for k in range(1, t + 1):
            thr = (k - 0.5) / (t + 1)
            sign = 1.0 if k % 2 == 1 else -1.0
            for off, s in ((0.5, sign), (-0.5, -sign)):
                P["l0.w1"][10, u] = S
                P["l0.w1"][t, u] = G
                P["l0.b1"][u] = -S * thr + off - G
                P["l0.w2"][u, 11] = s
                u += 1
    # Layer 1 is all zeros (identity).
    P["unembed.w"] = np.zeros((d, 2), np.float32)
    P["unembed.b"] = np.zeros((2,), np.float32)
    P["unembed.w"][11, 1] = 1.0
    P["unembed.b"][0] = 0.5                     # class1 iff parity == 1
    return P