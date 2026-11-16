import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=128, causal=True)

def build_params():
    d, H, dh, dm = 32, 2, 16, 128
    P = {}
    f = np.float32
    emb = np.zeros((7, d), f)
    emb[1, 0] = 1.0    # token 1 -> +1
    emb[2, 0] = -1.0   # token 2 -> -1
    P["embed"] = emb
    P["pos_embed"] = np.zeros((8, d), f)
    for i in range(2):
        P[f"l{i}.wq"] = np.zeros((H, d, dh), f)
        P[f"l{i}.wk"] = np.zeros((H, d, dh), f)
        P[f"l{i}.wv"] = np.zeros((H, d, dh), f)
        P[f"l{i}.bq"] = np.zeros((H, dh), f)
        P[f"l{i}.bk"] = np.zeros((H, dh), f)
        P[f"l{i}.bv"] = np.zeros((H, dh), f)
        P[f"l{i}.wo"] = np.zeros((H, dh, d), f)
        P[f"l{i}.bo"] = np.zeros((d,), f)
        P[f"l{i}.w1"] = np.zeros((d, dm), f)
        P[f"l{i}.b1"] = np.zeros((dm,), f)
        P[f"l{i}.w2"] = np.zeros((dm, d), f)
        P[f"l{i}.b2"] = np.zeros((d,), f)
    # Layer 0, head 0: uniform attention (q=k=0) averaging the +1/-1 value over positions 0..t
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 1] = 1.0          # u = (c1 - c2)/(t+1) written to dim 1
    # Layer 0 MLP: g = relu(16u) - relu(16u - 1)  (0 if u<=0, 1 if u>=1/16) -> dim 2
    P["l0.w1"][1, 0] = 16.0
    P["l0.w1"][1, 1] = 16.0
    P["l0.b1"][1] = -1.0
    P["l0.w2"][0, 2] = 1.0
    P["l0.w2"][1, 2] = -1.0
    # Layer 1 is left as a no-op (all zeros).
    uw = np.zeros((d, 2), f)
    uw[2, 1] = 1.0
    ub = np.zeros((2,), f)
    ub[1] = -0.5
    P["unembed.w"] = uw
    P["unembed.b"] = ub
    return P