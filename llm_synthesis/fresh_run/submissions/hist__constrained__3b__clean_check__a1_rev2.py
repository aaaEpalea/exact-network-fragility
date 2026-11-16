import numpy as np
CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=128, causal=True)

def build_params():
    L, H, d, dh, dm = 2, 2, 32, 16, 128
    f = np.float32
    P = {"embed": np.zeros((7, d), f), "pos_embed": np.zeros((8, d), f)}
    for t in range(1, 7):
        P["embed"][t, t - 1] = 1.0
    P["embed"][0, 6] = 1.0
    for i in range(L):
        P[f"l{i}.wq"] = np.zeros((H, d, dh), f); P[f"l{i}.wk"] = np.zeros((H, d, dh), f)
        P[f"l{i}.wv"] = np.zeros((H, d, dh), f); P[f"l{i}.bq"] = np.zeros((H, dh), f)
        P[f"l{i}.bk"] = np.zeros((H, dh), f); P[f"l{i}.bv"] = np.zeros((H, dh), f)
        P[f"l{i}.wo"] = np.zeros((H, dh, d), f); P[f"l{i}.bo"] = np.zeros(d, f)
        P[f"l{i}.w1"] = np.zeros((d, dm), f); P[f"l{i}.b1"] = np.zeros(dm, f)
        P[f"l{i}.w2"] = np.zeros((dm, d), f); P[f"l{i}.b2"] = np.zeros(d, f)
    # layer 0, head 0: attend to same token (and BOS); output ~ 1/(n+1) in dim 7
    s = 12.0 * np.sqrt(dh)
    for t in range(6):
        P["l0.wq"][0, t, t] = s
        P["l0.wk"][0, t, t] = 1.0
    P["l0.wk"][0, 6, 6] = 1.0
    P["l0.bq"][0, 6] = s
    P["l0.wv"][0, 6, 0] = 1.0
    P["l0.wo"][0, 0, 7] = 1.0
    # layer 0 MLP: map 1/(n+1) -> n into dim 8
    g = [1.0 / (8 - k) for k in range(7)]
    sl = [-1.0 / (g[k + 1] - g[k]) for k in range(6)]
    for k in range(6):
        P["l0.w1"][7, k] = 1.0
        P["l0.b1"][k] = -g[k]
        P["l0.w2"][k, 8] = sl[k] - (sl[k - 1] if k > 0 else 0.0)
    # layer 1: identity (all zeros)
    P["unembed.w"] = np.zeros((d, 2), f); P["unembed.w"][8, 0] = 1.0   # col 0 = count, col 1 unused
    P["unembed.b"] = np.array([7.0, 0.0], f)
    return P