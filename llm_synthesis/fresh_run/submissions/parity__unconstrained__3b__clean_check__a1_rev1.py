import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=128, causal=True)

def build_params():
    d, H, dh, dm = 32, 2, 16, 128
    P = {}
    f32 = np.float32

    # embeddings: dim0 = [token==1]; dims 1..8 = one-hot of position 0..7
    embed = np.zeros((7, d), f32)
    embed[1, 0] = 1.0
    pos = np.zeros((8, d), f32)
    for p in range(8):
        pos[p, 1 + p] = 1.0
    P["embed"], P["pos_embed"] = embed, pos

    # all layer tensors start at zero (layer 1 stays entirely zero = identity)
    for i in range(2):
        P[f"l{i}.wq"] = np.zeros((H, d, dh), f32)
        P[f"l{i}.wk"] = np.zeros((H, d, dh), f32)
        P[f"l{i}.wv"] = np.zeros((H, d, dh), f32)
        P[f"l{i}.bq"] = np.zeros((H, dh), f32)
        P[f"l{i}.bk"] = np.zeros((H, dh), f32)
        P[f"l{i}.bv"] = np.zeros((H, dh), f32)
        P[f"l{i}.wo"] = np.zeros((H, dh, d), f32)
        P[f"l{i}.bo"] = np.zeros(d, f32)
        P[f"l{i}.w1"] = np.zeros((d, dm), f32)
        P[f"l{i}.b1"] = np.zeros(dm, f32)
        P[f"l{i}.w2"] = np.zeros((dm, d), f32)
        P[f"l{i}.b2"] = np.zeros(d, f32)

    # layer 0, head 0: zero q/k -> uniform causal average over positions 0..t
    P["l0.wv"][0, 0, 0] = 1.0     # value = [token==1]
    P["l0.wo"][0, 0, 9] = 1.0     # f = c/(t+1) written to dim 9

    # layer 0 MLP: unit (T,k) = relu((T+1) f - k - M(1 - [pos==T]))
    M = 20.0
    w1, b1, w2 = P["l0.w1"], P["l0.b1"], P["l0.w2"]
    for T in range(1, 8):
        for k in range(8):
            u = (T - 1) * 8 + k
            w1[9, u] = T + 1
            w1[1 + T, u] = M
            b1[u] = -k - M
            w2[u, 10] = 1.0 if k == 0 else 2.0 * (-1) ** k  # zigzag = c mod 2

    # unembed: logit1 = p, logit0 = 1 - p
    uw = np.zeros((d, 2), f32)
    uw[10, 1] = 1.0
    uw[10, 0] = -1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.array([1.0, 0.0], f32)
    return P