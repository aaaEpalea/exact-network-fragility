import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=12, d_head=1, d_mlp=56, causal=True)

def build_params():
    d, H, dh, dm = 12, 1, 1, 56
    P = {}
    # embeddings: dim0 = [token==1]; dims 1..8 = one-hot of position
    embed = np.zeros((7, d), np.float32)
    embed[1, 0] = 1.0
    pos = np.zeros((8, d), np.float32)
    for p in range(8):
        pos[p, 1 + p] = 1.0
    P["embed"], P["pos_embed"] = embed, pos

    # attention: zero q/k -> uniform causal averaging over positions 0..t (BOS included)
    P["l0.wq"] = np.zeros((H, d, dh), np.float32)
    P["l0.wk"] = np.zeros((H, d, dh), np.float32)
    P["l0.bq"] = np.zeros((H, dh), np.float32)
    P["l0.bk"] = np.zeros((H, dh), np.float32)
    wv = np.zeros((H, d, dh), np.float32); wv[0, 0, 0] = 1.0   # value = [token==1]
    P["l0.wv"] = wv
    P["l0.bv"] = np.zeros((H, dh), np.float32)
    wo = np.zeros((H, dh, d), np.float32); wo[0, 0, 9] = 1.0   # write f = c/(t+1) into dim 9
    P["l0.wo"] = wo
    P["l0.bo"] = np.zeros(d, np.float32)

    # MLP: for each position T (1..7) and k (0..7): unit = relu((T+1)f - k - M(1 - [pos==T]))
    M = 20.0
    w1 = np.zeros((d, dm), np.float32)
    b1 = np.zeros(dm, np.float32)
    w2 = np.zeros((dm, d), np.float32)
    for T in range(1, 8):
        for k in range(8):
            u = (T - 1) * 8 + k
            w1[9, u] = T + 1          # (T+1)*f = c when pos == T
            w1[1 + T, u] = M          # gate on position one-hot
            b1[u] = -k - M
            w2[u, 10] = 1.0 if k == 0 else 2.0 * (-1) ** k   # parity(c) = relu(c) - 2relu(c-1) + 2relu(c-2) ...
    P["l0.w1"], P["l0.b1"], P["l0.w2"] = w1, b1, w2
    P["l0.b2"] = np.zeros(d, np.float32)

    # unembed: logit1 = p, logit0 = 1 - p  (p = parity in dim 10)
    uw = np.zeros((d, 2), np.float32); uw[10, 1] = 1.0; uw[10, 0] = -1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.array([1.0, 0.0], np.float32)
    return P