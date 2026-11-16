import numpy as np

CONFIG = dict(n_layers=3, n_heads=1, d_model=38, d_head=8, d_mlp=28, causal=False)

def build_params():
    d, H, dh, dm, L = 38, 1, 8, 28, 3
    P = {}
    # residual layout: 0..6 token one-hot | 7..14 position one-hot | 15: 1/T
    # 16..22: one-hot of T=L+1 (T=2..8) | 23..30: one-hot of target position j*=T-i
    # 31..37: gathered token one-hot
    emb = np.zeros((7, d), np.float32)
    for t in range(7): emb[t, t] = 1
    pos = np.zeros((8, d), np.float32)
    for p in range(8): pos[p, 7 + p] = 1
    P["embed"], P["pos_embed"] = emb, pos
    for i in range(L):
        for n, s in [("wq", (H, d, dh)), ("wk", (H, d, dh)), ("wv", (H, d, dh)),
                     ("bq", (H, dh)), ("bk", (H, dh)), ("bv", (H, dh)),
                     ("wo", (H, dh, d)), ("bo", (d,)),
                     ("w1", (d, dm)), ("b1", (dm,)), ("w2", (dm, d)), ("b2", (d,))]:
            P[f"l{i}.{n}"] = np.zeros(s, np.float32)

    # Layer 0 attention: uniform attention (q=k=0) averaging the BOS indicator -> 1/T in dim 15
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 15] = 1.0
    # Layer 0 MLP: triangular "hat" bumps around f=1/T converting 1/T into one-hot of T
    delta = 1.0 / 112.0
    for T in range(2, 9):
        u = 1.0 / T
        for k, (off, c) in enumerate([(-delta, 1.0), (0.0, -2.0), (delta, 1.0)]):
            h = 3 * (T - 2) + k
            P["l0.w1"][15, h] = 1.0
            P["l0.b1"][h] = -(u + off)
            P["l0.w2"][h, 16 + T - 2] = c / delta

    # Layer 1 MLP: AND(position i, T) -> one-hot of j* = T - i
    h = 0
    for T in range(2, 9):
        for i in range(1, T):
            P["l1.w1"][7 + i, h] = 1.0
            P["l1.w1"][16 + T - 2, h] = 1.0
            P["l1.b1"][h] = -1.0
            P["l1.w2"][h, 23 + (T - i)] = 1.0
            h += 1

    # Layer 2 attention: query = onehot(j*), key = onehot(position), value = token one-hot
    S = 40.0 * np.sqrt(dh)
    for j in range(8):
        P["l2.wq"][0, 23 + j, j] = S
        P["l2.wk"][0, 7 + j, j] = 1.0
    for t in range(7):
        P["l2.wv"][0, t, t] = 1.0
        P["l2.wo"][0, t, 31 + t] = 1.0

    uw = np.zeros((d, 7), np.float32)
    for t in range(7): uw[31 + t, t] = 1.0
    P["unembed.w"], P["unembed.b"] = uw, np.zeros(7, np.float32)
    return P