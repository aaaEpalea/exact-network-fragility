import numpy as np

CONFIG = dict(n_layers=3, n_heads=1, d_model=36, d_head=8, d_mlp=28, causal=False)

def build_params():
    d, H, dh, dm, L = 36, 1, 8, 28, 3
    P = {}
    # layout: 0-7 pos one-hot | 8-13 token one-hot (tok1..6) | 14 f=1/T
    #         15-21 one-hot of T=2..8 (T = L+1) | 22-29 one-hot of target position | 30-35 gathered token
    emb = np.zeros((7, d), np.float32)
    for t in range(1, 7):
        emb[t, 8 + t - 1] = 1
    pe = np.zeros((8, d), np.float32)
    for p in range(8):
        pe[p, p] = 1
    P["embed"], P["pos_embed"] = emb, pe
    for i in range(L):
        P[f"l{i}.wq"] = np.zeros((H, d, dh), np.float32)
        P[f"l{i}.wk"] = np.zeros((H, d, dh), np.float32)
        P[f"l{i}.wv"] = np.zeros((H, d, dh), np.float32)
        P[f"l{i}.bq"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.bk"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.bv"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.wo"] = np.zeros((H, dh, d), np.float32)
        P[f"l{i}.bo"] = np.zeros(d, np.float32)
        P[f"l{i}.w1"] = np.zeros((d, dm), np.float32)
        P[f"l{i}.b1"] = np.zeros(dm, np.float32)
        P[f"l{i}.w2"] = np.zeros((dm, d), np.float32)
        P[f"l{i}.b2"] = np.zeros(d, np.float32)

    # Layer 0 attention: uniform attention, value = BOS indicator -> f = 1/T at every position
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 14] = 1.0
    # Layer 0 MLP: triangular hat functions of f centred at 1/T -> one-hot of T
    pts = [0.0] + [1.0 / T for T in range(8, 1, -1)] + [1.0]   # ascending
    for T in range(2, 9):
        c = 1.0 / T
        idx = pts.index(c) if c in pts else int(np.argmin([abs(c - q) for q in pts]))
        clo, chi = pts[idx - 1], pts[idx + 1]
        s1, s2 = 1.0 / (c - clo), 1.0 / (chi - c)
        n = 3 * (T - 2)
        out = 15 + T - 2
        P["l0.w1"][14, n] = s1;  P["l0.b1"][n] = -s1 * clo;  P["l0.w2"][n, out] = 1.0
        P["l0.w1"][14, n + 1] = 1.0;  P["l0.b1"][n + 1] = -c;  P["l0.w2"][n + 1, out] = -(s1 + s2)
        P["l0.w1"][14, n + 2] = 1.0;  P["l0.b1"][n + 2] = -chi;  P["l0.w2"][n + 2, out] = s2

    # Layer 1 MLP: neuron per (T, i): fires iff length-code T and position i; outputs one-hot of j = T - i
    n = 0
    for T in range(2, 9):
        for i in range(1, T):
            P["l1.w1"][15 + T - 2, n] = 1.0
            P["l1.w1"][i, n] = 1.0
            P["l1.b1"][n] = -1.0
            P["l1.w2"][n, 22 + (T - i)] = 1.0
            n += 1

    # Layer 2 attention: query = target one-hot, key = position one-hot, value = token one-hot
    B = 60.0
    for m in range(8):
        P["l2.wq"][0, 22 + m, m] = B
        P["l2.wk"][0, m, m] = 1.0
    for k in range(6):
        P["l2.wv"][0, 8 + k, k] = 1.0
        P["l2.wo"][0, k, 30 + k] = 1.0

    uw = np.zeros((d, 7), np.float32)
    for k in range(6):
        uw[30 + k, k + 1] = 1.0
    P["unembed.w"], P["unembed.b"] = uw, np.zeros(7, np.float32)
    return {k: v.astype(np.float32) for k, v in P.items()}