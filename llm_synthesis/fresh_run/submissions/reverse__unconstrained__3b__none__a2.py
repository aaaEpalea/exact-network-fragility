import numpy as np

CONFIG = dict(n_layers=3, n_heads=1, d_model=38, d_head=8, d_mlp=28, causal=False)

def build_params():
    d, dh, H, dm = 38, 8, 1, 28
    P = {}
    z = lambda *s: np.zeros(s, np.float32)

    # token one-hot -> dims 0..6 ; position one-hot -> dims 7..14
    emb = z(7, d)
    for t in range(7): emb[t, t] = 1.0
    pe = z(8, d)
    for p in range(8): pe[p, 7 + p] = 1.0
    P["embed"], P["pos_embed"] = emb, pe

    for i in range(3):
        for n in ["wq", "wk", "wv"]: P[f"l{i}.{n}"] = z(H, d, dh)
        for n in ["bq", "bk", "bv"]: P[f"l{i}.{n}"] = z(H, dh)
        P[f"l{i}.wo"] = z(H, dh, d); P[f"l{i}.bo"] = z(d)
        P[f"l{i}.w1"] = z(d, dm); P[f"l{i}.b1"] = z(dm)
        P[f"l{i}.w2"] = z(dm, d); P[f"l{i}.b2"] = z(d)

    # Layer 0 attention: uniform attention (q=k=0); value = 1000*[token==BOS]
    # -> dim 15 gets u = 1000/T at every position.
    P["l0.wv"][0, 0, 0] = 1000.0
    P["l0.wo"][0, 0, 15] = 1.0
    # Layer 0 MLP: tent functions around u = 1000/T, giving one-hot(T) in dims 16..22
    delta = 10.0
    n = 0
    for T in range(2, 9):
        c = 1000.0 / T
        for kk, coef in ((1, 1.0), (0, -2.0), (-1, 1.0)):
            P["l0.w1"][15, n] = 1.0 / delta
            P["l0.b1"][n] = -c / delta + kk
            P["l0.w2"][n, 16 + T - 2] = coef
            n += 1

    # Layer 1 MLP: AND(T, pos i) -> one-hot of m = T-i in dims 23+m
    # (layer 1 attention stays zero)
    n = 0
    for T in range(2, 9):
        for i in range(1, T):
            P["l1.w1"][16 + T - 2, n] = 1.0
            P["l1.w1"][7 + i, n] = 1.0
            P["l1.b1"][n] = -1.0
            P["l1.w2"][n, 23 + (T - i)] = 1.0
            n += 1

    # Layer 2 attention: query = S*onehot(m), key = onehot(pos), value = token one-hot
    S = 100.0
    for m in range(8):
        P["l2.wq"][0, 23 + m, m] = S
    for p in range(8):
        P["l2.wk"][0, 7 + p, p] = 1.0
    for t in range(7):
        P["l2.wv"][0, t, t] = 1.0
        P["l2.wo"][0, t, 31 + t] = 1.0

    uw = z(d, 7)
    for t in range(7): uw[31 + t, t] = 1.0
    P["unembed.w"], P["unembed.b"] = uw, z(7)
    return P