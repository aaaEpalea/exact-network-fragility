import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=64, causal=False)

def build_params():
    d, H, dh, dm = 32, 2, 16, 64
    P = {}
    f = lambda *s: np.zeros(s, np.float32)

    # residual layout: 0..6 token one-hot (0 = BOS), 7 = pos i, 8 = pos^2, 9 = const 1,
    # 10 = a = 1/(L+1), 11 = c = L+1, 12..18 = attended token one-hot
    embed = f(7, d)
    for t in range(7):
        embed[t, t] = 1.0
    pos = f(8, d)
    for p in range(8):
        pos[p, 7] = p
        pos[p, 8] = p * p
        pos[p, 9] = 1.0
    P["embed"] = embed
    P["pos_embed"] = pos

    for i in range(2):
        for n, s in [("wq", (H, d, dh)), ("wk", (H, d, dh)), ("wv", (H, d, dh)),
                     ("bq", (H, dh)), ("bk", (H, dh)), ("bv", (H, dh)),
                     ("wo", (H, dh, d)), ("bo", (d,)),
                     ("w1", (d, dm)), ("b1", (dm,)), ("w2", (dm, d)), ("b2", (d,))]:
            P[f"l{i}.{n}"] = f(*s)

    # ---- layer 0: uniform attention averages the BOS indicator -> a = 1/(L+1)
    P["l0.wv"][0, 0, 0] = 1.0          # head 0 value = BOS indicator
    P["l0.wo"][0, 0, 10] = 1.0         # write a into dim 10
    # MLP: piecewise-linear interpolation of c = 1/a through (1/n, n), n = 2..8
    s_prev = 0.0
    for k in range(6):
        s_k = -float((7 - k) * (8 - k))
        P["l0.w1"][10, k] = 1.0
        P["l0.b1"][k] = -1.0 / (8 - k)
        P["l0.w2"][k, 11] = s_k - s_prev
        s_prev = s_k
    P["l0.b2"][11] = 8.0               # c = 8 at a = 1/8

    # ---- layer 1: head 0 attends from position i to j = c - i
    K = 10.0
    P["l1.wq"][0, 9, 0] = 4.0          # q0 = 4
    P["l1.wq"][0, 7, 1] = 4.0          # q1 = 4 (i - c)
    P["l1.wq"][0, 11, 1] = -4.0
    P["l1.wk"][0, 8, 0] = -K           # k0 = -K j^2
    P["l1.wk"][0, 7, 1] = -2 * K       # k1 = -2K j
    # score = -K j^2 - 2K j (i - c) = -K (i + j - c)^2 + (terms constant in j)
    for m in range(7):
        P["l1.wv"][0, m, m] = 1.0      # value = token one-hot
        P["l1.wo"][0, m, 12 + m] = 1.0
    # head 1 and layer-1 MLP stay zero

    uw = f(d, 7)
    for t in range(7):
        uw[12 + t, t] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = f(7)
    return P