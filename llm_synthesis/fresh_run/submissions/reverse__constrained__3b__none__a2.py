import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=64, causal=False)

def build_params():
    d, H, dh, dm = 32, 2, 16, 64
    P = {}
    # residual layout: 0-6 token one-hot (dim 0 = BOS flag), 7 = pos i, 8 = pos^2,
    # 10 = 1/n (n = L+1), 11 = n, 12-18 = attended token copy (output region)
    embed = np.zeros((7, d))
    for t in range(7):
        embed[t, t] = 1.0
    pos = np.zeros((8, d))
    for p in range(8):
        pos[p, 7] = p
        pos[p, 8] = p * p
    P["embed"] = embed
    P["pos_embed"] = pos

    def z(name, shape):
        P[name] = np.zeros(shape)

    # ---- layer 0: uniform attention averages the BOS flag -> 1/n; MLP maps 1/n -> n
    for n in ["wq", "wk"]:
        z(f"l0.{n}", (H, d, dh))
    for n in ["bq", "bk", "bv"]:
        z(f"l0.{n}", (H, dh))
    z("l0.wv", (H, d, dh)); z("l0.wo", (H, dh, d)); z("l0.bo", (d,))
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 10] = 1.0

    ns = [8, 7, 6, 5, 4, 3, 2]
    us = [1.0 / n for n in ns]
    fs = [float(n) for n in ns]
    slopes = [(fs[m + 1] - fs[m]) / (us[m + 1] - us[m]) for m in range(6)]
    z("l0.w1", (d, dm)); z("l0.b1", (dm,)); z("l0.w2", (dm, d)); z("l0.b2", (d,))
    for k in range(6):
        P["l0.w1"][10, k] = 1.0
        P["l0.b1"][k] = -us[k]
        coef = slopes[0] if k == 0 else slopes[k] - slopes[k - 1]
        P["l0.w2"][k, 11] = coef
    P["l0.b2"][11] = fs[0]  # f(u) = 8 + sum of hinges

    # ---- layer 1: head 0 attends to position j = n - i, copies its token one-hot
    C = 10.0
    for n in ["wq", "wk", "wv"]:
        z(f"l1.{n}", (H, d, dh))
    for n in ["bq", "bk", "bv"]:
        z(f"l1.{n}", (H, dh))
    z("l1.wo", (H, dh, d)); z("l1.bo", (d,))
    P["l1.wq"][0, 7, 0] = 1.0      # q0 = i
    P["l1.wq"][0, 11, 1] = 1.0     # q1 = n
    P["l1.bq"][0, 2] = 1.0         # q2 = 1
    P["l1.wk"][0, 7, 0] = -8 * C   # k0 = -8C j
    P["l1.wk"][0, 7, 1] = 8 * C    # k1 = 8C j
    P["l1.wk"][0, 8, 2] = -4 * C   # k2 = -4C j^2
    # score = (q.k)/4 = -2C i j + 2C n j - C j^2  == -C (i+j-n)^2 + (terms in i, n only)
    for c in range(7):
        P["l1.wv"][0, c, c] = 1.0
        P["l1.wo"][0, c, 12 + c] = 1.0
    z("l1.w1", (d, dm)); z("l1.b1", (dm,)); z("l1.w2", (dm, d)); z("l1.b2", (d,))

    uw = np.zeros((d, 7)); 
    for c in range(7):
        uw[12 + c, c] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.zeros(7)
    return {k: np.asarray(v, dtype=np.float32) for k, v in P.items()}