import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=64, causal=False)

def build_params():
    d, H, dh, dm = 32, 2, 16, 64
    P = {}
    # residual layout: 0 BOS flag | 1..6 token one-hot | 7 pos p | 8 p^2 | 9 const 1
    #                  10 u=1/T   | 11 T                | 16..21 fetched token one-hot
    E = np.zeros((7, d))
    E[0, 0] = 1.0
    for t in range(1, 7):
        E[t, t] = 1.0
    PE = np.zeros((8, d))
    for p in range(8):
        PE[p, 7] = p
        PE[p, 8] = p * p
        PE[p, 9] = 1.0
    P["embed"] = E
    P["pos_embed"] = PE

    def zeros_layer(i):
        P[f"l{i}.wq"] = np.zeros((H, d, dh)); P[f"l{i}.wk"] = np.zeros((H, d, dh))
        P[f"l{i}.wv"] = np.zeros((H, d, dh)); P[f"l{i}.bq"] = np.zeros((H, dh))
        P[f"l{i}.bk"] = np.zeros((H, dh));   P[f"l{i}.bv"] = np.zeros((H, dh))
        P[f"l{i}.wo"] = np.zeros((H, dh, d)); P[f"l{i}.bo"] = np.zeros(d)
        P[f"l{i}.w1"] = np.zeros((d, dm));   P[f"l{i}.b1"] = np.zeros(dm)
        P[f"l{i}.w2"] = np.zeros((dm, d));   P[f"l{i}.b2"] = np.zeros(d)
    zeros_layer(0); zeros_layer(1)

    # ---- Layer 0: head 0 uniform attention, value = BOS flag -> dim 10 = 1/T
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 10] = 1.0
    # MLP0: piecewise-linear interpolation of T = 1/u through u=1/8..1/2 -> dim 11
    a = [1.0 / t for t in range(8, 1, -1)]      # ascending u: 1/8 ... 1/2
    Tv = [float(t) for t in range(8, 1, -1)]    # 8 ... 2
    m = [(Tv[k] - Tv[k - 1]) / (a[k] - a[k - 1]) for k in range(1, 7)]
    coef = [m[0]] + [m[k] - m[k - 1] for k in range(1, 6)]
    for k in range(6):
        P["l0.w1"][10, k] = 1.0
        P["l0.b1"][k] = -a[k]
        P["l0.w2"][k, 11] = coef[k]
    P["l0.b2"][11] = 8.0

    # ---- Layer 1: head 0 attends to position j = T - i
    A = 240.0   # score = A*(tg*j - j^2/2)/4 -> gap A/8 = 30 between target and neighbors
    P["l1.wq"][0, 11, 0] = A
    P["l1.wq"][0, 7, 0] = -A
    P["l1.wq"][0, 9, 1] = -A / 2.0
    P["l1.wk"][0, 7, 0] = 1.0
    P["l1.wk"][0, 8, 1] = 1.0
    for t in range(1, 7):
        P["l1.wv"][0, t, t] = 1.0
        P["l1.wo"][0, t, 15 + t] = 1.0

    # ---- Unembed: dims 16..21 -> logits for tokens 1..6
    W = np.zeros((d, 7))
    for t in range(1, 7):
        W[15 + t, t] = 1.0
    P["unembed.w"] = W
    P["unembed.b"] = np.zeros(7)

    return {k: np.asarray(v, dtype=np.float32) for k, v in P.items()}