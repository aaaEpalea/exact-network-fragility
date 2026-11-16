import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=9, d_head=7, d_mlp=6, causal=False)

def build_params():
    d, dh, dm, H = 9, 7, 6, 1
    P = {}
    # token one-hot in dims 0..6 (dim 0 = BOS); dim 7 = attention result a; dim 8 = output
    emb = np.zeros((7, d), np.float32)
    for t in range(7):
        emb[t, t] = 1.0
    P["embed"] = emb
    P["pos_embed"] = np.zeros((8, d), np.float32)

    A = 30.0 * np.sqrt(dh)          # so that (q.k)/sqrt(dh) = 30
    wq = np.zeros((H, d, dh), np.float32)
    wk = np.zeros((H, d, dh), np.float32)
    wv = np.zeros((H, d, dh), np.float32)
    for i in range(7):
        wq[0, i, i] = A
        wk[0, i, i] = 1.0
    wv[0, 0, 0] = 1.0               # value = BOS indicator
    bq = np.zeros((H, dh), np.float32)
    bq[0, 0] = A                    # every query gets +30 on BOS key
    P["l0.wq"] = wq; P["l0.wk"] = wk; P["l0.wv"] = wv
    P["l0.bq"] = bq
    P["l0.bk"] = np.zeros((H, dh), np.float32)
    P["l0.bv"] = np.zeros((H, dh), np.float32)
    wo = np.zeros((H, dh, d), np.float32)
    wo[0, 0, 7] = 1.0               # write a into dim 7
    P["l0.wo"] = wo
    P["l0.bo"] = np.zeros(d, np.float32)

    # MLP: piecewise-linear map a=1/(c+1) -> c, exact at the nodes
    a_nodes = [1.0 / (8 - k) for k in range(7)]   # 1/8 .. 1/2
    c_nodes = [7.0 - k for k in range(7)]         # 7 .. 1
    slopes = [(c_nodes[k+1] - c_nodes[k]) / (a_nodes[k+1] - a_nodes[k]) for k in range(6)]
    w1 = np.zeros((d, dm), np.float32)
    b1 = np.zeros(dm, np.float32)
    w2 = np.zeros((dm, d), np.float32)
    for k in range(6):
        w1[7, k] = 1.0
        b1[k] = -a_nodes[k]
        w2[k, 8] = slopes[k] - (slopes[k-1] if k > 0 else 0.0)
    P["l0.w1"] = w1; P["l0.b1"] = b1; P["l0.w2"] = w2
    P["l0.b2"] = np.zeros(d, np.float32)

    uw = np.zeros((d, 1), np.float32)
    uw[8, 0] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.array([7.0], np.float32)   # f(a_0) = 7
    return P