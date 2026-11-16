import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=4, d_head=1, d_mlp=7, causal=True)

def build_params():
    d, dh, H, dm = 4, 1, 1, 7
    P = {}
    # residual dims: 0 = [token==1], 1 = key score (0 for BOS/token 1, -40 otherwise),
    #                2 = attention output f, 3 = parity score g
    emb = np.zeros((7, d))
    emb[1, 0] = 1.0
    for t in range(2, 7):
        emb[t, 1] = -40.0
    P["embed"] = emb
    P["pos_embed"] = np.zeros((8, d))

    wq = np.zeros((H, d, dh)); bq = np.ones((H, dh))      # constant query = 1
    wk = np.zeros((H, d, dh)); wk[0, 1, 0] = 1.0          # key = dim 1
    wv = np.zeros((H, d, dh)); wv[0, 0, 0] = 1.0          # value = [token==1]
    wo = np.zeros((H, dh, d)); wo[0, 0, 2] = 1.0          # write to dim 2
    P["l0.wq"], P["l0.bq"] = wq, bq
    P["l0.wk"], P["l0.bk"] = wk, np.zeros((H, dh))
    P["l0.wv"], P["l0.bv"] = wv, np.zeros((H, dh))
    P["l0.wo"], P["l0.bo"] = wo, np.zeros(d)

    # After attention, f = c/(c+1), c = number of ones so far (BOS and each 1 get weight 1).
    fc = np.array([c / (c + 1.0) for c in range(8)])
    pc = np.array([c % 2 for c in range(8)], dtype=float)
    slopes = (pc[1:] - pc[:-1]) / (fc[1:] - fc[:-1])
    a = np.concatenate([[slopes[0]], slopes[1:] - slopes[:-1]])
    w1 = np.zeros((d, dm)); b1 = np.zeros(dm)
    w2 = np.zeros((dm, d))
    for j in range(7):
        w1[2, j] = 1.0
        b1[j] = -fc[j]
        w2[j, 3] = a[j]
    P["l0.w1"], P["l0.b1"], P["l0.w2"], P["l0.b2"] = w1, b1, w2, np.zeros(d)

    uw = np.zeros((d, 2)); uw[3, 1] = 1.0
    ub = np.array([0.5, 0.0])
    P["unembed.w"], P["unembed.b"] = uw, ub
    return {k: np.asarray(v, dtype=np.float32) for k, v in P.items()}