import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=3, d_head=1, d_mlp=6, causal=False)

def build_params():
    d, H, dh, dm = 3, 1, 1, 6
    P = {}
    # dim0 = BOS indicator, dim1 = attention average f, dim2 = output
    embed = np.zeros((7, d)); embed[0, 0] = 1.0
    P["embed"] = embed
    P["pos_embed"] = np.zeros((8, d))
    # attention: zero q,k -> uniform weights over all positions 0..L
    P["l0.wq"] = np.zeros((H, d, dh)); P["l0.bq"] = np.zeros((H, dh))
    P["l0.wk"] = np.zeros((H, d, dh)); P["l0.bk"] = np.zeros((H, dh))
    wv = np.zeros((H, d, dh)); wv[0, 0, 0] = 1.0       # value = BOS indicator
    P["l0.wv"] = wv; P["l0.bv"] = np.zeros((H, dh))
    wo = np.zeros((H, dh, d)); wo[0, 0, 1] = 1.0       # write f = 1/(L+1) into dim1
    P["l0.wo"] = wo; P["l0.bo"] = np.zeros(d)
    # MLP: piecewise-linear interpolation of f=1/(n+1) -> n, n=1..7
    f = np.array([1.0 / (n + 1) for n in range(7, 0, -1)])   # 1/8 ... 1/2 (increasing)
    v = np.array([float(n) for n in range(7, 0, -1)])        # 7 ... 1
    slopes = (v[1:] - v[:-1]) / (f[1:] - f[:-1])             # 6 segments
    w1 = np.zeros((d, dm)); b1 = np.zeros(dm); w2 = np.zeros((dm, d))
    prev = 0.0
    for j in range(6):
        w1[1, j] = 1.0
        b1[j] = -f[j]
        w2[j, 2] = slopes[j] - prev
        prev = slopes[j]
    b2 = np.zeros(d); b2[2] = 7.0
    P["l0.w1"] = w1; P["l0.b1"] = b1; P["l0.w2"] = w2; P["l0.b2"] = b2
    uw = np.zeros((d, 1)); uw[2, 0] = 1.0
    P["unembed.w"] = uw; P["unembed.b"] = np.zeros(1)
    return {k: np.asarray(a, dtype=np.float32) for k, a in P.items()}