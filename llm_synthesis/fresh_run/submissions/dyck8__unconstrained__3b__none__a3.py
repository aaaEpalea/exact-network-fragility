import numpy as np

CONFIG = dict(n_layers=2, n_heads=1, d_model=12, d_head=1, d_mlp=16, causal=True)

def build_params():
    d, H, dh, dm = 12, 1, 1, 16
    P = {}
    # dims: 0 = a(token) (+1 open, -1 close, 0 else); 1..8 = one-hot position;
    #       9 = running mean of a; 10 = bad flag; 11 = mean of bad flag
    E = np.zeros((7, d), np.float32)
    E[1, 0] = 1.0
    E[2, 0] = -1.0
    P["embed"] = E
    PE = np.zeros((8, d), np.float32)
    for t in range(8):
        PE[t, 1 + t] = 1.0
    P["pos_embed"] = PE

    K = 10.0
    # ---- layer 0: uniform causal attention averages a over positions 0..t
    P["l0.wq"] = np.zeros((H, d, dh), np.float32)
    P["l0.wk"] = np.zeros((H, d, dh), np.float32)
    wv = np.zeros((H, d, dh), np.float32); wv[0, 0, 0] = 1.0
    P["l0.wv"] = wv
    P["l0.bq"] = np.zeros((H, dh), np.float32)
    P["l0.bk"] = np.zeros((H, dh), np.float32)
    P["l0.bv"] = np.zeros((H, dh), np.float32)
    wo = np.zeros((H, dh, d), np.float32); wo[0, 0, 9] = 1.0
    P["l0.wo"] = wo
    P["l0.bo"] = np.zeros(d, np.float32)
    # MLP: bad_t = relu(-B) - relu(-B-1), B = (t+1)*avg, gated by position one-hot
    w1 = np.zeros((d, dm), np.float32)
    b1 = np.zeros(dm, np.float32)
    w2 = np.zeros((dm, d), np.float32)
    for t in range(1, 8):
        ia, ib = t - 1, 7 + t - 1
        w1[9, ia] = -(t + 1.0); w1[1 + t, ia] = K; b1[ia] = -K
        w1[9, ib] = -(t + 1.0); w1[1 + t, ib] = K; b1[ib] = -K - 1.0
        w2[ia, 10] = 1.0
        w2[ib, 10] = -1.0
    P["l0.w1"], P["l0.b1"], P["l0.w2"] = w1, b1, w2
    P["l0.b2"] = np.zeros(d, np.float32)

    # ---- layer 1: uniform causal attention averages bad flag over 0..t
    P["l1.wq"] = np.zeros((H, d, dh), np.float32)
    P["l1.wk"] = np.zeros((H, d, dh), np.float32)
    wv = np.zeros((H, d, dh), np.float32); wv[0, 10, 0] = 1.0
    P["l1.wv"] = wv
    P["l1.bq"] = np.zeros((H, dh), np.float32)
    P["l1.bk"] = np.zeros((H, dh), np.float32)
    P["l1.bv"] = np.zeros((H, dh), np.float32)
    wo = np.zeros((H, dh, d), np.float32); wo[0, 0, 11] = 1.0
    P["l1.wo"] = wo
    P["l1.bo"] = np.zeros(d, np.float32)
    P["l1.w1"] = np.zeros((d, dm), np.float32)
    P["l1.b1"] = np.zeros(dm, np.float32)
    P["l1.w2"] = np.zeros((dm, d), np.float32)
    P["l1.b2"] = np.zeros(d, np.float32)

    # ---- unembed: class0 = 16*mean_bad, class1 = 1
    uw = np.zeros((d, 2), np.float32); uw[11, 0] = 16.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.array([0.0, 1.0], np.float32)
    return P