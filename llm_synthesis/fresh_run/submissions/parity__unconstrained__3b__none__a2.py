import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=11, d_head=1, d_mlp=56, causal=True)

def build_params():
    d, dm, M = 11, 56, 20.0
    P = {}
    # residual layout: 0..7 position one-hot, 8 is_one flag, 9 mean (attention output), 10 parity
    E = np.zeros((7, d), np.float32); E[1, 8] = 1.0
    PE = np.zeros((8, d), np.float32)
    for p in range(8): PE[p, p] = 1.0
    P["embed"] = E; P["pos_embed"] = PE
    P["l0.wq"] = np.zeros((1, d, 1), np.float32)
    P["l0.wk"] = np.zeros((1, d, 1), np.float32)
    wv = np.zeros((1, d, 1), np.float32); wv[0, 8, 0] = 1.0
    P["l0.wv"] = wv
    P["l0.bq"] = np.zeros((1, 1), np.float32)
    P["l0.bk"] = np.zeros((1, 1), np.float32)
    P["l0.bv"] = np.zeros((1, 1), np.float32)
    wo = np.zeros((1, 1, d), np.float32); wo[0, 0, 9] = 1.0
    P["l0.wo"] = wo
    P["l0.bo"] = np.zeros(d, np.float32)
    w1 = np.zeros((d, dm), np.float32); b1 = np.zeros(dm, np.float32)
    w2 = np.zeros((dm, d), np.float32)
    j = 0
    for t in range(1, 8):
        for k in range(1, t + 1):
            sign = 1.0 if k % 2 == 1 else -1.0
            # unit A: relu(c - k + 1) when at position t; unit B: relu(c - k)
            for off, sg in ((1, 1.0), (0, -1.0)):
                w1[9, j] = t + 1          # c = (t+1)*mean
                w1[t, j] = M              # gate on position t
                b1[j] = -k + off - M
                w2[j, 10] = sign * sg
                j += 1
    assert j == dm
    P["l0.w1"] = w1; P["l0.b1"] = b1; P["l0.w2"] = w2
    P["l0.b2"] = np.zeros(d, np.float32)
    uw = np.zeros((d, 2), np.float32); uw[10, 0] = -1.0; uw[10, 1] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.array([0.5, -0.5], np.float32)
    return P