import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=11, d_head=1, d_mlp=56, causal=True)

def build_params():
    d, dm, H, dh = 11, 56, 1, 1
    K, M = 32.0, 100.0
    P = {}
    embed = np.zeros((7, d), np.float32)
    embed[1, 8] = 1.0                      # token 1 -> is1 feature (dim 8)
    pos = np.zeros((8, d), np.float32)
    for t in range(8):
        pos[t, t] = 1.0                    # one-hot position (dims 0..7)
    P["embed"], P["pos_embed"] = embed, pos

    # attention: zero q/k -> uniform causal average over positions 0..t
    P["l0.wq"] = np.zeros((H, d, dh), np.float32)
    P["l0.wk"] = np.zeros((H, d, dh), np.float32)
    P["l0.bq"] = np.zeros((H, dh), np.float32)
    P["l0.bk"] = np.zeros((H, dh), np.float32)
    wv = np.zeros((H, d, dh), np.float32); wv[0, 8, 0] = 1.0
    P["l0.wv"] = wv
    P["l0.bv"] = np.zeros((H, dh), np.float32)
    wo = np.zeros((H, dh, d), np.float32); wo[0, 0, 9] = 1.0   # m = c/(t+1) -> dim 9
    P["l0.wo"] = wo
    P["l0.bo"] = np.zeros(d, np.float32)

    w1 = np.zeros((d, dm), np.float32); b1 = np.zeros(dm, np.float32)
    w2 = np.zeros((dm, d), np.float32)
    u = 0
    for t in range(1, 8):
        for j in range(1, t + 1):
            thr = (j - 0.5) / (t + 1)
            sign = 1.0 if j % 2 == 1 else -1.0
            for off, outsign in ((1.0, sign), (0.0, -sign)):
                w1[9, u] = K
                w1[t, u] = M
                b1[u] = -K * thr - M + off
                w2[u, 10] = outsign        # parity accumulator in dim 10
                u += 1
    assert u == dm
    P["l0.w1"], P["l0.b1"], P["l0.w2"] = w1, b1, w2
    P["l0.b2"] = np.zeros(d, np.float32)

    uw = np.zeros((d, 2), np.float32)
    uw[10, 1] = 1.0
    uw[10, 0] = 0.0
    ub = np.array([0.0, -0.5], np.float32)  # logit1 = p-0.5, logit0 = 0
    P["unembed.w"], P["unembed.b"] = uw, ub
    return P