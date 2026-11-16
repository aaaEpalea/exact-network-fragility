import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=16, d_head=8, d_mlp=64, causal=True)

def build_params():
    d, H, dh, dm = 16, 2, 8, 64
    P = {}
    # residual layout: 0 = [tok==1], 1 = f (fraction of ones), 2..8 = one-hot position t (t=1..7), 9 = parity value p
    embed = np.zeros((7, d), np.float32)
    embed[1, 0] = 1.0
    pos = np.zeros((8, d), np.float32)
    for t in range(1, 8):
        pos[t, 1 + t] = 1.0
    P["embed"], P["pos_embed"] = embed, pos

    for i in range(2):
        for n in ("wq", "wk", "wv"):
            P[f"l{i}.{n}"] = np.zeros((H, d, dh), np.float32)
        for n in ("bq", "bk", "bv"):
            P[f"l{i}.{n}"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.wo"] = np.zeros((H, dh, d), np.float32)
        P[f"l{i}.bo"] = np.zeros(d, np.float32)
        P[f"l{i}.w1"] = np.zeros((d, dm), np.float32)
        P[f"l{i}.b1"] = np.zeros(dm, np.float32)
        P[f"l{i}.w2"] = np.zeros((dm, d), np.float32)
        P[f"l{i}.b2"] = np.zeros(d, np.float32)

    # Layer 0, head 0: q=k=0 -> uniform causal attention over positions 0..t;
    # value = [tok==1]; output written to dim 1 => f = c/(t+1)
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 1] = 1.0

    # Layer 0 MLP: for each position t, t neurons forming a zigzag in g=(t+1)f=c with p(c)=c%2
    n = 0
    for t in range(1, 8):
        for k in range(t):
            P["l0.w1"][1, n] = float(t + 1)        # (t+1)*f = c
            P["l0.w1"][1 + t, n] = 100.0           # gate: only active at position t
            P["l0.b1"][n] = -float(k) - 100.0
            P["l0.w2"][n, 9] = 1.0 if k == 0 else (2.0 if k % 2 == 0 else -2.0)
            n += 1
    # Layer 1 stays all-zero (identity).

    P["unembed.w"] = np.zeros((d, 2), np.float32)
    P["unembed.w"][9, 1] = 1.0
    P["unembed.b"] = np.array([0.5, 0.0], np.float32)
    return P