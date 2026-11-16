import numpy as np

D, H, DH, DM = 15, 1, 6, 1
BETA = 10.0
CONFIG = dict(n_layers=2, n_heads=H, d_model=D, d_head=DH, d_mlp=DM, causal=False)

def build_params():
    f = np.float32
    P = {}
    # dims: 0 = position j, 1 = j^2, 2..7 = token one-hot (tokens 1..6),
    #       8 = mean position (T-1)/2, 9..14 = copied token one-hot
    emb = np.zeros((7, D), f)
    for t in range(1, 7):
        emb[t, 1 + t] = 1.0
    pos = np.zeros((8, D), f)
    for j in range(8):
        pos[j, 0] = j
        pos[j, 1] = j * j
    P["embed"], P["pos_embed"] = emb, pos

    def zeros_layer(i):
        P[f"l{i}.wq"] = np.zeros((H, D, DH), f); P[f"l{i}.wk"] = np.zeros((H, D, DH), f)
        P[f"l{i}.wv"] = np.zeros((H, D, DH), f)
        P[f"l{i}.bq"] = np.zeros((H, DH), f); P[f"l{i}.bk"] = np.zeros((H, DH), f)
        P[f"l{i}.bv"] = np.zeros((H, DH), f)
        P[f"l{i}.wo"] = np.zeros((H, DH, D), f); P[f"l{i}.bo"] = np.zeros(D, f)
        P[f"l{i}.w1"] = np.zeros((D, DM), f); P[f"l{i}.b1"] = np.zeros(DM, f)
        P[f"l{i}.w2"] = np.zeros((DM, D), f); P[f"l{i}.b2"] = np.zeros(D, f)

    # Layer 0: uniform attention over all positions, value = position index
    # -> every position receives m = (T-1)/2, written to dim 8.
    zeros_layer(0)
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 8] = 1.0

    # Layer 1: score(i, j) = -BETA*(j - a)^2 + const, with a = T - i = 2m + 1 - i
    zeros_layer(1)
    c = np.sqrt(DH)  # cancels the 1/sqrt(d_head) scaling
    P["l1.wq"][0, 8, 0] = 4 * BETA * c
    P["l1.wq"][0, 0, 0] = -2 * BETA * c
    P["l1.bq"][0, 0] = 2 * BETA * c
    P["l1.bq"][0, 1] = -BETA * c
    P["l1.wk"][0, 0, 0] = 1.0   # key feature j
    P["l1.wk"][0, 1, 1] = 1.0   # key feature j^2
    for t in range(6):
        P["l1.wv"][0, 2 + t, t] = 1.0
        P["l1.wo"][0, t, 9 + t] = 1.0

    uw = np.zeros((D, 7), f)
    for t in range(6):
        uw[9 + t, t + 1] = 1.0
    P["unembed.w"], P["unembed.b"] = uw, np.zeros(7, f)
    return P