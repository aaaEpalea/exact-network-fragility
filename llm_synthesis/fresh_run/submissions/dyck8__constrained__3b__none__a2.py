import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=128, causal=True)

def build_params():
    d, H, dh, dm = 32, 2, 16, 128
    P = {}
    P["embed"] = np.zeros((7, d), np.float32)
    P["pos_embed"] = np.zeros((8, d), np.float32)
    for i in range(2):
        P[f"l{i}.wq"] = np.zeros((H, d, dh), np.float32)
        P[f"l{i}.wk"] = np.zeros((H, d, dh), np.float32)
        P[f"l{i}.wv"] = np.zeros((H, d, dh), np.float32)
        P[f"l{i}.bq"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.bk"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.bv"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.wo"] = np.zeros((H, dh, d), np.float32)
        P[f"l{i}.bo"] = np.zeros((d,), np.float32)
        P[f"l{i}.w1"] = np.zeros((d, dm), np.float32)
        P[f"l{i}.b1"] = np.zeros((dm,), np.float32)
        P[f"l{i}.w2"] = np.zeros((dm, d), np.float32)
        P[f"l{i}.b2"] = np.zeros((d,), np.float32)
    P["unembed.w"] = np.zeros((d, 2), np.float32)
    P["unembed.b"] = np.zeros((2,), np.float32)

    # token value in dim 0: '(' -> +1, ')' -> -1, others (incl. BOS) -> 0
    P["embed"][1, 0] = 1.0
    P["embed"][2, 0] = -1.0

    # Layer 0, head 0: q=k=0 -> uniform causal attention over positions 0..t.
    # value = dim0, written to dim1: m_t = balance_t / (t+1)
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 1] = 1.0
    # Layer 0 MLP: f_t = relu(-8 * m_t) -> dim2. Balance<0 gives f >= 1; balance>=0 gives f = 0.
    P["l0.w1"][1, 0] = -8.0
    P["l0.w2"][0, 2] = 1.0

    # Layer 1, head 0: uniform causal average of f -> dim3 (0 if never bad, else >= 1/8)
    P["l1.wv"][0, 2, 0] = 1.0
    P["l1.wo"][0, 0, 3] = 1.0

    # Output: logit0 = 0, logit1 = 1 - 16*mean_f  (argmax=1 iff prefix valid)
    P["unembed.w"][3, 1] = -16.0
    P["unembed.b"][1] = 1.0
    return P