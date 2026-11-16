import numpy as np
CONFIG = dict(n_layers=2, n_heads=2, d_model=16, d_head=8, d_mlp=64, causal=True)
def build_params():
    d, H, dh, dm = 16, 2, 8, 64
    f32 = np.float32
    P = {}
    emb = np.zeros((7, d)); emb[1, 0] = 1.0
    pe = np.zeros((8, d))
    for p in range(8): pe[p, 1 + p] = 1.0
    P["embed"] = emb; P["pos_embed"] = pe
    for i in range(2):
        for n in ("wq", "wk", "wv"): P[f"l{i}.{n}"] = np.zeros((H, d, dh))
        for n in ("bq", "bk", "bv"): P[f"l{i}.{n}"] = np.zeros((H, dh))
        P[f"l{i}.wo"] = np.zeros((H, dh, d)); P[f"l{i}.bo"] = np.zeros(d)
        P[f"l{i}.w1"] = np.zeros((d, dm)); P[f"l{i}.b1"] = np.zeros(dm)
        P[f"l{i}.w2"] = np.zeros((dm, d)); P[f"l{i}.b2"] = np.zeros(d)
    # layer 0 head 0: uniform attention (zero scores), averages the "token==1" indicator into dim 9
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 9] = 1.0
    M = 100.0
    u = 0
    for t in range(1, 8):
        N = t + 1
        v = [2.0 * (c % 2) - 1.0 for c in range(N)]
        slopes = [(v[j + 1] - v[j]) * N for j in range(t)]
        prev = 0.0
        for j in range(t):
            s = slopes[j] - prev; prev = slopes[j]
            P["l0.w1"][9, u] = 1.0
            P["l0.w1"][1 + t, u] = M
            P["l0.b1"][u] = -(j / N) - M
            P["l0.w2"][u, 10] = s
            u += 1
    uw = np.zeros((d, 2)); uw[10, 1] = 0.5; uw[10, 0] = -0.5
    P["unembed.w"] = uw; P["unembed.b"] = np.array([0.5, -0.5])
    return {k: np.asarray(v, dtype=f32) for k, v in P.items()}