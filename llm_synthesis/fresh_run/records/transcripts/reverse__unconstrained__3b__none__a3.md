# Construct a transformer that computes `reverse` (no training)

You are asked to construct the parameters of a small transformer that exactly implements a specified function. The parameters must be derived by your own reasoning, not obtained by training.

## The model family (fixed)
The permitted model family: a decoder/encoder transformer with explicit parameters (no layer norm, ReLU MLP, softmax attention scaled by 1/sqrt(d_head)).
   x = embed[tok] + pos_embed[pos];  per layer:  x += sum_h softmax(q_h k_h^T / sqrt(d_head)) v_h W_o[h] + b_o ;  x += relu(x W1 + b1) W2 + b2 ;  y = x Wu + bu.
Parameters (flat dict name -> float32 array):
   embed [7,d]  pos_embed [8,d]
   l{i}.wq/wk/wv [H,d,dh]  l{i}.bq/bk/bv [H,dh]  l{i}.wo [H,dh,d]  l{i}.bo [d]  l{i}.w1 [d,dm]  l{i}.b1 [dm]  l{i}.w2 [dm,d]  l{i}.b2 [d]
   unembed.w [d,d_out]  unembed.b [d_out]
Config: n_layers, n_heads, d_model, d_head, d_mlp, causal (bool).

The forward pass is exactly this code (JAX):
```python
def forward(params, cfg, tokens):
    """tokens (B,T) int -> (B,T,d_out)"""
    T = tokens.shape[1]; x = params["embed"][tokens] + params["pos_embed"][:T][None]; dh = cfg["d_head"]; mask = jnp.tril(jnp.ones((T, T), bool)) if cfg["causal"] else jnp.ones((T, T), bool)
    for i in range(cfg["n_layers"]):
        p = lambda n: params[f"l{i}.{n}"]
        q = jnp.einsum("btd,hde->bhte", x, p("wq")) + p("bq")[None, :, None]; k = jnp.einsum("btd,hde->bhte", x, p("wk")) + p("bk")[None, :, None]; v = jnp.einsum("btd,hde->bhte", x, p("wv")) + p("bv")[None, :, None]
        s = jnp.einsum("bhte,bhse->bhts", q, k) / jnp.sqrt(dh * 1.0); s = jnp.where(mask[None, None], s, -1e9); a = jax.nn.softmax(s, -1); o = jnp.einsum("bhts,bhse->bhte", a, v)
        x = x + jnp.einsum("bhte,hed->btd", o, p("wo")) + p("bo"); x = x + jax.nn.relu(x @ p("w1") + p("b1")) @ p("w2") + p("b2")
    return x @ params["unembed.w"] + params["unembed.b"]
```
Token ids: BOS = 0 (always prepended at position 0), input tokens are 1..6 at positions 1..L, with 1 <= L <= 7. Outputs are read at positions 1..L (the output at position 0 is ignored). The output width is d_out = 7.

## The function
The output at position i (1..L) is the token x[L+1-i], i.e. the input sequence reversed. Categorical output: the predicted token id is argmax over the 7 output logits (index = token id).

Reference implementation (the ground truth, Python):
```python
def _reverse(x): return list(x[::-1])
```

## Correctness requirement
Your network must be exactly correct on ALL 335,922 input sequences of length 1..7 over tokens 1..6 (every output position of every sequence). Nothing else is evaluated about correctness; only exactly correct submissions are accepted.

## Condition
UNCONSTRAINED SYNTHESIS. You choose the architecture yourself within these limits: n_layers <= 4, n_heads <= 4, d_model <= 64, d_head <= 32, d_mlp <= 256, causal = true or false, and at most 60000 parameters in total.

## Deliverable
DELIVERABLE (one Python file `submission.py`, numpy only):
```python
import numpy as np
CONFIG = dict(n_layers=..., n_heads=..., d_model=..., d_head=..., d_mlp=..., causal=...)
def build_params():
    # return a dict {tensor_name: np.ndarray} containing EVERY tensor with the exact shapes above
    ...
```
You write the program that computes the weights; the weights must come from your own analytical construction (loops, formulas, structured fills are fine). Do NOT train, fit, search or optimise: no gradient descent, no random search, no least-squares fitting to enumerated inputs, no solver calls. Only numpy may be imported.

## No feedback
NO FEEDBACK. This is a single attempt: you will not receive any information about how your submission performs. Think carefully, verify your construction by reasoning, and submit one final answer.

Report: the final deliverable, the number of parameters, and in a few sentences how the construction works.


```python
import numpy as np

CONFIG = dict(n_layers=3, n_heads=1, d_model=36, d_head=8, d_mlp=28, causal=False)

def build_params():
    d, H, dh, dm, L = 36, 1, 8, 28, 3
    P = {}
    # layout: 0-7 pos one-hot | 8-13 token one-hot (tok1..6) | 14 f=1/T
    #         15-21 one-hot of T=2..8 (T = L+1) | 22-29 one-hot of target position | 30-35 gathered token
    emb = np.zeros((7, d), np.float32)
    for t in range(1, 7):
        emb[t, 8 + t - 1] = 1
    pe = np.zeros((8, d), np.float32)
    for p in range(8):
        pe[p, p] = 1
    P["embed"], P["pos_embed"] = emb, pe
    for i in range(L):
        P[f"l{i}.wq"] = np.zeros((H, d, dh), np.float32)
        P[f"l{i}.wk"] = np.zeros((H, d, dh), np.float32)
        P[f"l{i}.wv"] = np.zeros((H, d, dh), np.float32)
        P[f"l{i}.bq"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.bk"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.bv"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.wo"] = np.zeros((H, dh, d), np.float32)
        P[f"l{i}.bo"] = np.zeros(d, np.float32)
        P[f"l{i}.w1"] = np.zeros((d, dm), np.float32)
        P[f"l{i}.b1"] = np.zeros(dm, np.float32)
        P[f"l{i}.w2"] = np.zeros((dm, d), np.float32)
        P[f"l{i}.b2"] = np.zeros(d, np.float32)

    # Layer 0 attention: uniform attention, value = BOS indicator -> f = 1/T at every position
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 14] = 1.0
    # Layer 0 MLP: triangular hat functions of f centred at 1/T -> one-hot of T
    pts = [0.0] + [1.0 / T for T in range(8, 1, -1)] + [1.0]   # ascending
    for T in range(2, 9):
        c = 1.0 / T
        idx = pts.index(c) if c in pts else int(np.argmin([abs(c - q) for q in pts]))
        clo, chi = pts[idx - 1], pts[idx + 1]
        s1, s2 = 1.0 / (c - clo), 1.0 / (chi - c)
        n = 3 * (T - 2)
        out = 15 + T - 2
        P["l0.w1"][14, n] = s1;  P["l0.b1"][n] = -s1 * clo;  P["l0.w2"][n, out] = 1.0
        P["l0.w1"][14, n + 1] = 1.0;  P["l0.b1"][n + 1] = -c;  P["l0.w2"][n + 1, out] = -(s1 + s2)
        P["l0.w1"][14, n + 2] = 1.0;  P["l0.b1"][n + 2] = -chi;  P["l0.w2"][n + 2, out] = s2

    # Layer 1 MLP: neuron per (T, i): fires iff length-code T and position i; outputs one-hot of j = T - i
    n = 0
    for T in range(2, 9):
        for i in range(1, T):
            P["l1.w1"][15 + T - 2, n] = 1.0
            P["l1.w1"][i, n] = 1.0
            P["l1.b1"][n] = -1.0
            P["l1.w2"][n, 22 + (T - i)] = 1.0
            n += 1

    # Layer 2 attention: query = target one-hot, key = position one-hot, value = token one-hot
    B = 60.0
    for m in range(8):
        P["l2.wq"][0, 22 + m, m] = B
        P["l2.wk"][0, m, m] = 1.0
    for k in range(6):
        P["l2.wv"][0, 8 + k, k] = 1.0
        P["l2.wo"][0, k, 30 + k] = 1.0

    uw = np.zeros((d, 7), np.float32)
    for k in range(6):
        uw[30 + k, k + 1] = 1.0
    P["unembed.w"], P["unembed.b"] = uw, np.zeros(7, np.float32)
    return {k: v.astype(np.float32) for k, v in P.items()}
```

**Parameter count:** 10,675. Each of the 3 layers has 1,212 attention parameters and 2,080 MLP parameters, giving 9,876. Embeddings add 540 and the unembedding adds 259.

**How it works:** The model is non-causal (`causal=False`) and assumes the sequence is fed at length T = L+1, with no padding. The hidden state holds one-hot blocks.
- **Layer 0 attention:** All scores are zero, so attention is uniform. The value is the BOS indicator (position-0 one-hot), so every position receives f = 1/T.
- **Layer 0 MLP:** Triangular "hat" functions of f centred at 1/2, …, 1/8 turn f into a one-hot code of T. Each hat is built from three ReLUs, and the spacing between neighbouring values is far larger than float error.
- **Layer 1 MLP:** One neuron per (T, i) pair, computed as relu(T-onehot[T] + pos-onehot[i] − 1), writes a one-hot of the target position j = T − i = L+1−i. Layer 1's attention is zero.
- **Layer 2 attention:** The query is 60 × the target one-hot and the key is the position one-hot, so the score is 60/√8 ≈ 21 at j and 0 elsewhere. Softmax leakage is around 1e-9. The value is the token one-hot, and it is copied into dedicated dimensions.
- **Unembedding:** Those dimensions map straight to logits 1..6. Position 0 is ignored.

I did not run or test this; I checked it only by hand.