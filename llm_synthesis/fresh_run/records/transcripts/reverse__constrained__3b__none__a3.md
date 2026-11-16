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
CONSTRAINED SYNTHESIS. You must use exactly this architecture (no other choice is allowed):
```
{"n_layers": 2, "n_heads": 2, "d_model": 32, "d_head": 16, "d_mlp": 64, "causal": false}
```
This fixes every tensor shape (17543 parameters in total).

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


I haven't run this code, so correctness rests on the derivation below, not on testing.

## Deliverable: `submission.py`

```python
import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=64, causal=False)

def build_params():
    d, H, dh, dm = 32, 2, 16, 64
    P = {}
    f = lambda *s: np.zeros(s, np.float32)

    # residual layout: 0..6 token one-hot (0 = BOS), 7 = pos i, 8 = pos^2, 9 = const 1,
    # 10 = a = 1/(L+1), 11 = c = L+1, 12..18 = attended token one-hot
    embed = f(7, d)
    for t in range(7):
        embed[t, t] = 1.0
    pos = f(8, d)
    for p in range(8):
        pos[p, 7] = p
        pos[p, 8] = p * p
        pos[p, 9] = 1.0
    P["embed"] = embed
    P["pos_embed"] = pos

    for i in range(2):
        for n, s in [("wq", (H, d, dh)), ("wk", (H, d, dh)), ("wv", (H, d, dh)),
                     ("bq", (H, dh)), ("bk", (H, dh)), ("bv", (H, dh)),
                     ("wo", (H, dh, d)), ("bo", (d,)),
                     ("w1", (d, dm)), ("b1", (dm,)), ("w2", (dm, d)), ("b2", (d,))]:
            P[f"l{i}.{n}"] = f(*s)

    # ---- layer 0: uniform attention averages the BOS indicator -> a = 1/(L+1)
    P["l0.wv"][0, 0, 0] = 1.0          # head 0 value = BOS indicator
    P["l0.wo"][0, 0, 10] = 1.0         # write a into dim 10
    # MLP: piecewise-linear interpolation of c = 1/a through (1/n, n), n = 2..8
    s_prev = 0.0
    for k in range(6):
        s_k = -float((7 - k) * (8 - k))
        P["l0.w1"][10, k] = 1.0
        P["l0.b1"][k] = -1.0 / (8 - k)
        P["l0.w2"][k, 11] = s_k - s_prev
        s_prev = s_k
    P["l0.b2"][11] = 8.0               # c = 8 at a = 1/8

    # ---- layer 1: head 0 attends from position i to j = c - i
    K = 10.0
    P["l1.wq"][0, 9, 0] = 4.0          # q0 = 4
    P["l1.wq"][0, 7, 1] = 4.0          # q1 = 4 (i - c)
    P["l1.wq"][0, 11, 1] = -4.0
    P["l1.wk"][0, 8, 0] = -K           # k0 = -K j^2
    P["l1.wk"][0, 7, 1] = -2 * K       # k1 = -2K j
    # score = -K j^2 - 2K j (i - c) = -K (i + j - c)^2 + (terms constant in j)
    for m in range(7):
        P["l1.wv"][0, m, m] = 1.0      # value = token one-hot
        P["l1.wo"][0, m, 12 + m] = 1.0
    # head 1 and layer-1 MLP stay zero

    uw = f(d, 7)
    for t in range(7):
        uw[12 + t, t] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = f(7)
    return P
```

**Parameter count:** 17,543, matching the fixed architecture.

## How it works

1. **Layer 0 attention** has all-zero queries and keys, so attention is uniform over the L+1 positions. Head 0's value is the BOS indicator, so every position receives a = 1/(L+1).
2. **Layer 0 MLP** computes c = L+1 from a. It uses six ReLUs to build the piecewise-linear function through the points (1/n, n) for n = 2..8, which is exact at every possible length. c is written to dim 11. The residual also holds the position index i (dim 7), i² (dim 8) and a constant 1 (dim 9) from `pos_embed`.
3. **Layer 1 attention**, head 0, uses the query/key construction to make the score equal −K·(i + j − c)² plus terms that depend only on the query position i. Those i-only terms cancel in the softmax.
   - The score is peaked exactly at j = c − i = L+1−i, with K = 10.
   - Neighbours (i + j − c = ±1) get weight about e⁻¹⁰ relative to the peak.
   - BOS at j = 0 never matches for i in 1..L.
   - The value is the token one-hot, copied into dims 12–18.
4. **Unembed** maps dim 12+t to logit t, so the argmax is the token at position L+1−i, which is the reversed sequence.

Layer 1's MLP, layer 0's head 1 (zero values) and layer 1's head 1 are unused and left at zero. This assumes each sequence is evaluated at its own length T = L+1, with no padding. The embedding table has no pad token, so this is the only reading consistent with the setup.