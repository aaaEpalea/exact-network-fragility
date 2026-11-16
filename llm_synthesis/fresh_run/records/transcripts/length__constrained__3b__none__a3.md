# Construct a transformer that computes `length` (no training)

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
Token ids: BOS = 0 (always prepended at position 0), input tokens are 1..6 at positions 1..L, with 1 <= L <= 7. Outputs are read at positions 1..L (the output at position 0 is ignored). The output width is d_out = 1.

## The function
For every position i of the input sequence the output is the length L of the input sequence (a number; the BOS token is not counted). Numeric output: |y - L| <= 0.5 counts as correct.

Reference implementation (the ground truth, Python):
```python
def _length(x): return [len(x)] * len(x)
```

## Correctness requirement
Your network must be exactly correct on ALL 335,922 input sequences of length 1..7 over tokens 1..6 (every output position of every sequence). Nothing else is evaluated about correctness; only exactly correct submissions are accepted.

## Condition
CONSTRAINED SYNTHESIS. You must use exactly this architecture (no other choice is allowed):
```
{"n_layers": 1, "n_heads": 1, "d_model": 16, "d_head": 8, "d_mlp": 32, "causal": false}
```
This fixes every tensor shape (1881 parameters in total).

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


**submission.py**

```python
import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=16, d_head=8, d_mlp=32, causal=False)

def build_params():
    d, H, dh, dm = 16, 1, 8, 32
    P = {}
    f = lambda *s: np.zeros(s, dtype=np.float32)
    P["embed"] = f(7, d)
    P["pos_embed"] = f(8, d)
    P["embed"][0, 0] = 1.0            # BOS indicator in residual dim 0

    # attention: zero q/k -> uniform attention over all L+1 positions
    P["l0.wq"] = f(H, d, dh); P["l0.bq"] = f(H, dh)
    P["l0.wk"] = f(H, d, dh); P["l0.bk"] = f(H, dh)
    P["l0.wv"] = f(H, d, dh); P["l0.bv"] = f(H, dh)
    P["l0.wo"] = f(H, dh, d); P["l0.bo"] = f(d)
    P["l0.wv"][0, 0, 0] = 1.0         # value = BOS indicator
    P["l0.wo"][0, 0, 1] = 1.0         # write s = 1/(L+1) into residual dim 1

    # MLP: piecewise-linear map g(s) = L at s = 1/(L+1), L = 1..7
    P["l0.w1"] = f(d, dm); P["l0.b1"] = f(dm)
    P["l0.w2"] = f(dm, d); P["l0.b2"] = f(d)
    knots = [1.0 / (8 - j) for j in range(6)]          # 1/8, 1/7, ..., 1/3
    slopes = [-1.0 / (1.0 / (7 - j) - 1.0 / (8 - j)) for j in range(6)]
    prev = 0.0
    for j in range(6):
        P["l0.w1"][1, j] = 1.0
        P["l0.b1"][j] = -knots[j]
        P["l0.w2"][j, 2] = slopes[j] - prev
        prev = slopes[j]
    P["l0.b2"][2] = 7.0               # g(1/8) = 7

    P["unembed.w"] = f(d, 1); P["unembed.b"] = f(1)
    P["unembed.w"][2, 0] = 1.0
    return P
```

**Parameter count:** 1881, since every tensor has the exact required shape. Most entries are zero.

**How it works:**
- The embedding puts a 1 in residual dim 0 only for BOS. Token and position embeddings are otherwise zero.
- The query and key weights and biases are all zero, so every attention score is 0. The attention is non-causal, so each position attends uniformly to all L+1 positions, BOS included. The value is the BOS indicator, so the attention output is 1/(L+1). The output projection writes this into residual dim 1, at every position.
- The MLP applies a piecewise-linear function g to that value s. It uses 6 ReLU units `relu(s − 1/(8−j))` for j = 0..5, with output weights equal to the slope changes and a bias of 7. This makes g(1/(L+1)) = L exactly at all seven knots, L = 1..7.
- The unembedding reads residual dim 2, which holds g(s), so y = L at every position.
- The assumed input is T = L+1 tokens with no padding, so the attention denominator is exactly L+1. The error is only float32 rounding, orders of magnitude below the 0.5 tolerance, so all 335,922 sequences should be correct.
- I did not run this. The correctness argument rests on the construction above.
