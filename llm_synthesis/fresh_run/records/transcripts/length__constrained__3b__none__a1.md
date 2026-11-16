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

```python
import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=16, d_head=8, d_mlp=32, causal=False)

def build_params():
    d, dh, dm, H = 16, 8, 32, 1
    P = {}
    z = lambda *s: np.zeros(s, np.float32)

    # embeddings: BOS carries a 1 in dim 0, everything else is 0
    P["embed"] = z(7, d); P["embed"][0, 0] = 1.0
    P["pos_embed"] = z(8, d)

    # attention: zero q/k -> uniform attention over all L+1 positions
    P["l0.wq"] = z(H, d, dh); P["l0.bq"] = z(H, dh)
    P["l0.wk"] = z(H, d, dh); P["l0.bk"] = z(H, dh)
    P["l0.wv"] = z(H, d, dh); P["l0.bv"] = z(H, dh)
    P["l0.wv"][0, 0, 0] = 1.0            # value = BOS indicator
    P["l0.wo"] = z(H, dh, d)
    P["l0.wo"][0, 0, 1] = 1.0            # mean -> residual dim 1 = 1/(L+1)
    P["l0.bo"] = z(d)

    # MLP: piecewise-linear interpolation of g(1/(L+1)) = L
    ts = [1.0 / (L + 1) for L in range(7, 0, -1)]   # ascending: 1/8 ... 1/2
    ys = [float(L) for L in range(7, 0, -1)]        # 7 ... 1
    slopes = [(ys[j + 1] - ys[j]) / (ts[j + 1] - ts[j]) for j in range(6)]
    P["l0.w1"] = z(d, dm); P["l0.b1"] = z(dm)
    P["l0.w2"] = z(dm, d); P["l0.b2"] = z(d)
    for j in range(6):
        P["l0.w1"][1, j] = 1.0
        P["l0.b1"][j] = -ts[j]
        P["l0.w2"][j, 2] = slopes[j] - (slopes[j - 1] if j > 0 else 0.0)
    P["l0.b2"][2] = ys[0]                            # g(1/8) = 7

    # unembed reads dim 2
    P["unembed.w"] = z(d, 1); P["unembed.w"][2, 0] = 1.0
    P["unembed.b"] = z(1)
    return P
```

**Parameter count:** 1881 (fixed by the architecture; every tensor is present with its exact shape).

**How it works:**
- BOS has a 1 in embedding dim 0, and no other token does.
- Queries and keys are all zero, so attention is uniform over all L+1 positions. The value is the BOS indicator, and the output projection writes the attention output into residual dim 1. Every position therefore holds a = 1/(L+1).
- The ReLU MLP uses 6 hidden units to build a piecewise-linear function that passes exactly through the points (1/(L+1), L) for L = 1..7. It starts at 7 for a = 1/8, and each hinge at 1/7, 1/6, …, 1/3 adds the slope change between segments.
- The result is written to residual dim 2, and the unembedding reads it. The output equals L exactly, up to float32 rounding of about 1e-5, well inside the 0.5 tolerance.
- I verified this by reasoning only. I did not run it, and there is no feedback.