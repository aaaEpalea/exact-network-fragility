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


**Deliverable: `submission.py`**

```python
import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=3, d_head=1, d_mlp=6, causal=False)

def build_params():
    d, H, dh, dm = 3, 1, 1, 6
    f = np.float32
    P = {}
    # dim0 = BOS indicator, dim1 = mean of that indicator (u = 1/(L+1)), dim2 = output
    embed = np.zeros((7, d), f); embed[0, 0] = 1.0
    P["embed"] = embed
    P["pos_embed"] = np.zeros((8, d), f)

    # attention: zero q/k -> all scores 0 -> exactly uniform attention over L+1 positions
    P["l0.wq"] = np.zeros((H, d, dh), f); P["l0.bq"] = np.zeros((H, dh), f)
    P["l0.wk"] = np.zeros((H, d, dh), f); P["l0.bk"] = np.zeros((H, dh), f)
    wv = np.zeros((H, d, dh), f); wv[0, 0, 0] = 1.0       # v = BOS indicator
    P["l0.wv"] = wv; P["l0.bv"] = np.zeros((H, dh), f)
    wo = np.zeros((H, dh, d), f); wo[0, 0, 1] = 1.0       # write the mean into dim1
    P["l0.wo"] = wo; P["l0.bo"] = np.zeros(d, f)

    # MLP: piecewise-linear interpolation of f(1/(L+1)) = L, nodes u_k = 1/(8-k), k=0..6, values 7-k
    u = [1.0 / (8 - k) for k in range(7)]
    val = [7.0 - k for k in range(7)]
    s = [(val[j + 1] - val[j]) / (u[j + 1] - u[j]) for j in range(6)]
    w1 = np.zeros((d, dm), f); b1 = np.zeros(dm, f); w2 = np.zeros((dm, d), f)
    for j in range(6):
        w1[1, j] = 1.0
        b1[j] = -u[j]
        w2[j, 2] = s[0] if j == 0 else s[j] - s[j - 1]
    b2 = np.zeros(d, f); b2[2] = val[0]
    P["l0.w1"], P["l0.b1"], P["l0.w2"], P["l0.b2"] = w1, b1, w2, b2

    uw = np.zeros((d, 1), f); uw[2, 0] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.zeros(1, f)
    return P
```

**Parameter count:** 112 (embed 21, pos_embed 24, attention 18, MLP 45, unembed 4).

**How it works:**
- **Attention:** The model is non-causal with one head. The query and key weights and biases are all zero, so every score is 0 and softmax is exactly uniform over all L+1 positions, including BOS.
- **Value:** The value is the BOS indicator (embedding dim 0, which is 1 only for token 0). Every position therefore receives u = 1/(L+1) in dim 1.
- **MLP:** The MLP computes L = 1/u − 1 as a piecewise-linear function. It has six ReLU units, `relu(u − 1/(8−j))` for j = 0..5. Their output coefficients are the slope of the first segment, then the slope changes at each node. The bias is 7, the value at u = 1/8.
- **Exactness:** The interpolation passes exactly through the seven points (1/(L+1), L) for L = 1..7, so the output equals L up to float32 rounding, far inside the ±0.5 tolerance. The unembedding reads dim 2.

I did not run this; the correctness argument above is from reasoning only.