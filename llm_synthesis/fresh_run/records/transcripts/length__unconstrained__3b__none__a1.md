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

**Deliverable `submission.py`**

```python
import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=3, d_head=1, d_mlp=6, causal=False)

def build_params():
    d, H, dh, dm = 3, 1, 1, 6
    P = {}
    # dim0 = BOS indicator, dim1 = attention average f, dim2 = output
    embed = np.zeros((7, d)); embed[0, 0] = 1.0
    P["embed"] = embed
    P["pos_embed"] = np.zeros((8, d))
    # attention: zero q,k -> uniform weights over all positions 0..L
    P["l0.wq"] = np.zeros((H, d, dh)); P["l0.bq"] = np.zeros((H, dh))
    P["l0.wk"] = np.zeros((H, d, dh)); P["l0.bk"] = np.zeros((H, dh))
    wv = np.zeros((H, d, dh)); wv[0, 0, 0] = 1.0       # value = BOS indicator
    P["l0.wv"] = wv; P["l0.bv"] = np.zeros((H, dh))
    wo = np.zeros((H, dh, d)); wo[0, 0, 1] = 1.0       # write f = 1/(L+1) into dim1
    P["l0.wo"] = wo; P["l0.bo"] = np.zeros(d)
    # MLP: piecewise-linear interpolation of f=1/(n+1) -> n, n=1..7
    f = np.array([1.0 / (n + 1) for n in range(7, 0, -1)])   # 1/8 ... 1/2 (increasing)
    v = np.array([float(n) for n in range(7, 0, -1)])        # 7 ... 1
    slopes = (v[1:] - v[:-1]) / (f[1:] - f[:-1])             # 6 segments
    w1 = np.zeros((d, dm)); b1 = np.zeros(dm); w2 = np.zeros((dm, d))
    prev = 0.0
    for j in range(6):
        w1[1, j] = 1.0
        b1[j] = -f[j]
        w2[j, 2] = slopes[j] - prev
        prev = slopes[j]
    b2 = np.zeros(d); b2[2] = 7.0
    P["l0.w1"] = w1; P["l0.b1"] = b1; P["l0.w2"] = w2; P["l0.b2"] = b2
    uw = np.zeros((d, 1)); uw[2, 0] = 1.0
    P["unembed.w"] = uw; P["unembed.b"] = np.zeros(1)
    return {k: np.asarray(a, dtype=np.float32) for k, a in P.items()}
```

**Parameter count:** 21 (embed) + 24 (pos_embed) + 18 (attention: wq, wk, wv, bq, bk, bv, wo, bo, 3 each) + 18 + 6 + 18 + 3 (MLP: w1, b1, w2, b2) + 4 (unembed) = **112 parameters**.

**How it works:**
- The model is non-causal, so each position attends to all L+1 tokens (BOS plus the L inputs). The queries and keys are zero, so the attention is exactly uniform.
- The value is the BOS indicator, so every position receives f = 1/(L+1). This is one of 1/2, 1/3, …, 1/8.
- The ReLU MLP is a 6-knot piecewise-linear function through the points (1/(n+1), n) for n = 1..7. Its knots are at 1/8, 1/7, …, 1/3, with slope changes taken from the segment slopes, and the bias 7 is added at the start.
- That function is exact at every possible f. The output is therefore L to within float32 rounding, well inside the 0.5 tolerance, at every position (including BOS) for every sequence of length 1..7.