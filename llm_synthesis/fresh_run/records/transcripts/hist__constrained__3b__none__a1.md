# Construct a transformer that computes `hist` (no training)

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
For every position i the output is the number of positions j (including i itself) whose token equals x[i] (a number). Numeric output: |y - target| <= 0.5 counts as correct.

Reference implementation (the ground truth, Python):
```python
def _hist(x): return [x.count(v) for v in x]
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
    d, H, dh, dm = 16, 1, 8, 32
    P = {}
    P["embed"] = np.zeros((7, d), np.float32)
    P["pos_embed"] = np.zeros((8, d), np.float32)
    for t in range(1, 7):
        P["embed"][t, t - 1] = 1.0      # one-hot token identity in dims 0..5
    P["embed"][0, 6] = 1.0              # BOS flag in dim 6

    s8 = np.sqrt(8.0)
    C = 20.0
    M = C + np.log(2.0)
    wq = np.zeros((H, d, dh)); wk = np.zeros((H, d, dh)); wv = np.zeros((H, d, dh))
    bq = np.zeros((H, dh)); bk = np.zeros((H, dh)); bv = np.zeros((H, dh))
    for t in range(6):
        wq[0, t, t] = s8 * M
        wk[0, t, t] = 1.0
    bq[0, 6] = s8 * C                   # constant query component
    wk[0, 6, 6] = 1.0                   # BOS key flag
    wv[0, 6, 0] = 1.0                   # value = BOS flag
    wo = np.zeros((H, dh, d)); wo[0, 0, 8] = 1.0   # write attention output to dim 8
    P.update({"l0.wq": wq, "l0.wk": wk, "l0.wv": wv, "l0.bq": bq, "l0.bk": bk,
              "l0.bv": bv, "l0.wo": wo, "l0.bo": np.zeros(d)})

    # a(c) = 1/(2c+1); piecewise-linear decode a -> c through nodes c=1..7
    a = np.array([1.0 / (2 * c + 1) for c in range(1, 8)])
    slopes = np.array([1.0 / (a[k] - a[k + 1]) for k in range(6)])
    w = np.zeros(6)
    w[0] = slopes[0]
    for k in range(1, 6):
        w[k] = slopes[k] - slopes[k - 1]
    w1 = np.zeros((d, dm)); b1 = np.zeros(dm); w2 = np.zeros((dm, d)); b2 = np.zeros(d)
    for k in range(6):
        w1[8, k] = -1.0
        b1[k] = a[k]
        w2[k, 9] = w[k]
    b2[9] = 1.0
    P.update({"l0.w1": w1, "l0.b1": b1, "l0.w2": w2, "l0.b2": b2})

    uw = np.zeros((d, 1)); uw[9, 0] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.zeros(1)
    return {k: np.asarray(v, np.float32) for k, v in P.items()}
```

**Parameter count:** 1881 (embed 112 + pos_embed 128 + attention 3·128 + 24 + 8 qkv/biases... all tensors fixed by the architecture, including wo 128 + bo 16, MLP 512+32+512+16, unembed 16+1).

**How it works:**
- Tokens are embedded as one-hot vectors in dims 0–5, and BOS has its own flag in dim 6.
- The single attention head gives a query a large score (≈20.69 in softmax units) on keys with the same token, including itself. BOS gets a fixed score of 20, and all other keys score 0, so their weight is about e⁻²⁰ and negligible.
- The value is the BOS flag, so the attention output is the weight on BOS: a = e²⁰ / (c·e^(20+ln2) + e²⁰) = 1/(2c+1), where c is the number of matching positions. This is a strictly monotone function of the count c = 1..7.
- The attention output is written to residual dim 8.
- The MLP uses 6 ReLUs, relu(a_k − a) for k = 1..6 with a_k = 1/(2k+1). Their weights make a piecewise-linear interpolation that passes exactly through (a_c, c) for c = 1..7, and the result goes to dim 9, which the unembedding reads out.
- The decode is exact at every node, and float32 error is around 1e-5 even in the steepest segment, so every output is far inside the ±0.5 tolerance.
