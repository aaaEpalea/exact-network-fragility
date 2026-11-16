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
    d, dh, dm, H = 16, 8, 32, 1
    P = {}
    z = lambda *s: np.zeros(s, np.float64)
    embed = z(7, d)
    for t in range(7):
        embed[t, t] = 1.0            # one-hot token in dims 0..6 (dim 0 = BOS)
    P["embed"] = embed
    P["pos_embed"] = z(8, d)

    S = np.sqrt(dh)
    A = 10.0                          # effective match score
    wq, wk, wv, wo = z(H, d, dh), z(H, d, dh), z(H, d, dh), z(H, dh, d)
    bq, bk, bv = z(H, dh), z(H, dh), z(H, dh)
    for t in range(1, 7):
        wq[0, t, t - 1] = A * S       # query: scaled one-hot of own token
        wk[0, t, t - 1] = 1.0         # key: one-hot of own token
    bq[0, 6] = 1.0                    # constant query feature
    wk[0, 0, 6] = A * S               # BOS key gets effective score A
    wv[0, 0, 0] = 1.0                 # value: BOS indicator
    wo[0, 0, 7] = 1.0                 # write attention-to-BOS weight into dim 7
    P.update({"l0.wq": wq, "l0.wk": wk, "l0.wv": wv, "l0.bq": bq, "l0.bk": bk,
              "l0.bv": bv, "l0.wo": wo, "l0.bo": z(d)})

    # w ~= 1/(c+1); decode c = 1/w - 1 by piecewise-linear interpolation (exact at knots)
    us = [1.0 / (c + 1) for c in range(7, 0, -1)]   # 1/8 ... 1/2
    fs = list(range(7, 0, -1))                      # 7 ... 1
    m = [(fs[i + 1] - fs[i]) / (us[i + 1] - us[i]) for i in range(6)]
    w1, b1, w2 = z(d, dm), z(dm), z(dm, d)
    uw, ub = z(d, 1), 0.0
    uw = z(d, 1)
    uw[7, 0] = m[0]
    for j in range(1, 6):
        k = j - 1
        w1[7, k] = 1.0
        b1[k] = -us[j]
        w2[k, 8 + k] = 1.0
        uw[8 + k, 0] = m[j] - m[j - 1]
    P.update({"l0.w1": w1, "l0.b1": b1, "l0.w2": w2, "l0.b2": z(d)})
    P["unembed.w"] = uw
    P["unembed.b"] = np.array([fs[0] - m[0] * us[0]])
    return {k: np.asarray(v, np.float32) for k, v in P.items()}
```

**Parameter count:** 1881 (the fixed architecture).

**How it works:**
- **Embedding:** each token is a one-hot vector in dims 0..6, with dim 0 marking BOS.
- **Attention:** the query is the token one-hot scaled so that a matching token scores 10 after the 1/√d_head scaling. Non-matching tokens score 0. BOS gets score 10 through a constant query feature (from `bq`) and a BOS-only key feature.
- **Attention output:** the value is the BOS indicator, so the head writes the attention weight on BOS into dim 7. That weight is w = e^10 / (e^10·(c+1) + (L−c)) ≈ 1/(c+1), where c is the number of equal tokens including the token itself. The error is about 1e-4 at most.
- **MLP and unembed:** the MLP and unembed turn w into c = 1/w − 1 with a piecewise-linear function. A linear term in w plus 5 ReLU hinges at w = 1/7, 1/6, 1/5, 1/4, 1/3 interpolate through the knots (1/(c+1), c) for c = 1..7. The interpolation is exact at the knots, so the output is within about 1e-3 of the integer count, well inside the 0.5 tolerance.

I did not run this against the 335,922 sequences; correctness rests on the construction above.
