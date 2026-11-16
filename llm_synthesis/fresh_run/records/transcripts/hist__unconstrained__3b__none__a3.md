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

The construction is below. I have not run it; I checked it by hand.

```python
import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=9, d_head=7, d_mlp=6, causal=False)

def build_params():
    d, dh, dm, H = 9, 7, 6, 1
    P = {}
    # token one-hot in dims 0..6 (dim 0 = BOS); dim 7 = attention result a; dim 8 = output
    emb = np.zeros((7, d), np.float32)
    for t in range(7):
        emb[t, t] = 1.0
    P["embed"] = emb
    P["pos_embed"] = np.zeros((8, d), np.float32)

    A = 30.0 * np.sqrt(dh)          # so that (q.k)/sqrt(dh) = 30
    wq = np.zeros((H, d, dh), np.float32)
    wk = np.zeros((H, d, dh), np.float32)
    wv = np.zeros((H, d, dh), np.float32)
    for i in range(7):
        wq[0, i, i] = A
        wk[0, i, i] = 1.0
    wv[0, 0, 0] = 1.0               # value = BOS indicator
    bq = np.zeros((H, dh), np.float32)
    bq[0, 0] = A                    # every query gets +30 on BOS key
    P["l0.wq"] = wq; P["l0.wk"] = wk; P["l0.wv"] = wv
    P["l0.bq"] = bq
    P["l0.bk"] = np.zeros((H, dh), np.float32)
    P["l0.bv"] = np.zeros((H, dh), np.float32)
    wo = np.zeros((H, dh, d), np.float32)
    wo[0, 0, 7] = 1.0               # write a into dim 7
    P["l0.wo"] = wo
    P["l0.bo"] = np.zeros(d, np.float32)

    # MLP: piecewise-linear map a=1/(c+1) -> c, exact at the nodes
    a_nodes = [1.0 / (8 - k) for k in range(7)]   # 1/8 .. 1/2
    c_nodes = [7.0 - k for k in range(7)]         # 7 .. 1
    slopes = [(c_nodes[k+1] - c_nodes[k]) / (a_nodes[k+1] - a_nodes[k]) for k in range(6)]
    w1 = np.zeros((d, dm), np.float32)
    b1 = np.zeros(dm, np.float32)
    w2 = np.zeros((dm, d), np.float32)
    for k in range(6):
        w1[7, k] = 1.0
        b1[k] = -a_nodes[k]
        w2[k, 8] = slopes[k] - (slopes[k-1] if k > 0 else 0.0)
    P["l0.w1"] = w1; P["l0.b1"] = b1; P["l0.w2"] = w2
    P["l0.b2"] = np.zeros(d, np.float32)

    uw = np.zeros((d, 1), np.float32)
    uw[8, 0] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.array([7.0], np.float32)   # f(a_0) = 7
    return P
```

**Parameter count:** 63 + 72 + 3·63 + 3·7 + 63 + 9 + 54 + 6 + 54 + 9 + 9 + 1 = 559 parameters.

**How it works:**
- **Attention (one layer, one head, non-causal):** The query for token t scores 30 on every key with the same token, including itself. It also scores 30 on the BOS key, through a constant query bias on dim 0 that matches only the BOS key. Every other key scores 0, so its weight is about e^-30 and negligible.
- **Attention weights:** The BOS key and the c matching keys share nearly all the weight equally, so the BOS weight is 1/(c+1).
- **Value and output:** The value is the BOS indicator, so the head writes a = 1/(c+1) into dim 7 for every token position. This holds for every L, because a depends only on c.
- **MLP:** Six ReLU hinges at a = 1/8, ..., 1/3 build a piecewise-linear function through the points (1/(c+1), c) for c = 1..7. The output bias supplies the value 7 at a = 1/8. The function is exact at every possible a, and the unembedding reads the result from dim 8. The correct counts are integers and the tolerance is 0.5, so float32 error in a and in the interpolation is far below what could cause a miss.