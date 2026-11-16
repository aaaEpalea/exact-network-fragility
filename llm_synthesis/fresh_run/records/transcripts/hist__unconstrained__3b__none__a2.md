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


**Deliverable: `submission.py`** (not run or tested; I checked it by hand only).

```python
import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=9, d_head=7, d_mlp=6, causal=False)

def build_params():
    d, H, dh, dm = 9, 1, 7, 6
    P = {}
    # embeddings: token t (1..6) -> one-hot at dim t-1 ; BOS -> dim 6
    E = np.zeros((7, d), np.float32)
    for t in range(1, 7):
        E[t, t - 1] = 1.0
    E[0, 6] = 1.0
    P["embed"] = E
    P["pos_embed"] = np.zeros((8, d), np.float32)

    A = 20.0 * np.sqrt(dh)          # score gap of 20 after 1/sqrt(dh)
    wq = np.zeros((H, d, dh), np.float32)
    wk = np.zeros((H, d, dh), np.float32)
    wv = np.zeros((H, d, dh), np.float32)
    for i in range(6):
        wq[0, i, i] = A             # query = A * onehot(own token) ...
    for i in range(7):
        wk[0, i, i] = 1.0           # key = [onehot(token), isBOS]
    bq = np.zeros((H, dh), np.float32)
    bq[0, 6] = A                    # ... plus A on the BOS dim (every non-BOS query likes BOS)
    wv[0, 6, 0] = 1.0               # value = BOS indicator
    wo = np.zeros((H, dh, d), np.float32)
    wo[0, 0, 7] = 1.0               # write attention output (= weight on BOS) to dim 7
    P["l0.wq"], P["l0.wk"], P["l0.wv"] = wq, wk, wv
    P["l0.bq"] = bq
    P["l0.bk"] = np.zeros((H, dh), np.float32)
    P["l0.bv"] = np.zeros((H, dh), np.float32)
    P["l0.wo"] = wo
    P["l0.bo"] = np.zeros(d, np.float32)

    # MLP: piecewise-linear interpolation of c = 1/w - 1 through knots w = 1/8..1/2
    ws = np.array([1.0 / k for k in range(8, 1, -1)])   # 1/8 ... 1/2 (increasing)
    vals = np.array([k - 1 for k in range(8, 1, -1)], float)  # 7 ... 1
    slopes = (vals[1:] - vals[:-1]) / (ws[1:] - ws[:-1])      # 6 segments
    coef = np.concatenate([[slopes[0]], slopes[1:] - slopes[:-1]])
    w1 = np.zeros((d, dm), np.float32)
    b1 = np.zeros(dm, np.float32)
    w2 = np.zeros((dm, d), np.float32)
    for m in range(6):
        w1[7, m] = 1.0
        b1[m] = -ws[m]
        w2[m, 8] = coef[m]
    b2 = np.zeros(d, np.float32)
    b2[8] = vals[0]
    P["l0.w1"], P["l0.b1"], P["l0.w2"], P["l0.b2"] = w1, b1, w2, b2

    uw = np.zeros((d, 1), np.float32)
    uw[8, 0] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.zeros(1, np.float32)
    return P
```

**Parameter count:** 549 (embed 63, pos_embed 72, wq/wk/wv 3×63, bq/bk/bv 3×7, wo 63, bo 9, w1 54, b1 6, w2 54, b2 9, unembed 10).

**How it works:**
- The model is non-causal, so every position sees the whole sequence, BOS included. It has one layer, one head and a ReLU MLP.
- Each token is embedded as a one-hot vector in dims 0–5, and BOS sets dim 6.
- Attention scores are about 20 for keys that match the query's token and for the BOS key. Every other key scores 0, which is a gap of 20 in logits. A query at a token position therefore attends uniformly (up to about e⁻²⁰) over its c equal-token positions plus BOS. Each of those gets weight 1/(c+1).
- The head's value is the BOS indicator, so its output is w = 1/(c+1) for c in 1..7. That is exactly the weight on BOS, and it is written to dim 7.
- The MLP computes c = 1/w − 1 by piecewise-linear interpolation. It uses six ReLU units with knots at w = 1/8, …, 1/3, and a seventh knot at 1/2 is reached by the last segment. The result is exact at all seven reachable values of w and goes to the output through the unembedding.
- The error comes only from the e⁻²⁰ softmax leakage, which is far inside the 0.5 tolerance.

**Assumption I could not check:** each sequence is run at its own length T = L+1, with no padding. If shorter sequences were padded inside a longer batch, the non-causal attention would also see the pad positions and the counts would be wrong.
