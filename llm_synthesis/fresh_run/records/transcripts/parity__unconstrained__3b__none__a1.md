# Construct a transformer that computes `parity` (no training)

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
Token ids: BOS = 0 (always prepended at position 0), input tokens are 1..6 at positions 1..L, with 1 <= L <= 7. Outputs are read at positions 1..L (the output at position 0 is ignored). The output width is d_out = 2.

## The function
Causal streaming task. The output at position t is the parity (0/1) of the number of tokens equal to 1 among x[1..t]. Categorical output over 2 classes: predicted class = argmax of the 2 logits. The model may use causal or bidirectional attention but the output at position t must depend only on x[1..t].

Reference implementation (the ground truth, Python):
```python
def _parity(x):
    c, out = 0, []
    for v in x: c += (v == 1); out.append(c % 2)
    return out
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

CONFIG = dict(n_layers=1, n_heads=1, d_model=4, d_head=1, d_mlp=7, causal=True)

def build_params():
    d, dh, H, dm = 4, 1, 1, 7
    P = {}
    # residual dims: 0 = [token==1], 1 = key score (0 for BOS/token 1, -40 otherwise),
    #                2 = attention output f, 3 = parity score g
    emb = np.zeros((7, d))
    emb[1, 0] = 1.0
    for t in range(2, 7):
        emb[t, 1] = -40.0
    P["embed"] = emb
    P["pos_embed"] = np.zeros((8, d))

    wq = np.zeros((H, d, dh)); bq = np.ones((H, dh))      # constant query = 1
    wk = np.zeros((H, d, dh)); wk[0, 1, 0] = 1.0          # key = dim 1
    wv = np.zeros((H, d, dh)); wv[0, 0, 0] = 1.0          # value = [token==1]
    wo = np.zeros((H, dh, d)); wo[0, 0, 2] = 1.0          # write to dim 2
    P["l0.wq"], P["l0.bq"] = wq, bq
    P["l0.wk"], P["l0.bk"] = wk, np.zeros((H, dh))
    P["l0.wv"], P["l0.bv"] = wv, np.zeros((H, dh))
    P["l0.wo"], P["l0.bo"] = wo, np.zeros(d)

    # After attention, f = c/(c+1), c = number of ones so far (BOS and each 1 get weight 1).
    fc = np.array([c / (c + 1.0) for c in range(8)])
    pc = np.array([c % 2 for c in range(8)], dtype=float)
    slopes = (pc[1:] - pc[:-1]) / (fc[1:] - fc[:-1])
    a = np.concatenate([[slopes[0]], slopes[1:] - slopes[:-1]])
    w1 = np.zeros((d, dm)); b1 = np.zeros(dm)
    w2 = np.zeros((dm, d))
    for j in range(7):
        w1[2, j] = 1.0
        b1[j] = -fc[j]
        w2[j, 3] = a[j]
    P["l0.w1"], P["l0.b1"], P["l0.w2"], P["l0.b2"] = w1, b1, w2, np.zeros(d)

    uw = np.zeros((d, 2)); uw[3, 1] = 1.0
    ub = np.array([0.5, 0.0])
    P["unembed.w"], P["unembed.b"] = uw, ub
    return {k: np.asarray(v, dtype=np.float32) for k, v in P.items()}
```

**Parameter count:** 160.

**How it works:**
- **Attention (one head, causal):** The query is a constant 1. The key score is 0 for BOS and for tokens equal to 1, and −40 for tokens 2..6, so those tokens get essentially zero attention weight. Only BOS and the ones in the prefix are attended to, each with equal weight. The value is the indicator [token==1], which is 0 for BOS. The head's output is therefore f = c/(c+1), where c is the number of ones among x[1..t]. This is a distinct number for each c in 0..7, independent of position.
- **MLP:** Seven ReLU hinge units at thresholds f_j = j/(j+1) build a piecewise-linear function g(f) that interpolates the points (c/(c+1), c mod 2) for c = 0..7. Each hinge's coefficient is the change in slope between consecutive segments.
- **Unembedding:** Logit 1 is g, and logit 0 is the constant 0.5. The argmax is therefore the parity, with a margin of about 0.5 against float32 error of roughly 1e-5 or less. The leakage from the −40 scores is about 1e-17.
- **Causality:** The causal mask means position t only sees x[1..t].

I checked this by reasoning only and have not run it.
