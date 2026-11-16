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


Below is `submission.py`, plus the parameter count and a summary of the construction. I have not run it, and there is no feedback in this setting, so correctness rests on the reasoning below.

```python
import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=64, causal=False)

def build_params():
    d, H, dh, dm = 32, 2, 16, 64
    P = {}
    # residual layout: 0-6 token one-hot (dim 0 = BOS flag), 7 = pos i, 8 = pos^2,
    # 10 = 1/n (n = L+1), 11 = n, 12-18 = attended token copy (output region)
    embed = np.zeros((7, d))
    for t in range(7):
        embed[t, t] = 1.0
    pos = np.zeros((8, d))
    for p in range(8):
        pos[p, 7] = p
        pos[p, 8] = p * p
    P["embed"] = embed
    P["pos_embed"] = pos

    def z(name, shape):
        P[name] = np.zeros(shape)

    # ---- layer 0: uniform attention averages the BOS flag -> 1/n; MLP maps 1/n -> n
    for n in ["wq", "wk"]:
        z(f"l0.{n}", (H, d, dh))
    for n in ["bq", "bk", "bv"]:
        z(f"l0.{n}", (H, dh))
    z("l0.wv", (H, d, dh)); z("l0.wo", (H, dh, d)); z("l0.bo", (d,))
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 10] = 1.0

    ns = [8, 7, 6, 5, 4, 3, 2]
    us = [1.0 / n for n in ns]
    fs = [float(n) for n in ns]
    slopes = [(fs[m + 1] - fs[m]) / (us[m + 1] - us[m]) for m in range(6)]
    z("l0.w1", (d, dm)); z("l0.b1", (dm,)); z("l0.w2", (dm, d)); z("l0.b2", (d,))
    for k in range(6):
        P["l0.w1"][10, k] = 1.0
        P["l0.b1"][k] = -us[k]
        coef = slopes[0] if k == 0 else slopes[k] - slopes[k - 1]
        P["l0.w2"][k, 11] = coef
    P["l0.b2"][11] = fs[0]  # f(u) = 8 + sum of hinges

    # ---- layer 1: head 0 attends to position j = n - i, copies its token one-hot
    C = 10.0
    for n in ["wq", "wk", "wv"]:
        z(f"l1.{n}", (H, d, dh))
    for n in ["bq", "bk", "bv"]:
        z(f"l1.{n}", (H, dh))
    z("l1.wo", (H, dh, d)); z("l1.bo", (d,))
    P["l1.wq"][0, 7, 0] = 1.0      # q0 = i
    P["l1.wq"][0, 11, 1] = 1.0     # q1 = n
    P["l1.bq"][0, 2] = 1.0         # q2 = 1
    P["l1.wk"][0, 7, 0] = -8 * C   # k0 = -8C j
    P["l1.wk"][0, 7, 1] = 8 * C    # k1 = 8C j
    P["l1.wk"][0, 8, 2] = -4 * C   # k2 = -4C j^2
    # score = (q.k)/4 = -2C i j + 2C n j - C j^2  == -C (i+j-n)^2 + (terms in i, n only)
    for c in range(7):
        P["l1.wv"][0, c, c] = 1.0
        P["l1.wo"][0, c, 12 + c] = 1.0
    z("l1.w1", (d, dm)); z("l1.b1", (dm,)); z("l1.w2", (dm, d)); z("l1.b2", (d,))

    uw = np.zeros((d, 7)); 
    for c in range(7):
        uw[12 + c, c] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.zeros(7)
    return {k: np.asarray(v, dtype=np.float32) for k, v in P.items()}
```

**Parameters:** 17,543, since every tensor has the shape the fixed architecture dictates.

**How it works:** I assumed each sequence is fed as exactly L+1 tokens (BOS plus L inputs), with no padding. Under that assumption the non-causal attention sees every position, including BOS.

1. **Layer 0 computes the sequence length.**
   - Head 0 uses zero query and key weights, so its softmax is uniform over the n = L+1 positions.
   - Its value is the BOS indicator, so the head outputs 1/n into residual dimension 10.
   - The MLP maps 1/n to n with six ReLU hinges. This piecewise-linear interpolation is exact at the seven possible values of n (2 to 8). The result goes into dimension 11.
2. **Layer 1 does the reversal.**
   - Head 0 is given the score −C·(i+j−n)², up to terms that depend only on the query position i and n, which do not change the softmax over j. Here i is the query position and j the key position.
   - It is realised bilinearly with query features (i, n, 1) and key features (−8Cj, 8Cj, −4Cj²). With C = 10 and the 1/√16 scaling, this is exact.
   - The score peaks only at j = n − i = L+1−i, which is the reversed position. It lies in 1..L, so BOS is never the target. Every other position scores at least 10 lower in the score units, so the target gets about 1 − 1e-4 of the attention weight.
   - The value and output projections copy that position's token one-hot into dimensions 12–18.
3. **Unembedding.** It reads only dimensions 12–18, so the argmax is the reversed token.

The second head in each attention layer and the layer-1 MLP are all zeros.