# Construct a transformer that computes `majority8` (no training)

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
Causal streaming task. A counter starts at 8 and is saturated to the range [0,16]; token 1 increments it, token 2 decrements it, all other tokens leave it unchanged. The output at position t is 1 if the counter after x[1..t] is > 8 (more 1s than 2s so far), else 0. Categorical output over 2 classes (argmax). The output at position t must depend only on x[1..t].

Reference implementation (the ground truth, Python):
```python
def _majority(x, N=8):
    n = 2 * N + 1; s = N; out = []
    for v in x: s = min(s + 1, n - 1) if v == 1 else max(s - 1, 0) if v == 2 else s; out.append(int(s > N))
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


## Deliverable: `submission.py`

```python
import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=3, d_head=1, d_mlp=1, causal=True)

def build_params():
    d, H, dh, dm = 3, 1, 1, 1
    f = lambda *s: np.zeros(s, dtype=np.float32)
    P = {}
    # dim0 of the embedding holds the signed vote: +1 for token 1, -1 for token 2, 0 otherwise (BOS = token 0 -> 0)
    P["embed"] = f(7, d)
    P["embed"][1, 0] = 1.0
    P["embed"][2, 0] = -1.0
    P["pos_embed"] = f(8, d)

    # Layer 0, one head: zero q/k => uniform causal attention; value = dim0, written to dim1
    P["l0.wq"] = f(H, d, dh); P["l0.bq"] = f(H, dh)
    P["l0.wk"] = f(H, d, dh); P["l0.bk"] = f(H, dh)
    P["l0.wv"] = f(H, d, dh); P["l0.wv"][0, 0, 0] = 1.0
    P["l0.bv"] = f(H, dh)
    P["l0.wo"] = f(H, dh, d); P["l0.wo"][0, 0, 1] = 1.0
    P["l0.bo"] = f(d)

    # MLP: h = relu(8*avg - 0.5), written to dim2
    P["l0.w1"] = f(d, dm); P["l0.w1"][1, 0] = 8.0
    P["l0.b1"] = f(dm);    P["l0.b1"][0] = -0.5
    P["l0.w2"] = f(dm, d); P["l0.w2"][0, 2] = 1.0
    P["l0.b2"] = f(d)

    # Output: logit0 = 0.25 (constant), logit1 = h
    P["unembed.w"] = f(d, 2); P["unembed.w"][2, 1] = 1.0
    P["unembed.b"] = np.array([0.25, 0.0], dtype=np.float32)
    return P
```

**Parameter count:** 81 (embed 21, pos_embed 24, layer 36, unembed 8 = 81 total, including zeros).

**How it works:**
- With at most 7 input tokens, the counter stays within 1..15 and never saturates. So the output at position t is 1 exactly when S > 0, where S = (#1s − #2s) among x[1..t].
- Each token's embedding carries a vote: +1 for token 1, −1 for token 2, 0 for everything else (including BOS).
- The single causal attention head has all-zero query and key weights, so every score is 0. Masked scores are −1e9 and underflow to exactly 0 weight. Position t therefore averages the votes over positions 0..t. The value head copies the vote and `wo` writes it to dim1, giving avg = S/(t+1), where t+1 ≤ 8.
- The ReLU MLP computes h = relu(8·avg − 0.5). If S ≥ 1, then 8·avg ≥ 1 and h ≥ 0.5. If S ≤ 0, then h = 0.
- The unembedding sets logit0 = 0.25 (constant bias) and logit1 = h. Argmax is therefore class 1 exactly when S > 0, with a margin of at least 0.25. The output at position t depends only on x[1..t] because the attention is causal.

I verified this by reasoning only and have not run it.