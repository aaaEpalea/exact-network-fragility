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
CONSTRAINED SYNTHESIS. You must use exactly this architecture (no other choice is allowed):
```
{"n_layers": 2, "n_heads": 2, "d_model": 32, "d_head": 16, "d_mlp": 128, "causal": true}
```
This fixes every tensor shape (25698 parameters in total).

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

**Deliverable (`submission.py`)**

```python
import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=128, causal=True)

def build_params():
    d, H, dh, dm = 32, 2, 16, 128
    P = {}
    P["embed"] = np.zeros((7, d), np.float32)
    P["embed"][1, 0] = 1.0    # token 1 -> +1
    P["embed"][2, 0] = -1.0   # token 2 -> -1
    P["pos_embed"] = np.zeros((8, d), np.float32)
    for i in range(2):
        P[f"l{i}.wq"] = np.zeros((H, d, dh), np.float32)
        P[f"l{i}.wk"] = np.zeros((H, d, dh), np.float32)
        P[f"l{i}.wv"] = np.zeros((H, d, dh), np.float32)
        P[f"l{i}.bq"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.bk"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.bv"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.wo"] = np.zeros((H, dh, d), np.float32)
        P[f"l{i}.bo"] = np.zeros((d,), np.float32)
        P[f"l{i}.w1"] = np.zeros((d, dm), np.float32)
        P[f"l{i}.b1"] = np.zeros((dm,), np.float32)
        P[f"l{i}.w2"] = np.zeros((dm, d), np.float32)
        P[f"l{i}.b2"] = np.zeros((d,), np.float32)
    # Layer 0, head 0: q=k=0 gives uniform causal attention; copy value dim 0 into residual dim 1
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 1] = 1.0
    # Unembed: logit0 = 0, logit1 = 16*avg - 1
    P["unembed.w"] = np.zeros((d, 2), np.float32)
    P["unembed.w"][1, 1] = 16.0
    P["unembed.b"] = np.array([0.0, -1.0], np.float32)
    return P
```

**Parameter count:** 25698, matching the fixed architecture. I counted this by hand and did not run the code.

**How it works:**
- The length limit of 7 means the saturating counter never hits 0 or 16. It stays within 1..15, so the output at position t is simply "number of 1s so far > number of 2s so far".
- The embedding stores a value of +1 for token 1, −1 for token 2 and 0 for every other token in dimension 0. Positions get no embedding.
- In layer 0, head 0 has all-zero query and key weights, so its scores are all equal. With the causal mask this gives uniform attention over positions 0..t, including BOS, which carries value 0. The head then outputs (n₁−n₂)/(t+1) into residual dimension 1.
- Because n₁−n₂ is an integer and t+1 ≤ 8, the average is either ≥ 1/8 (when n₁>n₂) or ≤ 0 (otherwise).
- The unembedding sets logit0 = 0 and logit1 = 16·avg − 1. That is ≥ 1 when n₁>n₂ and ≤ −1 otherwise, so argmax gives the correct class with a margin of 1 and no ties.
- Head 1, the layer-1 attention and both MLPs have all-zero weights, so they leave the residual stream unchanged.
- I verified this by reasoning only. I did not run the code against the 335,922 sequences.