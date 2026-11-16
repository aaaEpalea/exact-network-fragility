# Construct a transformer that computes `dyck8` (no training)

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
Causal streaming task (valid-prefix check). Token 1 is an opening bracket, token 2 a closing bracket, tokens 3..6 are neutral. The depth starts at 0 and is capped at 8 (token 1 gives min(depth+1, 8)); token 2 gives depth-1; if token 2 arrives at depth 0 the sequence becomes invalid forever (absorbing). The output at position t is 1 while the prefix x[1..t] is still valid, else 0. Categorical output over 2 classes (argmax). The output at position t must depend only on x[1..t].

Reference implementation (the ground truth, Python):
```python
def _dyck(x, N=8):
    dead = N + 1; s = 0; out = []
    for v in x:
        if s == dead: pass
        elif v == 1: s = min(s + 1, N)
        elif v == 2: s = s - 1 if s > 0 else dead
        out.append(int(s != dead))
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
    P["unembed.w"] = np.zeros((d, 2), np.float32)
    P["unembed.b"] = np.zeros((2,), np.float32)

    # token value in dim 0: '(' -> +1, ')' -> -1, others (incl. BOS) -> 0
    P["embed"][1, 0] = 1.0
    P["embed"][2, 0] = -1.0

    # Layer 0, head 0: q=k=0 -> uniform causal attention over positions 0..t.
    # value = dim0, written to dim1: m_t = balance_t / (t+1)
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 1] = 1.0
    # Layer 0 MLP: f_t = relu(-8 * m_t) -> dim2. Balance<0 gives f >= 1; balance>=0 gives f = 0.
    P["l0.w1"][1, 0] = -8.0
    P["l0.w2"][0, 2] = 1.0

    # Layer 1, head 0: uniform causal average of f -> dim3 (0 if never bad, else >= 1/8)
    P["l1.wv"][0, 2, 0] = 1.0
    P["l1.wo"][0, 0, 3] = 1.0

    # Output: logit0 = 0, logit1 = 1 - 16*mean_f  (argmax=1 iff prefix valid)
    P["unembed.w"][3, 1] = -16.0
    P["unembed.b"][1] = 1.0
    return P
```

**Parameter count:** 25,698. Every tensor has the full required shape: 224 for embed, 256 for pos_embed, 12,576 per layer, and 66 for unembed. Most entries are zero.

**How it works.** With L ≤ 7 the depth cap of 8 is never reached. A prefix is therefore valid exactly when every prefix balance (#opens − #closes) is ≥ 0.

1. **Embedding.** Dimension 0 holds +1 for token 1, −1 for token 2, and 0 for everything else, including BOS.
2. **Layer 0 attention.** The query and key weights are zero, so attention is uniform over positions 0..t. The residual stream at position t then holds m_t = balance_t/(t+1), which has the same sign as the balance.
3. **Layer 0 MLP.** One ReLU unit computes f_t = relu(−8·m_t). If balance_t ≥ 0, f_t is 0. If balance_t ≤ −1, then m_t ≤ −1/8 (since t+1 ≤ 8), so f_t ≥ 1.
4. **Layer 1 attention.** Uniform causal attention averages f over positions 0..t. The mean is 0 if no prefix ever went negative, and ≥ 1/8 otherwise.
5. **Unembedding.** Logit 1 = 1 − 16·mean, which is 1 for a valid prefix and ≤ −1 once invalid. Logit 0 is 0, so the argmax is 1 for valid and 0 for invalid. The absorbing behavior follows because any earlier negative prefix keeps the mean ≥ 1/8.

Float rounding in the uniform average can leave tiny nonzero values when the balance is 0. These produce a negligible logit shift (about 1e-6), far from the ±1 margin. I did not run the construction; this is verified by reasoning only.
