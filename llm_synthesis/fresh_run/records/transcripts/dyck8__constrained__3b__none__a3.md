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

## Deliverable (`submission.py`)

```python
import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=128, causal=True)

def build_params():
    d, H, dh, dm = 32, 2, 16, 128
    P = {}
    P["embed"] = np.zeros((7, d), np.float32)
    P["embed"][1, 0] = 1.0          # '(' -> dim0
    P["embed"][2, 1] = 1.0          # ')' -> dim1
    P["pos_embed"] = np.zeros((8, d), np.float32)
    for i in range(2):
        for n in ("wq", "wk"):
            P[f"l{i}.{n}"] = np.zeros((H, d, dh), np.float32)
        for n in ("bq", "bk", "bv"):
            P[f"l{i}.{n}"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.wv"] = np.zeros((H, d, dh), np.float32)
        P[f"l{i}.wo"] = np.zeros((H, dh, d), np.float32)
        P[f"l{i}.bo"] = np.zeros(d, np.float32)
        P[f"l{i}.w1"] = np.zeros((d, dm), np.float32)
        P[f"l{i}.b1"] = np.zeros(dm, np.float32)
        P[f"l{i}.w2"] = np.zeros((dm, d), np.float32)
        P[f"l{i}.b2"] = np.zeros(d, np.float32)
    # Layer 0, head 0: uniform causal average of (is_open - is_close) -> dim2
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wv"][0, 1, 0] = -1.0
    P["l0.wo"][0, 0, 2] = 1.0
    # Layer 0 MLP: bad = clip(-8*o, 0, 1) -> dim3, where o = balance/(t+1)
    P["l0.w1"][2, 0] = -8.0
    P["l0.w1"][2, 1] = -8.0
    P["l0.b1"][1] = -1.0
    P["l0.w2"][0, 3] = 1.0
    P["l0.w2"][1, 3] = -1.0
    # Layer 1, head 0: uniform causal average of bad flags -> dim4
    P["l1.wv"][0, 3, 0] = 1.0
    P["l1.wo"][0, 0, 4] = 1.0
    # Unembed: class 0 = dead, class 1 = valid
    P["unembed.w"] = np.zeros((d, 2), np.float32)
    P["unembed.w"][4, 0] = 8.0
    P["unembed.b"] = np.array([-0.5, 0.0], np.float32)
    return P
```

**Parameter count:** 25,698. Every tensor has the required shape, and the layout matches the stated total.

**How it works:**
- Since L ≤ 7, the depth cap of 8 can never be reached. A prefix is therefore valid exactly when, at every position j up to t, #opens − #closes ≥ 0.
- **Layer 1 attention:** the query and key weights are zero, so causal attention is uniform over positions 0..t. BOS embeds to zero, so it only dilutes the average. The head averages (is_open − is_close), giving o = balance_t / (t+1) in dim 2.
- **Layer 1 MLP:** it computes bad_t = relu(−8o) − relu(−8o − 1). If the balance is negative, then −8o = 8|b|/(t+1) ≥ 1 (since t+1 ≤ 8), so bad = 1. If the balance is ≥ 0, then bad = 0. This puts a 0/1 flag "balance negative at this position" in dim 3.
- **Layer 2 attention:** a second uniform causal average of the flags gives m = (#bad positions)/(t+1). This is 0 if the prefix is valid, and at least 1/8 if any prefix has gone negative.
- **Unembed:** logit_dead = 8m − 0.5 and logit_valid = 0. With m = 0 valid wins by 0.5, and with m ≥ 1/8 dead wins by at least 0.5, so argmax matches the reference. Position embeddings, head 1 and the layer 2 MLP are all zero.

I did not run the network against the 335,922 sequences; I checked the construction by reasoning only.
