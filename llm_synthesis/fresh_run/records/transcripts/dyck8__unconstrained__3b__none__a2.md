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


I haven't run this; it is checked by reasoning only.

**`submission.py`**
```python
import numpy as np

CONFIG = dict(n_layers=2, n_heads=1, d_model=5, d_head=1, d_mlp=2, causal=True)

def build_params():
    d, H, dh, dm = 5, 1, 1, 2
    P = {}
    emb = np.zeros((7, d), np.float32)
    emb[1, 0] = 1.0    # '(' -> +1
    emb[2, 0] = -1.0   # ')' -> -1
    P["embed"] = emb
    P["pos_embed"] = np.zeros((8, d), np.float32)

    def layer(i):
        for n in ("wq", "wk", "wv"):
            P[f"l{i}.{n}"] = np.zeros((H, d, dh), np.float32)
        for n in ("bq", "bk", "bv"):
            P[f"l{i}.{n}"] = np.zeros((H, dh), np.float32)
        P[f"l{i}.wo"] = np.zeros((H, dh, d), np.float32)
        P[f"l{i}.bo"] = np.zeros(d, np.float32)
        P[f"l{i}.w1"] = np.zeros((d, dm), np.float32)
        P[f"l{i}.b1"] = np.zeros(dm, np.float32)
        P[f"l{i}.w2"] = np.zeros((dm, d), np.float32)
        P[f"l{i}.b2"] = np.zeros(d, np.float32)

    # Layer 0: uniform causal attention averages the +1/-1 values over positions 0..t
    # (BOS counts as 0), giving a = balance/(t+1) in dim1.
    # MLP: dim2 = [balance < 0] = relu(-8a) - relu(-8a - 1).
    layer(0)
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 1] = 1.0
    P["l0.w1"][1, 0] = -8.0
    P["l0.w1"][1, 1] = -8.0
    P["l0.b1"][:] = [0.0, -1.0]
    P["l0.w2"][0, 2] = 1.0
    P["l0.w2"][1, 2] = -1.0

    # Layer 1: uniform causal attention averages the flag in dim2 over positions 0..t.
    # The result is >= 1/8 iff some prefix has gone negative, else exactly 0.
    # MLP: dim4 = min(1, 8*avg), the "dead" indicator.
    layer(1)
    P["l1.wv"][0, 2, 0] = 1.0
    P["l1.wo"][0, 0, 3] = 1.0
    P["l1.w1"][3, 0] = 8.0
    P["l1.w1"][3, 1] = 8.0
    P["l1.b1"][:] = [0.0, -1.0]
    P["l1.w2"][0, 4] = 1.0
    P["l1.w2"][1, 4] = -1.0

    # Output: class 0 (invalid) logit = dead indicator; class 1 (valid) logit = 0.5.
    uw = np.zeros((d, 2), np.float32)
    uw[4, 0] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.array([0.0, 0.5], np.float32)
    return P
```

**Parameter count:** 197 (embed 35, pos_embed 40, 55 per layer × 2, unembed 12).

**How it works:**
- The cap at 8 never matters, because depth is at most 7 for sequences of length ≤ 7. So a prefix is valid exactly when every prefix balance (#open − #close) is ≥ 0.
- Layer 0 attention has all-zero q/k, so it is a uniform causal average. Tokens 1 and 2 carry values +1 and −1, and BOS carries 0. This gives balance/(t+1) at each position.
- The layer 0 MLP turns "balance < 0" into a 0/1 flag. A negative balance gives a value of at most −1/8, so relu(−8a) − relu(−8a − 1) is exactly 1. Otherwise it is 0.
- Layer 1 attention averages that flag over positions 0..t, which gives at least 1/8 if any prefix went negative and exactly 0 otherwise. Its MLP clips 8·avg to 0/1 to form a "dead" indicator.
- The unembedding sets the class-0 logit to the dead indicator and the class-1 logit to a constant 0.5. Argmax is therefore 1 while the prefix is valid and 0 once it has ever gone invalid. That reproduces the absorbing behaviour, and the output at t depends only on x[1..t].
