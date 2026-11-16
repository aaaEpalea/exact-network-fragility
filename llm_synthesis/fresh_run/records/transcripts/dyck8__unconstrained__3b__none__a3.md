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


Here is `submission.py`. I did not run it. I checked the construction by hand against the reference function.

```python
import numpy as np

CONFIG = dict(n_layers=2, n_heads=1, d_model=12, d_head=1, d_mlp=16, causal=True)

def build_params():
    d, H, dh, dm = 12, 1, 1, 16
    P = {}
    # dims: 0 = a(token) (+1 open, -1 close, 0 else); 1..8 = one-hot position;
    #       9 = running mean of a; 10 = bad flag; 11 = mean of bad flag
    E = np.zeros((7, d), np.float32)
    E[1, 0] = 1.0
    E[2, 0] = -1.0
    P["embed"] = E
    PE = np.zeros((8, d), np.float32)
    for t in range(8):
        PE[t, 1 + t] = 1.0
    P["pos_embed"] = PE

    K = 10.0
    # ---- layer 0: uniform causal attention averages a over positions 0..t
    P["l0.wq"] = np.zeros((H, d, dh), np.float32)
    P["l0.wk"] = np.zeros((H, d, dh), np.float32)
    wv = np.zeros((H, d, dh), np.float32); wv[0, 0, 0] = 1.0
    P["l0.wv"] = wv
    P["l0.bq"] = np.zeros((H, dh), np.float32)
    P["l0.bk"] = np.zeros((H, dh), np.float32)
    P["l0.bv"] = np.zeros((H, dh), np.float32)
    wo = np.zeros((H, dh, d), np.float32); wo[0, 0, 9] = 1.0
    P["l0.wo"] = wo
    P["l0.bo"] = np.zeros(d, np.float32)
    # MLP: bad_t = relu(-B) - relu(-B-1), B = (t+1)*avg, gated by position one-hot
    w1 = np.zeros((d, dm), np.float32)
    b1 = np.zeros(dm, np.float32)
    w2 = np.zeros((dm, d), np.float32)
    for t in range(1, 8):
        ia, ib = t - 1, 7 + t - 1
        w1[9, ia] = -(t + 1.0); w1[1 + t, ia] = K; b1[ia] = -K
        w1[9, ib] = -(t + 1.0); w1[1 + t, ib] = K; b1[ib] = -K - 1.0
        w2[ia, 10] = 1.0
        w2[ib, 10] = -1.0
    P["l0.w1"], P["l0.b1"], P["l0.w2"] = w1, b1, w2
    P["l0.b2"] = np.zeros(d, np.float32)

    # ---- layer 1: uniform causal attention averages bad flag over 0..t
    P["l1.wq"] = np.zeros((H, d, dh), np.float32)
    P["l1.wk"] = np.zeros((H, d, dh), np.float32)
    wv = np.zeros((H, d, dh), np.float32); wv[0, 10, 0] = 1.0
    P["l1.wv"] = wv
    P["l1.bq"] = np.zeros((H, dh), np.float32)
    P["l1.bk"] = np.zeros((H, dh), np.float32)
    P["l1.bv"] = np.zeros((H, dh), np.float32)
    wo = np.zeros((H, dh, d), np.float32); wo[0, 0, 11] = 1.0
    P["l1.wo"] = wo
    P["l1.bo"] = np.zeros(d, np.float32)
    P["l1.w1"] = np.zeros((d, dm), np.float32)
    P["l1.b1"] = np.zeros(dm, np.float32)
    P["l1.w2"] = np.zeros((dm, d), np.float32)
    P["l1.b2"] = np.zeros(d, np.float32)

    # ---- unembed: class0 = 16*mean_bad, class1 = 1
    uw = np.zeros((d, 2), np.float32); uw[11, 0] = 16.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.array([0.0, 1.0], np.float32)
    return P
```

**Parameter count:** 1156 (embed 84, pos_embed 96, two layers of 475 each, unembed 26), well under the 60,000 limit.

**How it works:**
- **Reduction.** Since L ≤ 7, depth never reaches the cap of 8. A prefix is valid exactly when every prefix's balance B_j (opens minus closes) is ≥ 0. So output 0 at t means some j ≤ t has B_j < 0.
- **Layer 0, attention.** The query and key weights are zero, so causal attention is uniform over positions 0..t. The value is a(token), which is +1 for an open bracket, −1 for a close bracket and 0 otherwise, and BOS contributes 0. The head writes avg = B_t/(t+1) into dim 9.
- **Layer 0, MLP.** The position one-hot (dims 1..8) gates a pair of units for each position t. Unit a_t computes relu(−(t+1)·avg + 10·onehot_t − 10), which equals relu(−B_t) at position t and is 0 elsewhere. Unit b_t is the same with an extra −1 in the bias. Their difference a_t − b_t equals 1 if B_t ≤ −1 and 0 otherwise. It is written into dim 10 as the "bad" flag.
- **Layer 1.** Uniform causal attention averages the bad flags over positions 0..t. The result is 0 if no earlier position was bad, and at least 1/8 otherwise, because at most 8 positions are averaged.
- **Unembed.** Class 0 gets 16·mean_bad and class 1 gets the constant 1. With no bad flag, class 1 wins (1 > 0). With any bad flag, class 0 gets at least 2, so it wins (2 > 1). This makes the invalid state absorbing.
- **Layer 1 MLP.** It is zeroed and exists only to satisfy the required shapes.
- **Numerics.** Float rounding in avg·(t+1) is around 1e-7, which is far below the margins of 1 and 1/8.
