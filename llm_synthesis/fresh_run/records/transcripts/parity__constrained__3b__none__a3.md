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
CONSTRAINED SYNTHESIS. You must use exactly this architecture (no other choice is allowed):
```
{"n_layers": 2, "n_heads": 2, "d_model": 16, "d_head": 8, "d_mlp": 64, "causal": true}
```
This fixes every tensor shape (6706 parameters in total).

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

CONFIG = dict(n_layers=2, n_heads=2, d_model=16, d_head=8, d_mlp=64, causal=True)

def build_params():
    d, H, dh, dm = 16, 2, 8, 64
    P = {}
    # residual layout: 0 = [tok==1], 1 = f (fraction of ones), 2..8 = one-hot position t (t=1..7), 9 = parity value p
    embed = np.zeros((7, d), np.float32)
    embed[1, 0] = 1.0
    pos = np.zeros((8, d), np.float32)
    for t in range(1, 8):
        pos[t, 1 + t] = 1.0
    P["embed"], P["pos_embed"] = embed, pos

    for i in range(2):
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

    # Layer 0, head 0: q=k=0 -> uniform causal attention over positions 0..t;
    # value = [tok==1]; output written to dim 1 => f = c/(t+1)
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 1] = 1.0

    # Layer 0 MLP: for each position t, t neurons forming a zigzag in g=(t+1)f=c with p(c)=c%2
    n = 0
    for t in range(1, 8):
        for k in range(t):
            P["l0.w1"][1, n] = float(t + 1)        # (t+1)*f = c
            P["l0.w1"][1 + t, n] = 100.0           # gate: only active at position t
            P["l0.b1"][n] = -float(k) - 100.0
            P["l0.w2"][n, 9] = 1.0 if k == 0 else (2.0 if k % 2 == 0 else -2.0)
            n += 1
    # Layer 1 stays all-zero (identity).

    P["unembed.w"] = np.zeros((d, 2), np.float32)
    P["unembed.w"][9, 1] = 1.0
    P["unembed.b"] = np.array([0.5, 0.0], np.float32)
    return P
```

**Parameter count:** 6706 (the fixed architecture; every tensor has the required shape).

**How it works:** I did not run this against the 335,922 sequences. I checked the construction by reasoning only.

1. **Counting the ones.** In layer 0, head 0 has zero query and key weights, so causal attention is uniform over positions 0..t. Its value is the indicator [token==1], so after the output projection the residual holds f = c/(t+1), where c is the number of ones in x[1..t]. BOS contributes 0 to the count and 1 to the denominator.
2. **Position.** The position embedding is a one-hot of t, so the MLP knows t.
3. **Parity from f.** For each position t, the MLP uses t neurons: relu((t+1)f − k) for k = 0..t−1, gated by +100 on the one-hot position and −100 in the bias so they fire only at that t. Their output weights are +1, −2, +2, −2, … They sum to a piecewise-linear zigzag in c that equals c mod 2 at every integer c from 0 to t. That uses 28 of the 64 neurons.
4. **Output.** The zigzag value p is written to a residual dimension. The unembedding gives logits [0.5, p], so class 1 wins exactly when p ≈ 1, with a margin of about 0.5 against float error around 1e-6. Layer 1 is all zeros, so it is an identity.