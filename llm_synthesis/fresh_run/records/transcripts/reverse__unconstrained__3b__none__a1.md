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


**submission.py**

```python
import numpy as np

D, H, DH, DM = 15, 1, 6, 1
BETA = 10.0
CONFIG = dict(n_layers=2, n_heads=H, d_model=D, d_head=DH, d_mlp=DM, causal=False)

def build_params():
    f = np.float32
    P = {}
    # dims: 0 = position j, 1 = j^2, 2..7 = token one-hot (tokens 1..6),
    #       8 = mean position (T-1)/2, 9..14 = copied token one-hot
    emb = np.zeros((7, D), f)
    for t in range(1, 7):
        emb[t, 1 + t] = 1.0
    pos = np.zeros((8, D), f)
    for j in range(8):
        pos[j, 0] = j
        pos[j, 1] = j * j
    P["embed"], P["pos_embed"] = emb, pos

    def zeros_layer(i):
        P[f"l{i}.wq"] = np.zeros((H, D, DH), f); P[f"l{i}.wk"] = np.zeros((H, D, DH), f)
        P[f"l{i}.wv"] = np.zeros((H, D, DH), f)
        P[f"l{i}.bq"] = np.zeros((H, DH), f); P[f"l{i}.bk"] = np.zeros((H, DH), f)
        P[f"l{i}.bv"] = np.zeros((H, DH), f)
        P[f"l{i}.wo"] = np.zeros((H, DH, D), f); P[f"l{i}.bo"] = np.zeros(D, f)
        P[f"l{i}.w1"] = np.zeros((D, DM), f); P[f"l{i}.b1"] = np.zeros(DM, f)
        P[f"l{i}.w2"] = np.zeros((DM, D), f); P[f"l{i}.b2"] = np.zeros(D, f)

    # Layer 0: uniform attention over all positions, value = position index
    # -> every position receives m = (T-1)/2, written to dim 8.
    zeros_layer(0)
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 8] = 1.0

    # Layer 1: score(i, j) = -BETA*(j - a)^2 + const, with a = T - i = 2m + 1 - i
    zeros_layer(1)
    c = np.sqrt(DH)  # cancels the 1/sqrt(d_head) scaling
    P["l1.wq"][0, 8, 0] = 4 * BETA * c
    P["l1.wq"][0, 0, 0] = -2 * BETA * c
    P["l1.bq"][0, 0] = 2 * BETA * c
    P["l1.bq"][0, 1] = -BETA * c
    P["l1.wk"][0, 0, 0] = 1.0   # key feature j
    P["l1.wk"][0, 1, 1] = 1.0   # key feature j^2
    for t in range(6):
        P["l1.wv"][0, 2 + t, t] = 1.0
        P["l1.wo"][0, t, 9 + t] = 1.0

    uw = np.zeros((D, 7), f)
    for t in range(6):
        uw[9 + t, t + 1] = 1.0
    P["unembed.w"], P["unembed.b"] = uw, np.zeros(7, f)
    return P
```

**Parameter count:** 1,216 (embed 105, pos_embed 120, each layer 521, unembed 112 with bias 7 — within the 60,000 limit).

**How it works:**
- The model is non-causal with 2 layers and 1 head. The MLPs are all zeros and unused.
- Position 0 is BOS and the sequence has T = L+1 positions. Layer 0 uses uniform attention (q and k are zero) with the position index as the value. Every position then receives the mean position m = (T-1)/2, exactly linear in T, and writes it to dim 8.
- Layer 1 needs each position i to read position j = T-i = L+1-i. The query and key are built so the score is -β(j-a)² plus a term constant in j, where a = 2m+1-i = T-i. The pieces are:
  - Expanding the square gives 2βaj - βj², which is bilinear.
  - The key features are j and j² (from pos_embed).
  - The query features are 2βa, linear in m and i, and the constant -β.
  - Both are pre-scaled by √d_head to cancel the 1/√d_head factor in the forward pass.
- With β=10, the nearest wrong position has a score lower by at least 10. The softmax error is therefore about 3e-4 or less, far below the margin needed for argmax.
- The value copies the one-hot of the attended token into dims 9–14, and the unembedding maps those dims to logits 1–6. The result is the reversed sequence at positions 1..L.
- I checked this by reasoning only and did not run it.