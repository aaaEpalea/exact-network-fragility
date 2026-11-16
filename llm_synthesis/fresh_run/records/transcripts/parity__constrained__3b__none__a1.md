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


**Result:** the construction is exactly correct on all 335,922 sequences. I checked this with a numpy reimplementation of the forward pass and found 0 errors, and the file has 6706 parameters, matching the required architecture.

**Deliverable (`submission.py`):**
```python
import numpy as np
CONFIG = dict(n_layers=2, n_heads=2, d_model=16, d_head=8, d_mlp=64, causal=True)
def build_params():
    d, H, dh, dm = 16, 2, 8, 64
    f32 = np.float32
    P = {}
    emb = np.zeros((7, d)); emb[1, 0] = 1.0
    pe = np.zeros((8, d))
    for p in range(8): pe[p, 1 + p] = 1.0
    P["embed"] = emb; P["pos_embed"] = pe
    for i in range(2):
        for n in ("wq", "wk", "wv"): P[f"l{i}.{n}"] = np.zeros((H, d, dh))
        for n in ("bq", "bk", "bv"): P[f"l{i}.{n}"] = np.zeros((H, dh))
        P[f"l{i}.wo"] = np.zeros((H, dh, d)); P[f"l{i}.bo"] = np.zeros(d)
        P[f"l{i}.w1"] = np.zeros((d, dm)); P[f"l{i}.b1"] = np.zeros(dm)
        P[f"l{i}.w2"] = np.zeros((dm, d)); P[f"l{i}.b2"] = np.zeros(d)
    # layer 0 head 0: uniform attention (zero scores), averages the "token==1" indicator into dim 9
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 9] = 1.0
    M = 100.0
    u = 0
    for t in range(1, 8):
        N = t + 1
        v = [2.0 * (c % 2) - 1.0 for c in range(N)]
        slopes = [(v[j + 1] - v[j]) * N for j in range(t)]
        prev = 0.0
        for j in range(t):
            s = slopes[j] - prev; prev = slopes[j]
            P["l0.w1"][9, u] = 1.0
            P["l0.w1"][1 + t, u] = M
            P["l0.b1"][u] = -(j / N) - M
            P["l0.w2"][u, 10] = s
            u += 1
    uw = np.zeros((d, 2)); uw[10, 1] = 0.5; uw[10, 0] = -0.5
    P["unembed.w"] = uw; P["unembed.b"] = np.array([0.5, -0.5])
    return {k: np.asarray(v, dtype=f32) for k, v in P.items()}
```

**How it works:**
- **Embeddings:** token 1 sets dimension 0 to 1, so dimension 0 is the "token is 1" indicator. The position embedding is a one-hot in dimensions 1–8, so dimension 1+t flags position t.
- **Layer-0 attention:** head 0 has all-zero query and key weights, so the causal softmax is uniform over positions 0..t. It averages the indicator and writes f = c/(t+1) into dimension 9, where c is the number of ones among x[1..t] (BOS contributes 0).
- **Layer-0 MLP:** for each position t = 1..7, the parity of c is a function of f at the t+1 known points c/(t+1). I use a piecewise-linear zigzag through those points, built from t ReLU units with breakpoints at j/(t+1).
- **Gating:** each unit has a large positive weight (M = 100) on its own position's one-hot, with a bias of -M. It is therefore zero at every other position and active only at its own, which uses 28 of the 64 hidden units.
- **Output:** the MLP writes g = ±1 (odd parity = +1, even = −1) into dimension 10. The unembedding maps it to logits (−g/2 + 1/2, g/2 − 1/2), so argmax gives the parity.
- **Remaining pieces:** layer 1 and head 1 are all zeros and pass the residual through unchanged. The slope and bias values are exact rationals, so float32 error (about 1e-6) is far below the ±1 logit margin.