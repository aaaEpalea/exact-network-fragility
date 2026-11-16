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

Reverse needs, at output position i, the token sitting at position j = L+1−i = T−i, where T = L+1 is the sequence length including BOS. Layer 1 computes T. Layer 2 uses a sharp attention to fetch the token at position T−i. I have not run this code, so the correctness argument below is by reasoning only.

```python
import numpy as np

CONFIG = dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=64, causal=False)

def build_params():
    d, H, dh, dm = 32, 2, 16, 64
    P = {}
    # residual layout: 0 BOS flag | 1..6 token one-hot | 7 pos p | 8 p^2 | 9 const 1
    #                  10 u=1/T   | 11 T                | 16..21 fetched token one-hot
    E = np.zeros((7, d))
    E[0, 0] = 1.0
    for t in range(1, 7):
        E[t, t] = 1.0
    PE = np.zeros((8, d))
    for p in range(8):
        PE[p, 7] = p
        PE[p, 8] = p * p
        PE[p, 9] = 1.0
    P["embed"] = E
    P["pos_embed"] = PE

    def zeros_layer(i):
        P[f"l{i}.wq"] = np.zeros((H, d, dh)); P[f"l{i}.wk"] = np.zeros((H, d, dh))
        P[f"l{i}.wv"] = np.zeros((H, d, dh)); P[f"l{i}.bq"] = np.zeros((H, dh))
        P[f"l{i}.bk"] = np.zeros((H, dh));   P[f"l{i}.bv"] = np.zeros((H, dh))
        P[f"l{i}.wo"] = np.zeros((H, dh, d)); P[f"l{i}.bo"] = np.zeros(d)
        P[f"l{i}.w1"] = np.zeros((d, dm));   P[f"l{i}.b1"] = np.zeros(dm)
        P[f"l{i}.w2"] = np.zeros((dm, d));   P[f"l{i}.b2"] = np.zeros(d)
    zeros_layer(0); zeros_layer(1)

    # ---- Layer 0: head 0 uniform attention, value = BOS flag -> dim 10 = 1/T
    P["l0.wv"][0, 0, 0] = 1.0
    P["l0.wo"][0, 0, 10] = 1.0
    # MLP0: piecewise-linear interpolation of T = 1/u through u=1/8..1/2 -> dim 11
    a = [1.0 / t for t in range(8, 1, -1)]      # ascending u: 1/8 ... 1/2
    Tv = [float(t) for t in range(8, 1, -1)]    # 8 ... 2
    m = [(Tv[k] - Tv[k - 1]) / (a[k] - a[k - 1]) for k in range(1, 7)]
    coef = [m[0]] + [m[k] - m[k - 1] for k in range(1, 6)]
    for k in range(6):
        P["l0.w1"][10, k] = 1.0
        P["l0.b1"][k] = -a[k]
        P["l0.w2"][k, 11] = coef[k]
    P["l0.b2"][11] = 8.0

    # ---- Layer 1: head 0 attends to position j = T - i
    A = 240.0   # score = A*(tg*j - j^2/2)/4 -> gap A/8 = 30 between target and neighbors
    P["l1.wq"][0, 11, 0] = A
    P["l1.wq"][0, 7, 0] = -A
    P["l1.wq"][0, 9, 1] = -A / 2.0
    P["l1.wk"][0, 7, 0] = 1.0
    P["l1.wk"][0, 8, 1] = 1.0
    for t in range(1, 7):
        P["l1.wv"][0, t, t] = 1.0
        P["l1.wo"][0, t, 15 + t] = 1.0

    # ---- Unembed: dims 16..21 -> logits for tokens 1..6
    W = np.zeros((d, 7))
    for t in range(1, 7):
        W[15 + t, t] = 1.0
    P["unembed.w"] = W
    P["unembed.b"] = np.zeros(7)

    return {k: np.asarray(v, dtype=np.float32) for k, v in P.items()}
```

**Parameter count:** 17,543 (every tensor has the fixed shape; most entries are zero).

**How it works:**
- **Embeddings:** The token embedding is a one-hot in dims 1–6, and BOS sets a flag in dim 0. The position embedding supplies p, p² and a constant 1.
- **Layer 1, head 0:** It has zero q and k, so attention is uniform over the T positions. Its value is the BOS flag, so every position receives u = 1/T in dim 10.
- **Layer 1 MLP:** Six ReLU units form a piecewise-linear interpolation of 1/u through the seven possible points u = 1/8, …, 1/2. This writes the exact length T into dim 11, accurate to about 1e-5.
- **Layer 2, head 0:** The query is built as A·(T−i, −½) and the key as (j, j²). The score at key position j is therefore (A/4)·(tg·j − j²/2), where tg = T−i. This equals −(A/8)(j−tg)² plus a term that does not depend on j. With A = 240, the target position beats its neighbours by 30 logits, so the softmax is essentially one-hot on position T−i.
- **Value and unembed:** The value and output maps copy the attended token's one-hot into dims 16–21. The unembedding maps those dims to logits 1–6, with logit 0 left at zero, so the argmax is the token at position L+1−i.
- **Unused weights:** Head 1 in both layers, the layer-2 MLP and all biases not mentioned are zero.