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

CONFIG = dict(n_layers=1, n_heads=1, d_model=11, d_head=1, d_mlp=56, causal=True)

def build_params():
    d, dm, H, dh = 11, 56, 1, 1
    K, M = 32.0, 100.0
    P = {}
    embed = np.zeros((7, d), np.float32)
    embed[1, 8] = 1.0                      # token 1 -> is1 feature (dim 8)
    pos = np.zeros((8, d), np.float32)
    for t in range(8):
        pos[t, t] = 1.0                    # one-hot position (dims 0..7)
    P["embed"], P["pos_embed"] = embed, pos

    # attention: zero q/k -> uniform causal average over positions 0..t
    P["l0.wq"] = np.zeros((H, d, dh), np.float32)
    P["l0.wk"] = np.zeros((H, d, dh), np.float32)
    P["l0.bq"] = np.zeros((H, dh), np.float32)
    P["l0.bk"] = np.zeros((H, dh), np.float32)
    wv = np.zeros((H, d, dh), np.float32); wv[0, 8, 0] = 1.0
    P["l0.wv"] = wv
    P["l0.bv"] = np.zeros((H, dh), np.float32)
    wo = np.zeros((H, dh, d), np.float32); wo[0, 0, 9] = 1.0   # m = c/(t+1) -> dim 9
    P["l0.wo"] = wo
    P["l0.bo"] = np.zeros(d, np.float32)

    w1 = np.zeros((d, dm), np.float32); b1 = np.zeros(dm, np.float32)
    w2 = np.zeros((dm, d), np.float32)
    u = 0
    for t in range(1, 8):
        for j in range(1, t + 1):
            thr = (j - 0.5) / (t + 1)
            sign = 1.0 if j % 2 == 1 else -1.0
            for off, outsign in ((1.0, sign), (0.0, -sign)):
                w1[9, u] = K
                w1[t, u] = M
                b1[u] = -K * thr - M + off
                w2[u, 10] = outsign        # parity accumulator in dim 10
                u += 1
    assert u == dm
    P["l0.w1"], P["l0.b1"], P["l0.w2"] = w1, b1, w2
    P["l0.b2"] = np.zeros(d, np.float32)

    uw = np.zeros((d, 2), np.float32)
    uw[10, 1] = 1.0
    uw[10, 0] = 0.0
    ub = np.array([0.0, -0.5], np.float32)  # logit1 = p-0.5, logit0 = 0
    P["unembed.w"], P["unembed.b"] = uw, ub
    return P
```

**Parameter count:** the tensors total 1,609 parameters, well under the 60,000 limit. That is embed 77, pos_embed 88, attention 11+11+11+11 weights plus 1+1+1+11 biases, w1 616, b1 56, w2 616, b2 11, and unembed 22+2.

**How it works:** I have not run this against the 335,922 sequences. The correctness argument is by construction.

1. Token 1 sets a feature in dim 8 of the embedding. Position embeddings are one-hot over dims 0–7.
2. Attention has zero queries and keys, so it is a uniform causal average over positions 0..t, BOS included. With value = the is1 feature, dim 9 receives m = c/(t+1), where c is the number of 1s so far.
3. Parity is the alternating sum of step functions: parity(c) = Σ_{j=1..t} (−1)^{j+1}·[c ≥ j].
4. For each pair (t, j) with j ≤ t, two ReLU units compute z = K(m − (j−½)/(t+1)) − M(1 − onehot_t), with K = 32 and M = 100, and add +1 to the first unit's bias. Their difference relu(z+1) − relu(z) is exactly [c ≥ j] at position t and exactly 0 at every other position.
5. For integer c the argument z is at least 2 when c ≥ j and at most −2 when c < j, since m is a multiple of 1/(t+1) ≥ 1/8. So each step is exact.
6. The signed sum accumulates in dim 10 as the parity p. The unembedding gives logits (0, p − 0.5), so argmax is class 1 exactly when p = 1.
7. Attention is causal, so the output at position t depends only on x[1..t].