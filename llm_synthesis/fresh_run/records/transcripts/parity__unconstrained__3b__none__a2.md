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


The construction is a single layer with one attention head and a gated ReLU MLP. I have not run it; I checked it only by reasoning.

```python
import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=11, d_head=1, d_mlp=56, causal=True)

def build_params():
    d, dm, M = 11, 56, 20.0
    P = {}
    # residual layout: 0..7 position one-hot, 8 is_one flag, 9 mean (attention output), 10 parity
    E = np.zeros((7, d), np.float32); E[1, 8] = 1.0
    PE = np.zeros((8, d), np.float32)
    for p in range(8): PE[p, p] = 1.0
    P["embed"] = E; P["pos_embed"] = PE
    P["l0.wq"] = np.zeros((1, d, 1), np.float32)
    P["l0.wk"] = np.zeros((1, d, 1), np.float32)
    wv = np.zeros((1, d, 1), np.float32); wv[0, 8, 0] = 1.0
    P["l0.wv"] = wv
    P["l0.bq"] = np.zeros((1, 1), np.float32)
    P["l0.bk"] = np.zeros((1, 1), np.float32)
    P["l0.bv"] = np.zeros((1, 1), np.float32)
    wo = np.zeros((1, 1, d), np.float32); wo[0, 0, 9] = 1.0
    P["l0.wo"] = wo
    P["l0.bo"] = np.zeros(d, np.float32)
    w1 = np.zeros((d, dm), np.float32); b1 = np.zeros(dm, np.float32)
    w2 = np.zeros((dm, d), np.float32)
    j = 0
    for t in range(1, 8):
        for k in range(1, t + 1):
            sign = 1.0 if k % 2 == 1 else -1.0
            # unit A: relu(c - k + 1) when at position t; unit B: relu(c - k)
            for off, sg in ((1, 1.0), (0, -1.0)):
                w1[9, j] = t + 1          # c = (t+1)*mean
                w1[t, j] = M              # gate on position t
                b1[j] = -k + off - M
                w2[j, 10] = sign * sg
                j += 1
    assert j == dm
    P["l0.w1"] = w1; P["l0.b1"] = b1; P["l0.w2"] = w2
    P["l0.b2"] = np.zeros(d, np.float32)
    uw = np.zeros((d, 2), np.float32); uw[10, 0] = -1.0; uw[10, 1] = 1.0
    P["unembed.w"] = uw
    P["unembed.b"] = np.array([0.5, -0.5], np.float32)
    return P
```

**Parameter count:** 7·11 + 8·11 = 165 for the embeddings. Attention contributes 11 (wv) + 11 (wo) + 11 (bo) + 3 (wq, wk, bq/bk/bv counted as 1 each) plus the zero tensors, about 40 in total. The MLP has 11·56 + 56 + 56·11 + 11 = 1299. The unembedding has 22 + 2 = 24. The total is about 1,540, far under the 60,000 limit.

**How it works:**
- **Counting:** The query and key weights are zero, so causal attention averages uniformly over positions 0..t, including BOS. The value reads an is_one flag (set only for token 1). The attention output is therefore m = c/(t+1), where c is the number of 1s among x[1..t].
- **Recovering c:** In the MLP, the position one-hot at dimension t gates a block of units. Inside that block, the weight (t+1)·m recovers the integer c exactly. Units for a different position get a bias of −M = −20 and stay at 0, since c ≤ 7.
- **Parity as a sum:** For integer c, [c ≥ k] = relu(c−k+1) − relu(c−k). Parity(c) is the alternating sum Σ_{k=1..t} (−1)^{k+1}·[c ≥ k]. This is written to dimension 10 through W2.
- **Output:** The logits are [0.5 − p, p − 0.5], so the argmax is the parity with a margin of 1. Position 0 is ignored. Because attention is causal, the output at t depends only on x[1..t].