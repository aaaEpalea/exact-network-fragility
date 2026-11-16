# Construct a transformer that computes `majority8` (no training)

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
Causal streaming task. A counter starts at 8 and is saturated to the range [0,16]; token 1 increments it, token 2 decrements it, all other tokens leave it unchanged. The output at position t is 1 if the counter after x[1..t] is > 8 (more 1s than 2s so far), else 0. Categorical output over 2 classes (argmax). The output at position t must depend only on x[1..t].

Reference implementation (the ground truth, Python):
```python
def _majority(x, N=8):
    n = 2 * N + 1; s = N; out = []
    for v in x: s = min(s + 1, n - 1) if v == 1 else max(s - 1, 0) if v == 2 else s; out.append(int(s > N))
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

I didn't run this, but the construction is simple enough to verify by hand. It's a single attention layer with uniform causal attention, plus a linear readout.

**Key observation.** The counter starts at 8, and sequences have at most 7 tokens, so it can never hit 0 or 16. Saturation never happens. The output at position t is therefore 1 iff (#1s − #2s) over x[1..t] is at least 1.

**Construction.**
- **Embedding.** Channel 0 of `embed` holds a signed value: token 1 gives +1, token 2 gives −1, and tokens 0 (BOS), 3, 4, 5 and 6 give 0. Position embeddings are zero.
- **Attention.** Layer 0 has one head with `wq = wk = 0` and zero biases, so all scores are 0. With the causal mask, softmax is then exactly uniform over positions 0..t, including BOS. The value is channel 0, and `wo` writes the head output into channel 1. After the layer, channel 1 equals (#1 − #2)/(t+1).
- **MLP.** All MLP weights are zero, so it does nothing.
- **Readout.** The logit for class 0 is 0. The logit for class 1 is channel 1 minus 1/16.
- **Why it's exact.** If the difference is 0 or negative, the average is ≤ 0, so the class-1 logit is ≤ −1/16 and the output is 0. If the difference is ≥ 1, the average is at least 1/8 (since t+1 ≤ 8), so the class-1 logit is ≥ 1/16 and the output is 1. The margin of 1/16 is far larger than any float32 error. Because the attention is causal, position t depends only on x[1..t].

**Parameter count:** 56.

```python
import numpy as np

CONFIG = dict(n_layers=1, n_heads=1, d_model=2, d_head=1, d_mlp=1, causal=True)

def build_params():
    f = np.float32
    embed = np.zeros((7, 2), f)
    embed[1, 0] = 1.0    # token 1: +1
    embed[2, 0] = -1.0   # token 2: -1
    P = {}
    P["embed"] = embed
    P["pos_embed"] = np.zeros((8, 2), f)
    P["l0.wq"] = np.zeros((1, 2, 1), f)
    P["l0.wk"] = np.zeros((1, 2, 1), f)
    wv = np.zeros((1, 2, 1), f); wv[0, 0, 0] = 1.0
    P["l0.wv"] = wv
    P["l0.bq"] = np.zeros((1, 1), f)
    P["l0.bk"] = np.zeros((1, 1), f)
    P["l0.bv"] = np.zeros((1, 1), f)
    wo = np.zeros((1, 1, 2), f); wo[0, 0, 1] = 1.0
    P["l0.wo"] = wo
    P["l0.bo"] = np.zeros(2, f)
    P["l0.w1"] = np.zeros((2, 1), f)
    P["l0.b1"] = np.zeros(1, f)
    P["l0.w2"] = np.zeros((1, 2), f)
    P["l0.b2"] = np.zeros(2, f)
    uw = np.zeros((2, 2), f); uw[1, 1] = 1.0
    P["unembed.w"] = uw
    ub = np.zeros(2, f); ub[1] = -1.0 / 16
    P["unembed.b"] = ub
    return P
```