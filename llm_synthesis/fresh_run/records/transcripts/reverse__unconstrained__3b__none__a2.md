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


Here is `submission.py`. I did not run it, and I could not test the assumption below; the correctness argument is by hand-tracing only.

```python
import numpy as np

CONFIG = dict(n_layers=3, n_heads=1, d_model=38, d_head=8, d_mlp=28, causal=False)

def build_params():
    d, dh, H, dm = 38, 8, 1, 28
    P = {}
    z = lambda *s: np.zeros(s, np.float32)

    # token one-hot -> dims 0..6 ; position one-hot -> dims 7..14
    emb = z(7, d)
    for t in range(7): emb[t, t] = 1.0
    pe = z(8, d)
    for p in range(8): pe[p, 7 + p] = 1.0
    P["embed"], P["pos_embed"] = emb, pe

    for i in range(3):
        for n in ["wq", "wk", "wv"]: P[f"l{i}.{n}"] = z(H, d, dh)
        for n in ["bq", "bk", "bv"]: P[f"l{i}.{n}"] = z(H, dh)
        P[f"l{i}.wo"] = z(H, dh, d); P[f"l{i}.bo"] = z(d)
        P[f"l{i}.w1"] = z(d, dm); P[f"l{i}.b1"] = z(dm)
        P[f"l{i}.w2"] = z(dm, d); P[f"l{i}.b2"] = z(d)

    # Layer 0 attention: uniform attention (q=k=0); value = 1000*[token==BOS]
    # -> dim 15 gets u = 1000/T at every position.
    P["l0.wv"][0, 0, 0] = 1000.0
    P["l0.wo"][0, 0, 15] = 1.0
    # Layer 0 MLP: tent functions around u = 1000/T, giving one-hot(T) in dims 16..22
    delta = 10.0
    n = 0
    for T in range(2, 9):
        c = 1000.0 / T
        for kk, coef in ((1, 1.0), (0, -2.0), (-1, 1.0)):
            P["l0.w1"][15, n] = 1.0 / delta
            P["l0.b1"][n] = -c / delta + kk
            P["l0.w2"][n, 16 + T - 2] = coef
            n += 1

    # Layer 1 MLP: AND(T, pos i) -> one-hot of m = T-i in dims 23+m
    # (layer 1 attention stays zero)
    n = 0
    for T in range(2, 9):
        for i in range(1, T):
            P["l1.w1"][16 + T - 2, n] = 1.0
            P["l1.w1"][7 + i, n] = 1.0
            P["l1.b1"][n] = -1.0
            P["l1.w2"][n, 23 + (T - i)] = 1.0
            n += 1

    # Layer 2 attention: query = S*onehot(m), key = onehot(pos), value = token one-hot
    S = 100.0
    for m in range(8):
        P["l2.wq"][0, 23 + m, m] = S
    for p in range(8):
        P["l2.wk"][0, 7 + p, p] = 1.0
    for t in range(7):
        P["l2.wv"][0, t, t] = 1.0
        P["l2.wo"][0, t, 31 + t] = 1.0

    uw = z(d, 7)
    for t in range(7): uw[31 + t, t] = 1.0
    P["unembed.w"], P["unembed.b"] = uw, z(7)
    return P
```

**Parameter count:** about 11.6k (embed 266 + pos_embed 304 + 3 layers × (4·304 attention weights + 2·1064 MLP weights + small biases)). This is well under the 60,000 limit.

**Assumption:** the model sees each sequence at its true length T = L+1, with no padding. This is the only way it can learn L. If the evaluator pads sequences to a fixed length, this construction would fail, and I have no way to check which case applies.

**How it works** (non-causal, 3 layers, one head):
1. **Layer 0 finds the length.** Attention is uniform (q = k = 0), and the value is 1000·[token is BOS]. Every position therefore receives u = 1000/T. The layer-0 MLP turns u into a one-hot of T using 3-ReLU tent functions, which are exact because the spacing between the values 1000/T is at least 17.9, larger than the tent width of 10.
2. **Layer 1 finds the target position.** Its attention is zero. Its MLP has one AND neuron, relu(onehot_T + onehot_pos − 1), for each valid (T, i) pair. Each such neuron writes a one-hot of the target index m = T − i. For output position i in 1..L this is exactly L+1−i.
3. **Layer 2 fetches the token.** Query is 100·onehot(m), key is onehot(position), and the value is the token one-hot. The attention score is about 35 for the matching position and 0 elsewhere. The weights on non-targets are therefore around 1e-15, so the attended value is the target token's one-hot, written into dims 31..37.
4. **Unembedding** maps dim 31+t to logit t, so the argmax is x[L+1−i].