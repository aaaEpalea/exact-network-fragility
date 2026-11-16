
## Part G. Trained-transformer comparisons and hybrids

Trained transformers are the natural comparison for compiled ones. Here we ask whether noise-aware training can make them tolerant, how their tolerance decays with length when properly trained, and whether a frozen digital counter bolted onto a transformer helps.

### G1. Noise-aware training of a trained transformer (hist)

**Question.** Can noise-aware training make a trained transformer tolerant?

**Code:** `code/noise_aware_hist.py`  
**Results:** `results/noise_aware_hist.json`


| item | clean | alpha50 |
|---|---|---|
| 16 | 0.9667 | 0.1964 |
| 32 | 0.68 | 0.05583 |


N=16: alpha50 0.196 at clean accuracy 0.967 (baseline about 0.075 at 0.993); N=32: clean accuracy only 0.68. It trades exactness for tolerance and does not fix the scaling. One seed.

> **Takeaway.** At N=16 it raises relative tolerance to alpha50 0.196 at clean accuracy 0.967 (baseline about 0.075 at 0.993); at N=32 clean accuracy is only 0.68. It trades exactness for tolerance and does not fix the scaling. One seed.

### G2. Hybrid transformer + frozen digital counters

**Question.** A hybrid: transformer plus frozen digital counters.

**Code:** `code/hybrid_transformer.py`, `code/hybrid2.py`, `code/retrain_baselines.py`  
**Results:** `results/hybrid_transformer.json`, `results/hybrid2.json`, `results/retrain_baselines.json`

`hybrid_transformer.json`


| item | clean | alpha50 | params_trainable |
|---|---|---|---|
| N16 · baseline | 0.9929 | 0.07912 | 35968 |
| N16 · hybrid | 1 | 0.06826 | 42496 |
| N32 · baseline | 0.86 | 0.03433 | 38016 |
| N32 · hybrid | 1 | 0.04559 | 50688 |
| N64 · baseline | 0.625 | 0.01821 | 42112 |
| N64 · hybrid | 1 | 0.03355 | 67072 |


`hybrid2.json`


| item | clean | alpha50 | params_trainable |
|---|---|---|---|
| hist · baseline · steps8000 · seed0 | 0.9933 | 0.04606 | 38016 |
| hist · hybrid · steps3000 · seed1 | 1 | 0.04261 | 50688 |
| hist · hybrid · steps3000 · seed2 | 1 | 0.04229 | 50688 |
| reverse · baseline · steps3000 · seed0 | 0.1933 | nan | 38016 |
| reverse · hybrid · steps3000 · seed0 | 0.4 | nan | 40128 |


`retrain_baselines.json`


| item | clean | alpha50 | params_trainable |
|---|---|---|---|
| N16 · seed0 | 0.9952 | 0.08556 | 35968 |
| N16 · seed1 | 0.9952 | 0.08041 | 35968 |
| N32 · seed1 | 0.9933 | 0.04749 | 38016 |
| N64 · seed0 | 0.875 | 0.03568 | 42112 |


hist task. Hybrid reaches clean accuracy 1.000 at N=16, 32, 64 with alpha50 0.068, 0.046, 0.034; the transformer alone, properly trained (8,000 steps), reaches 0.995 / 0.993 / 0.875 with alpha50 0.08-0.086 / 0.046-0.048 / 0.036. So the hybrid's lasting benefit is exactness at long lengths, not tolerance. `hybrid2.json`: reverse at N=32 not learned by either model in 3,000 steps (inconclusive).

> **Takeaway.** The hybrid reaches clean accuracy 1.000 at N=16, 32 and 64 with alpha50 0.068, 0.046, 0.034; the transformer alone, properly trained for 8,000 steps, reaches 0.995, 0.993, 0.875 with alpha50 0.08 to 0.086, 0.046 to 0.048, 0.036. The hybrid's lasting benefit is exactness at long lengths, not tolerance. An earlier claim that trained transformers lose tolerance strongly with length is weaker once they are trained properly (about N to the minus 0.6 to minus 0.8). The `reverse` experiment at N=32 is inconclusive.
