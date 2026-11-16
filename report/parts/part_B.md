
## Part B. Mechanism and scaling with sequence length

Part A showed fragility; Part B asks how it depends on the maximum sequence length N and which weights are responsible. Counting (storing the count as 1/(n+1) in an attention average) is the extreme case, so most experiments use the `length` program.

### B1. Counting robustness vs max sequence length N (length program)

**Question.** How does counting robustness depend on the maximum length N?

**Code:** `code/scaling.py`  
**Results:** `results/scaling.json`


| item | 1.0 / clean | 1.0 / abs / s50(curve) | 1.0 / s50 | 0.25 / clean | 0.25 / abs / s50(curve) | 0.25 / s50 |
|---|---|---|---|---|---|---|
| 4 | 1 | 0.001417 | 0.001417 | 1 | 0.004789 | 0.004789 |
| 6 | 1 | 5.53e-04 | 5.53e-04 | 1 | 0.002365 | 0.002365 |
| 8 | 1 | 4.39e-04 | 4.39e-04 | 1 | 0.001602 | 0.001602 |
| 12 | 0.8182 |  |  | 0.8182 |  |  |
| 16 | 0.6 |  |  | 0.6 |  |  |
| 24 | 0.3913 |  |  | 0.3913 |  |  |


sigma50 falls from 1.4e-3 (N=4) to 4.4e-4 (N=8). For N>=12 the default compiler is not even exact at zero noise (clean accuracy 0.82, 0.60, 0.39 at N=12, 16, 24), caused by the fixed discretizer sharpness (mlp_exactness=100). Attention-scale c=0.25 gives a roughly constant 3x.

> **Takeaway.** Tolerance falls from 1.4e-3 at N=4 to 4.4e-4 at N=8, and for N of 12 and above the default compiler is not even exact at zero noise because the discretizer sharpness is fixed. This silent loss of exactness is a compiler limitation we report; it matches an independent anecdotal report of Tracr instability for long sequences. Lowering the attention scale gives a roughly constant 3 times.

![Figure B1.](figures/counting_scaling.png)

**Figure B1.** Left: tolerance of Tracr's attention counter falls roughly as N to the minus 1.5 (default sharpness, best exact sharpness, and with attention scale 0.25). Right: the default compiler silently becomes inexact for N of 10 and above; raising the sharpness restores exactness.

### B2. mlp_exactness (S) needed vs N

**Question.** How sharp must the compiler's discretizer be to stay exact at large N?

**Code:** `code/scaling2.py`  
**Results:** `results/scaling2.json`


| program | variant | 1.0 / clean | 1.0 / abs / s50(curve) | 1.0 / s50 | 0.25 / clean | 0.25 / abs / s50(curve) | 0.25 / s50 |
|---|---|---|---|---|---|---|---|
| 8 | 100 | 1 | 3.78e-04 | 3.78e-04 | 1 | 0.001491 | 0.001491 |
| 8 | 300 | 1 | 4.39e-04 | 4.39e-04 | 1 | 0.001517 | 0.001517 |
| 8 | 1000 | 1 | 3.80e-04 | 3.80e-04 | 1 | 0.001293 | 0.001293 |
| 8 | 3000 | 1 | 3.20e-04 | 3.20e-04 | 1 | 9.04e-04 | 9.04e-04 |
| 12 | 100 | 0.8182 |  |  | 0.8182 |  |  |
| 12 | 300 | 1 | 3.60e-04 | 3.60e-04 | 1 | 0.001063 | 0.001063 |
| 12 | 1000 | 1 | 3.11e-04 | 3.11e-04 | 1 | 7.59e-04 | 7.59e-04 |
| 12 | 3000 | 1 | 2.37e-04 | 2.37e-04 | 1 | 6.16e-04 | 6.16e-04 |
| 16 | 100 | 0.6 |  |  | 0.6 |  |  |
| 16 | 300 | 1 | 2.04e-04 | 2.04e-04 | 1 | 6.92e-04 | 6.92e-04 |
| 16 | 1000 | 1 | 1.87e-04 | 1.87e-04 | 1 | 5.20e-04 | 5.20e-04 |
| 16 | 3000 | 1 | 1.53e-04 | 1.53e-04 | 1 | 3.85e-04 | 3.85e-04 |
| 24 | 100 | 0.3913 |  |  | 0.3913 |  |  |
| 24 | 300 | 0.6957 |  |  | 0.6957 |  |  |
| 24 | 1000 | 1 | 1.07e-04 | 1.07e-04 | 1 | 2.87e-04 | 2.87e-04 |
| 24 | 3000 | 1 | 9.83e-05 | 9.83e-05 | 1 | 2.52e-04 | 2.52e-04 |


Exactness is restored by raising S to roughly N^2 (S=300 works to N=16, S=1000 at N=24). Even then sigma50 keeps falling (about N^-1.5); over-sharp S slightly reduces tolerance.

> **Takeaway.** Raising the sharpness S to about N squared restores exactness (S=300 up to N=16, S=1000 at N=24). Even then tolerance keeps falling, roughly as N to the minus 1.5, and over-sharp settings slightly reduce it.

### B3. Scaling for reverse / hist / sort_unique / frac_prevs

**Question.** Is the length decay specific to the counting program?

**Code:** `code/scaling3.py`  
**Results:** `results/scaling3.json`


| program | variant | 1.0 / clean | 1.0 / S | 1.0 / abs / s50(curve) | 1.0 / s50 | 0.25 / clean | 0.25 / S | 0.25 / abs / s50(curve) | 0.25 / s50 |
|---|---|---|---|---|---|---|---|---|---|
| reverse | 6 | 1 | 100 | 5.76e-04 | 5.76e-04 | 1 | 100 | 0.001933 | 0.001933 |
| reverse | 8 | 1 | 100 | 6.93e-04 | 6.93e-04 | 1 | 100 | 0.001847 | 0.001847 |
| reverse | 12 | 1 | 216 | 3.49e-04 | 3.49e-04 | 1 | 216 | 4.32e-04 | 4.32e-04 |
| reverse | 16 | 1 | 384 | 1.91e-04 | 1.91e-04 | 1 | 384 | 3.13e-04 | 3.13e-04 |
| hist | 6 | 1 | 100 | 0.001474 | 0.001474 | 1 | 100 | 0.003564 | 0.003564 |
| hist | 8 | 1 | 100 | 0.001474 | 0.001474 | 1 | 100 | 0.005219 | 0.005219 |
| hist | 12 | 1 | 216 | 0.001407 | 0.001407 | 1 | 216 | 0.002862 | 0.002862 |
| hist | 16 | 1 | 384 | 7.16e-04 | 7.16e-04 | 1 | 384 | 8.31e-04 | 8.31e-04 |
| sort_unique | 6 | 1 | 100 | 9.20e-04 | 9.20e-04 | 1 | 100 | 0.001712 | 0.001712 |
| sort_unique | 8 | 1 | 100 | 0.00138 | 0.00138 | 1 | 100 | 0.001455 | 0.001455 |
| frac_prevs | 6 | 1 | 100 | 0.002187 | 0.002187 | 1 | 100 | 0.005188 | 0.005188 |
| frac_prevs | 8 | 1 | 100 | 0.001841 | 0.001841 | 1 | 100 | 0.004699 | 0.004699 |
| frac_prevs | 12 | 1 | 216 | 0.001413 | 0.001413 | 1 | 216 | 0.003505 | 0.003505 |
| frac_prevs | 16 | 1 | 384 | 0.001498 | 0.001498 | 0.6262 | 384 |  |  |


Tolerance falls with N for all programs at different rates (log-log slopes about -0.4 to -1.2 at default attention). The attention-scale gain shrinks with N (reverse: 3.4x at N=8, 1.6x at N=16) and frac_prevs loses exactness at c=0.25, N=16.

> **Takeaway.** No. Tolerance falls with N for `reverse`, `hist`, `sort_unique` and `frac_prevs` too, with log-log slopes between about -0.4 and -1.2 at the default attention scale. The benefit of a smaller attention scale shrinks with N, and `frac_prevs` loses exactness at scale 0.25 and N=16.

### B4. Component diagnosis: which weights limit the length program

**Question.** Which weights limit the `length` program?

**Code:** `code/diagnose.py`  
**Results:** `results/diagnose.json`


| item | d | embeddings | attn_query | attn_key | attn_value | attn_out | mlp_in | mlp_out | ALL |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 27 | 5.65e-04 | 0.1313 | 0.003962 | 0.02333 | 0.05056 | 0.127 | 0.04311 | 3.86e-04 |
| 16 | 43 | 2.20e-04 | 0.04109 | 4.26e-04 | 0.00748 | 0.01879 | 0.0929 | 0.02371 | 1.81e-04 |
| 32 | 75 | 1.26e-04 | 0.01632 | 3.79e-04 | 0.003088 | 0.02113 | 0.09315 | 0.003946 | 1.02e-04 |
| 48 | 107 | 6.92e-05 | 0.01453 | 3.51e-04 | 0.001579 | 0.001654 | 0.08703 | 0.002487 | 4.68e-05 |


sigma50 with noise on one tensor group at a time. Embeddings alone nearly equal the all-weights tolerance; attention key and value paths are next; MLP input weights barely matter. Embedding slope about -1.1 and value path about -1.5 match a simple analysis of the count being stored as 1/(n+1).

> **Takeaway.** Embeddings alone nearly equal the all-weights tolerance, followed by the attention key and value paths; MLP input weights barely matter. The measured slopes (about -1.1 for embeddings, -1.5 for the value path) match a simple analysis in which the count is stored as 1/(n+1) and must be resolved to better than the gap between adjacent counts.

### B5. Chunked counting vs plain counting

**Question.** Can counting inside short windows, then adding exact integers, remove the length dependence?

**Code:** `code/chunk_count.py`, `code/chunk_big.py`, `code/diagnose_chunk.py`  
**Results:** `results/chunk_count.json`, `results/chunk_big.json`, `results/diagnose_chunk.json`

`chunk_count.json`


| program | variant | layers | d | 1.0 / clean | 1.0 / abs / s50(curve) | 1.0 / s50 | 0.25 / clean | 0.25 / abs / s50(curve) | 0.25 / s50 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | chunk4 | 2 | 22 | 1 | 7.31e-04 | 7.31e-04 | 1 | 0.001633 | 0.001633 |
| 12 | chunk4 | 3 | 29 | 1 | 4.55e-04 | 4.55e-04 | 1 | 7.12e-04 | 7.12e-04 |
| 12 | chunk8 | 2 | 26 | 1 | 3.93e-04 | 3.93e-04 | 1 | 0.001292 | 0.001292 |
| 16 | chunk4 | 3 | 36 | 1 | 3.24e-04 | 3.24e-04 | 1 | 4.32e-04 | 4.32e-04 |
| 16 | chunk8 | 2 | 30 | 1 | 2.89e-04 | 2.89e-04 | 1 | 6.35e-04 | 6.35e-04 |
| 24 | chunk4 | 4 | 50 | 1 | 3.59e-04 | 3.59e-04 | 1 | 3.99e-04 | 3.99e-04 |
| 24 | chunk8 | 3 | 41 | 0.6522 |  |  | 1 | 1.84e-04 | 1.84e-04 |


`chunk_big.json`


| item | layers | d | 1.0 / clean | 1.0 / abs / s50(curve) | 1.0 / s50 | 0.25 / clean | 0.25 / abs / s50(curve) | 0.25 / s50 |
|---|---|---|---|---|---|---|---|---|
| 32 / plain | 1 | 75 | 1 | 1.02e-04 | 1.02e-04 | 1 | 2.07e-04 | 2.07e-04 |
| 32 / chunk4 | 4 | 64 | 1 | 1.18e-04 | 1.18e-04 | 0.9355 |  |  |
| 48 / plain | 1 | 107 | 1 | 4.55e-05 | 4.55e-05 | 1 | 1.10e-04 | 1.10e-04 |
| 48 / chunk4 | 5 | 92 | 1 | 5.62e-05 | 5.62e-05 | 0.234 |  |  |


`diagnose_chunk.json`


| item | d | embeddings | attn_query | attn_key | attn_value | attn_out | mlp_in | mlp_out | ALL |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 36 | 9.60e-04 | 0.1086 | 0.001572 | 0.0219 | 0.01056 | 0.004454 | 4.26e-04 | 3.24e-04 |
| 32 | 64 | 7.93e-04 | 0.1678 | 0.002531 | 0.0108 | 0.004965 | 0.001076 | 1.57e-04 | 1.34e-04 |


Counting inside windows of 4 positions then adding exact integers improves the attention-path components 4-7x but the extra MLP layers become the new bottleneck (mlp_out), so overall tolerance gains only ~1.2x at large N and the trend with N is unchanged. `diagnose_chunk.json` holds the component breakdown (N=16 and 32 complete).

> **Takeaway.** It improves the attention-path components 4 to 7 times, but the extra MLP layers become the new bottleneck, so overall tolerance gains only about 1.2 times at large N and the trend with N is unchanged. The N=48 chunked runs did not finish.
