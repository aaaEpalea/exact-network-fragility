
## Part A. Audit: compiled programs are fragile

We start from the observation that motivated everything else: programs compiled into transformer weights by Tracr are exactly correct, but how much noise can their parameters tolerate? We measure six programs under several noise protocols, compare them with trained transformers of the same depth, and ask which programs are fragile and why.

### A1. Tracr programs under Gaussian weight noise (baseline fragility)

**Question.** How fragile are six compiled Tracr programs under Gaussian noise on their weights, and which parameter groups matter?

**Code:** `code/run_sweeps2.py`, `code/fragility.py`  
**Results:** `results/results2.json`


| program | baseline | layers | d_model | params | nonzero_frac | abs | rel | zero_only | nonzero_only | group / embeddings | group / attn_query/key | group / attn_value/out | group / mlp_in | group / mlp_out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reverse | 1 | 4 | 65 | 58289 | 0.008389 | 5.80e-04 | 0.001609 | 6.32e-04 | 0.001601 | 5.85e-04 | 0.002917 | 0.01564 | 0.08507 | 8.37e-04 |
| sort_unique | 1 | 2 | 33 | 5805 | 0.02567 | 0.001192 | 0.002804 | 0.001204 | 0.003698 | 0.001586 | 0.002745 | 0.01677 | inf | 0.001436 |
| hist | 1 | 1 | 27 | 2391 | 0.04559 | 0.001399 | 0.002931 | 0.00164 | 0.003385 | 0.001737 | 0.003946 | 0.03311 | inf | 0.003433 |
| length | 1 | 1 | 27 | 2391 | 0.05813 | 2.91e-04 | 0.001172 | 2.92e-04 | 0.001601 | 4.00e-04 | 0.002143 | 0.02119 | inf | 0.07743 |
| frac_prevs | 1 | 2 | 19 | 2057 | 0.03597 | 0.001271 | 0.003001 | 0.001364 | 0.003635 | 0.001994 | 0.002216 | 0.006016 | 0.02983 | 0.002711 |
| sort_freq | 1 | 5 | 124 | 277618 | 0.01134 | nan | nan | nan | 1.07e-04 | 7.01e-04 | 0.001505 | 6.52e-04 | 4.59e-04 | nan |


Six Tracr programs compiled at N=8. `abs` = accuracy-vs-noise curve with noise on every weight (sigma50 = noise level where exact-sequence accuracy falls below 50%); `zero_only` / `nonzero_only` add noise only to the entries that are zero / nonzero in the compiled weights. Compiled weights are very sparse (0.3-6% nonzero) with values from 1 to 100. Noise of ~1e-4 to 1e-3 already breaks most programs; noise on the zero entries alone gives almost the same failure as noise on everything.

> **Takeaway.** Noise of about 1e-4 to 1e-3 per parameter already breaks most programs, even though the compiled weights are exact and roughly 94 to 99 percent of them are zero. Noise on the MLP input weights almost never matters (sigma50 infinite for three programs), while embeddings and attention value/output limit the tolerance. `sort_freq` is already inexact at the smallest noise tested.

![Figure A1.](figures/fragility_results.png)

**Figure A1.** (a) Exact-sequence accuracy collapses at 1e-4 to 1e-3 for six compiled programs (bands show seeds). (b) sigma50 with noise on all weights, only zero entries, or only non-zero entries. (c) sigma50 per parameter group (noise on one group at a time; darker is more fragile).

### A2. Compiled vs trained transformers, scale-free noise

**Question.** Is the fragility a property of the transformer architecture, or of compilation?

**Code:** `code/trained_baseline.py`, `code/compare_noise.py`, `code/noln.py`  
**Results:** `results/compare.json`, `results/noln.json`

`compare.json`


| program | variant | s50(curve) | baseline | rms / s50(curve) | mult / s50(curve) |
|---|---|---|---|---|---|
| compiled | hist / rms | 0.00575 |  |  |  |
| compiled | hist / mult | 0.002931 |  |  |  |
| compiled | sort_unique / rms | 0.004213 |  |  |  |
| compiled | sort_unique / mult | 0.002804 |  |  |  |
| compiled | reverse / rms | 0.00332 |  |  |  |
| compiled | reverse / mult | 0.001609 |  |  |  |
| trained | hist / 0 |  | 1 | 0.09022 | 0.08783 |
| trained | hist / 1 |  | 1 | 0.1331 | 0.1332 |
| trained | hist / 2 |  | 1 | 0.1416 | 0.1408 |
| trained | sort_unique / 0 |  | 1 | 0.2738 | 0.2463 |
| trained | sort_unique / 1 |  | 1 | 0.2682 | 0.2803 |
| trained | sort_unique / 2 |  | 1 | 0.2658 | 0.2671 |
| trained | reverse / 0 |  | 1 | 0.2776 | 0.2858 |
| trained | reverse / 1 |  | 1 | 0.2788 | 0.2687 |
| trained | reverse / 2 |  | 1 | 0.2595 | 0.2705 |


`noln.json`


| item | 0 / baseline | 0 / rms / s50(curve) | 0 / mult / s50(curve) | 1 / baseline | 1 / rms / s50(curve) | 1 / mult / s50(curve) |
|---|---|---|---|---|---|---|
| hist | 1 | 0.1866 | 0.1631 | 1 | 0.1613 | 0.1521 |
| sort_unique | 1 | 0.3437 | 0.3432 | 1 | 0.2905 | 0.2619 |


alpha50 = relative noise (sigma = alpha x RMS of each tensor) at which accuracy falls below 50%. Compiled models fail at alpha around 3e-3 to 6e-3, trained transformers of the same depth at 0.09 to 0.28 (roughly 15-170x more tolerant). `noln.json` repeats the trained models without LayerNorm (not the explanation).

> **Takeaway.** Trained transformers of the same depth tolerate relative noise 15 to 170 times larger than compiled ones, and removing LayerNorm does not change that. The fragility belongs to this *realization* of the computation, not to the architecture.

![Figure A2.](figures/compiled_vs_trained.png)

**Figure A2.** Exact-sequence accuracy against relative noise for compiled (red) and trained transformers (blue, with LayerNorm solid, without dashed) on `hist`, `sort_unique` and `reverse`. Compiled models fail one to two orders of magnitude earlier.

### A3. Which programs are fragile? static analysis + noise

**Question.** Which programs are fragile? A static analysis of what each program contains, plus noise.

**Code:** `code/audit_counting.py`  
**Results:** `results/audit_counting.json`


| item | clean | s50 | counting(SelectorWidth) | averaging(Aggregate) |
|---|---|---|---|---|
| map · 8 | 1 | 0.1416 | False | False |
| frac_prevs · 8 | 1 | 0.001805 | False | True |
| pair_balance · 8 | 1 | 8.34e-04 | False | True |
| length · 8 | 1 | 2.89e-04 | True | False |
| hist · 8 | 1 | 0.001511 | True | False |
| reverse · 8 | 1 | 7.19e-04 | True | True |
| sort_unique · 8 | 1 | 0.001299 | True | True |
| sort · 8 | 1 | 4.29e-04 | True | True |
| sort_freq · 8 | 1 | nan | True | True |
| map · 16 | 1 | 0.1083 | False | False |
| frac_prevs · 16 | 1 | 0.001489 | False | True |
| pair_balance · 16 | 1 | 8.76e-04 | False | True |
| length · 16 | 1 | 1.89e-04 | True | False |
| hist · 16 | 1 | 7.90e-04 | True | False |
| reverse · 16 | 1 | 2.01e-04 | True | True |
| sort · 16 | 1 | 1.37e-04 | True | True |
| sort_freq · 16 | 1 | nan | True | True |


Every fragile program contains an attention readout (counting via SelectorWidth, or plain numerical averaging). The program without attention (`map`) is robust. Averaging-only programs (frac_prevs, pair_balance) are about as fragile as hist, so the accurate label is 'precise analog readout from attention averages', with counting the extreme case. Correlational evidence; `mean_token` failed to compile.

> **Takeaway.** Every fragile program contains an attention readout: counting through `SelectorWidth`, or plain numerical averaging. The one program with no attention (`map`) is 100 to 500 times more tolerant. Averaging-only programs are as fragile as histograms, so the accurate description is *precise analog readout from attention averages*, with counting as the extreme case. This is correlational evidence across programs.

### A4. Control: programs with and without counting

**Question.** A control: do programs with and without counting differ?

**Code:** `code/control.py`  
**Results:** `results/control.json`


| item | map | 8 | length | 8 | hist | 8 | reverse | 8 | map | 16 | length | 16 | hist | 16 | reverse | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| value | 0.1449 | 3.02e-04 | 0.001519 | 7.15e-04 | 0.1086 | 1.88e-04 | 7.80e-04 | 1.98e-04 |  |  |  |  |  |  |  |  |


The per-token lookup program (no attention) has sigma50 around 0.11-0.15, 100-500x higher than the counting programs, and barely decays with N. The attention-without-counting control (shift by one) did not compile in Tracr.

> **Takeaway.** The per-token lookup has sigma50 around 0.11 to 0.15 and barely decays with length, against 1e-4 to 1e-3 for counting programs. The matching attention-without-counting control (shift by one) did not compile in Tracr, so the causal claim rests on the across-program comparison.
