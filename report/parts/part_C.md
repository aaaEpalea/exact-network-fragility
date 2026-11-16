
## Part C. Hardening Tracr-compiled models

If compiled networks are fragile, can they be made tolerant without losing exactness? We try exact rescalings that leave the function unchanged, lowering the attention scale, adding snapping layers, noise-aware fine-tuning, sensitivity-balanced rescaling, and training against a certificate. The honest summary is a few-fold gain at best, and no change in the decay with length.

### C1. Hand-hardening stack on Tracr models

**Question.** How much can exact, function-preserving rewrites of the compiled weights help?

**Code:** `code/hardening.py`, `code/run_hardening2.py`  
**Results:** `results/hardening2.json`


<details><summary>Full table (50 rows)</summary>

| program | variant | clean | abs / s50(curve) | nz / s50(curve) |
|---|---|---|---|---|
| hist | base | 1 | 0.001381 | 0.003121 |
| hist | balance | 1 | 0.001527 | 0.003523 |
| hist | c=0.25 | 1 | 0.00476 | 0.0115 |
| hist | balance+c=0.5 | 1 | 0.003151 | 0.007276 |
| hist | balance+c=0.25 | 1 | 0.006156 | 0.01412 |
| hist | balance+c=0.1 | 1 | 0.008346 | 0.02029 |
| hist | snap_all | 1 | 7.69e-04 | 0.001389 |
| hist | balance+snap | 1 | 8.12e-04 | 0.001305 |
| hist | ideal_clip | 1 | 0.001878 | 0.004151 |
| hist | balance+ideal_clip | 1 | 0.002626 | 0.006355 |
| sort_unique | base | 1 | 0.001141 | 0.003272 |
| sort_unique | balance | 1 | 0.001476 | 0.002832 |
| sort_unique | c=0.25 | 1 | 0.001329 | 0.007252 |
| sort_unique | balance+c=0.5 | 1 | 0.003144 | 0.005865 |
| sort_unique | balance+c=0.25 | 1 | 0.006048 | 0.01211 |
| sort_unique | balance+c=0.1 | 1 | 0.007788 | 0.03058 |
| sort_unique | snap_all | 1 | 5.38e-04 | 9.00e-04 |
| sort_unique | balance+snap | 1 | 5.45e-04 | 9.33e-04 |
| sort_unique | ideal_clip | 1 | 0.001141 | 0.003272 |
| sort_unique | balance+ideal_clip | 1 | 0.001476 | 0.002832 |
| reverse | base | 1 | 6.10e-04 | 0.001658 |
| reverse | balance | 1 | 6.65e-04 | 0.001853 |
| reverse | c=0.25 | 1 | 0.001578 | 0.007433 |
| reverse | balance+c=0.5 | 1 | 0.001355 | 0.003738 |
| reverse | balance+c=0.25 | 1 | 0.002603 | 0.0076 |
| reverse | balance+c=0.1 | 1 | 0.004359 | 0.01689 |
| reverse | snap_all | 1 | 2.56e-04 | 7.51e-04 |
| reverse | balance+snap | 1 | 2.52e-04 | 6.89e-04 |
| reverse | ideal_clip | 1 | 6.10e-04 | 0.001658 |
| reverse | balance+ideal_clip | 1 | 6.65e-04 | 0.001853 |
| length | base | 1 | 2.69e-04 | 0.001411 |
| length | balance | 1 | 3.50e-04 | 0.001376 |
| length | c=0.25 | 1 | 9.73e-04 | 0.005702 |
| length | balance+c=0.5 | 1 | 6.73e-04 | 0.002704 |
| length | balance+c=0.25 | 1 | 0.001236 | 0.004819 |
| length | balance+c=0.1 | 1 | 0.002297 | 0.01083 |
| length | snap_all | 1 | 1.05e-04 | 3.76e-04 |
| length | balance+snap | 1 | 1.20e-04 | 3.71e-04 |
| length | ideal_clip | 1 | 2.69e-04 | 0.001411 |
| length | balance+ideal_clip | 1 | 3.50e-04 | 0.001376 |
| frac_prevs | base | 1 | 0.001295 | 0.003856 |
| frac_prevs | balance | 1 | 0.001697 | 0.005559 |
| frac_prevs | c=0.25 | 1 | 0.004109 | 0.01499 |
| frac_prevs | balance+c=0.5 | 1 | 0.003358 | 0.01078 |
| frac_prevs | balance+c=0.25 | 1 | 0.005398 | 0.02083 |
| frac_prevs | balance+c=0.1 | 0.08 |  |  |
| frac_prevs | snap_all | 1 | 7.38e-04 | 0.001973 |
| frac_prevs | balance+snap | 1 | 7.93e-04 | 0.002247 |
| frac_prevs | ideal_clip | 1 | 0.001295 | 0.003856 |
| frac_prevs | balance+ideal_clip | 1 | 0.001697 | 0.005559 |

</details>


Variants applied to compiled weights: exact q/k + MLP rescaling (`balance`), attention scale `c`, snap layers, idealized weight-free clip. Rescaling alone gives only 1.1-1.3x; dense snap layers HURT (0.4-0.6x) because they add noisy parameters; lowering the attention scale gives 3-4x; combined 4-8x. `abs` = dense noise, `nz` = noise on nonzero entries only.

> **Takeaway.** Rescaling alone gives 1.1 to 1.3 times; adding dense snap layers hurts (0.4 to 0.6 times) because they add noisy parameters; lowering the attention scale gives 3 to 4 times and the combination 4 to 8 times. One variant (`frac_prevs`, scale 0.1) is not exact and is excluded.

### C2. Noise-aware fine-tuning from the compiled initialization

**Question.** Can noise-aware fine-tuning starting from the compiled weights fix it?

**Code:** `code/run_noise_train.py`  
**Results:** `results/noise_train.json`


| program | variant | exhaustive_ok | exhaustive_total | nonzero_frac | abs / s50(curve) | rms / s50(curve) |
|---|---|---|---|---|---|---|
| hist | from_base | 303708 | 335922 | 0.8361 | 0.001369 | 0.005187 |
| hist | from_balance+c=0.25 | 335916 | 335922 | 0.8361 | 0.005625 | 0.01423 |
| sort_unique | from_base | 1946 | 1956 | 0.8953 | 0.01937 | 0.01302 |
| sort_unique | from_balance+c=0.25 | 1956 | 1956 | 0.8953 | 0.01014 | 0.009431 |
| reverse | from_base | 23185 | 335922 | 0.9239 | 0.01574 | 0.02389 |
| reverse | from_balance+c=0.25 | 335922 | 335922 | 0.9239 | 0.01149 | 0.04778 |


Fine-tuning compiled weights with injected noise. From the raw compiled weights it destroys exactness (e.g. reverse N=8 only 23,185 of 335,922 sequences correct); from the rescaled c=0.25 initialization it keeps exactness (exhaustively verified at N=8) and gives 4-19x. Densifies the weights (84-92% nonzero).

> **Takeaway.** From the raw compiled weights, fine-tuning destroys exactness (for example `reverse`: only 23,185 of 335,922 sequences correct); from the rescaled, lowered-attention-scale initialization it keeps exactness (verified exhaustively at N=8) and gives 4 to 19 times, at the price of densifying the weights to 84 to 92 percent non-zero.

![Figure C.](figures/hardening_summary.png)

**Figure C.** Hardening closes about 2 to 14 times of the 15 to 170 times gap: compiled (red), compiled plus hardening (orange: rescale, attention temperature, noise-aware fine-tune; the asterisk marks a case with 6 of 335,922 sequences wrong after fine-tuning) and trained from scratch (blue, 3 seeds).

### C3. Sensitivity-balanced compilation (symmetry rescaling under noise)

**Question.** Can an optimizer find the best exact rescaling under noise (sensitivity balancing)?

**Code:** `code/balance.py`, `code/balance_scaling.py`  
**Results:** `results/balance_scaling.json`, `results/balance_32_plain.json`

`balance_scaling.json`


| case | tag | N | S | clean_before | clean_after | max_output_diff | s50_before | s50_after | gain | theta | log_scale_range |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plain c=0.25 · 16 | plain c=0.25 N=16 | 16 | 307 | 1 | 1 | 4.68e-05 | 6.49e-04 | 0.002026 | 3.122 | [6 values, mean 0.7204] | -2.016, 1.962 |
| plain c=0.25 · 32 | plain c=0.25 N=32 | 32 | 1228 | 1 | 1 | 2.20e-04 | 2.04e-04 | 7.66e-04 | 3.754 | [6 values, mean 0.8069] | -2.438, 2.105 |
| plain c=0.25 · 48 | plain c=0.25 N=48 | 48 | 2764 | 1 | 1 | 5.03e-04 | 1.11e-04 | 4.23e-04 | 3.802 | [6 values, mean 0.8176] | -2.546, 2.364 |
| chunk g=4 c=1 · 16 | chunk g=4 c=1 N=16 | 16 | 100 | 1 | 1 | 3.17e-04 | 3.31e-04 | 9.27e-04 | 2.796 | [16 values, mean 0.3307] | -0.6803, 2.078 |


`balance_32_plain.json`


| case | tag | N | S | clean_before | clean_after | max_output_diff | s50_before | s50_after | gain | theta | log_scale_range |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plain | plain | 32 | 1228 | 1 | 1 | 1.49e-04 | 1.01e-04 | 2.98e-04 | 2.961 | [6 values, mean 0.8144] | -2.24, 2.479 |


Optimizes only exact symmetries (residual-dimension scales, q/k, v/out, MLP hidden scale) under injected noise, so the function is unchanged (clean accuracy stays 1.00, max output change ~5e-4). Gain is a constant 2.8-3.8x across N=16-48; the log-log slope moves only from -1.6 to -1.4. The chunked N=48 run did not finish.

> **Takeaway.** Optimizing only exact symmetries gives a constant 2.8 to 3.8 times gain at N=16 to 48 with no change in function (max output change about 5e-4), and the slope with N moves only from -1.6 to -1.4.

### C4. Balancing generalized to other programs

**Question.** Does balancing carry over to other programs?

**Code:** `code/balance2.py`  
**Results:** `results/balance2.json`


| case | tag | name | N | c | S | clean_after | s50_before | s50_after | gain | clean_before | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reverse · 12 · 1.0 | reverse N=12 c=1.0 | reverse | 12 | 1 | 216 | 1 | 3.47e-04 | 0.001593 | 4.588 |  |  |
| reverse · 12 · 0.25 | reverse N=12 c=0.25 | reverse | 12 | 0.25 | 216 | 1 | 4.91e-04 | 0.003865 | 7.876 |  |  |
| reverse · 16 · 1.0 | reverse N=16 c=1.0 | reverse | 16 | 1 | 384 | 1 | 2.05e-04 | 0.001299 | 6.335 |  |  |
| reverse · 16 · 0.25 | reverse N=16 c=0.25 | reverse | 16 | 0.25 | 384 | 1 | 3.26e-04 | 0.002585 | 7.92 |  |  |
| hist · 16 · 1.0 | hist N=16 c=1.0 | hist | 16 | 1 | 384 | 1 | 6.33e-04 | 0.002421 | 3.824 |  |  |
| hist · 16 · 0.25 | hist N=16 c=0.25 | hist | 16 | 0.25 | 384 | 1 | 8.49e-04 | 0.004704 | 5.538 |  |  |
| hist · 24 · 1.0 | hist N=24 c=1.0 | hist | 24 | 1 | 864 | 1 | 3.62e-04 | 0.001491 | 4.118 |  |  |
| hist · 24 · 0.25 | hist N=24 c=0.25 | hist | 24 | 0.25 | 864 | 1 | 4.53e-04 | 0.002456 | 5.419 |  |  |
| frac_prevs · 16 · 1.0 | frac_prevs N=16 c=1.0 | frac_prevs | 16 | 1 | 384 | 1 | 0.001406 | 0.003762 | 2.677 |  |  |
| frac_prevs · 16 · 0.25 | frac_prevs N=16 c=0.25 |  |  |  |  |  |  |  |  | 0.5952 | True |
| sort_unique · 8 · 1.0 | sort_unique N=8 c=1.0 | sort_unique | 8 | 1 | 100 | 1 | 0.001221 | 0.00475 | 3.889 |  |  |
| sort_unique · 8 · 0.25 | sort_unique N=8 c=0.25 | sort_unique | 8 | 0.25 | 100 | 1 | 0.001287 | 0.008146 | 6.331 |  |  |


reverse, hist, sort_unique, frac_prevs at N=8-24. Balancing alone 2.7-6.3x; with attention scale c=0.25 the stack reaches 7-13x over default on five of six settings (frac_prevs 2.7x). The decay with N is unchanged.

> **Takeaway.** Balancing alone gives 2.7 to 6.3 times and, with attention scale 0.25, 7 to 13 times on five of six settings. The decay with N is unchanged.

### C5. Certificate-driven training with a general interval bound (negative result)

**Question.** Can a network be trained to maximize a provable radius under a general interval bound?

**Code:** `code/ibp_train.py`  
**Results:** `results/ibp_train.json`


| item | cert_compiled | cert_trained | tube_trained | clean | clean_len400 | sigma50_compiled | sigma50_trained | alpha50_compiled | alpha50_trained |
|---|---|---|---|---|---|---|---|---|---|
| maj(0>1) | 0 | 0 | None | 1 | 0.5312 | 0.03648 | 0.03108 | 0.09292 | 0.07851 |
| dyck(0,1) | 0 | 0 | None | 1 | 0.8125 | 0.0519 | 0.02621 | 0.1236 | 0.07571 |


Training the digital network's weights to maximize a provable radius under a general (loose) interval bound. The bound certifies radius 0 for the compiled weights and training never reached a certified radius; trained weights lost exactness on long sequences (length-400 accuracy 0.53 and 0.81) and tolerance got worse. NEGATIVE.

> **Takeaway.** No. The general bound certifies radius 0 for the compiled weights, training never reached a certified radius, trained weights lost exactness on long sequences (length-400 accuracy 0.53 and 0.81) and tolerance got worse. A negative result.
