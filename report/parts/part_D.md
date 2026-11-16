
## Part D. Digital (restoring-state) designs

Instead of repairing analog readouts, we replace them: the state is a one-hot vector that is snapped back to a valid value at every step, so errors cannot accumulate. We compare designs on the same tasks, add a bilinear (second-order) form with no extra parameters and weight-tied composition of many programs, and compare with trained GRUs. **This family of constructions follows Omlin and Giles (1996); see Part M.**

### D1. Same task, different designs: scaling of noise tolerance

**Question.** Same task, different designs: how does noise tolerance scale?

**Code:** `code/digital_counter.py`, `code/lsrl_counter.py`  
**Results:** `results/digital_counter.json`, `results/lsrl_counter.json`

`digital_counter.json`


| item | digital_snap | analog_nosnap |
|---|---|---|
| 8 | 0.1471 | 0.5116 |
| 16 | 0.09758 | 0.267 |
| 32 | 0.0801 | 0.144 |
| 64 | 0.07246 | 0.07493 |
| 128 | 0.04972 | 0.03906 |


`lsrl_counter.json`


| item | s50(curve) |
|---|---|
| value | nan |


Counting tokens: hand-built digital one-hot counter with per-step snapping (slope ~-0.36) vs no snapping (~-0.93) vs LSRL linear-state accumulator (~-1.96, 1/N^2) vs the Tracr attention counter (~-1.3 to -1.6). Toy designs; not directly comparable tasks.

> **Takeaway.** A hand-built digital one-hot counter with per-step snapping has a slope of about -0.36; without snapping about -0.93; a linear-state accumulator decays as 1/N squared; the Tracr attention counter about -1.3 to -1.6. These are toy designs on slightly different tasks, so they show tendencies, not a ranking.

![Figure D1.](figures/design_comparison.png)

**Figure D1.** Same task (counting tokens), different designs: digital one-hot counter with per-step snapping stays high; a linear-state accumulator decays as 1/N squared; Tracr decays roughly as N to the minus 1.5.

### D2. Prefix-histogram: digital vs Tracr (causal)

**Question.** Does the advantage hold on a richer causal program, the prefix histogram?

**Code:** `code/digital_prefix_hist.py`  
**Results:** `results/digital_prefix_hist.json`


| item | clean | s50 | params |
|---|---|---|---|
| digital_snap · 8 | 1 | 0.02736 | 28026 |
| digital_nosnap · 8 | 1 | 0.01765 | 28026 |
| digital_snap · 16 | 1 | 0.01579 | 97818 |
| digital_nosnap · 16 | 1 | 0.006228 | 97818 |
| digital_snap · 32 | 1 | 0.0154 | 364122 |
| digital_nosnap · 32 | 1 | 0.00235 | 364122 |
| digital_snap · 64 | 1 | 0.01493 | 1403610 |
| digital_nosnap · 64 | 1 | 6.46e-04 | 1403610 |
| digital_snap · 128 | 1 | 0.01308 | 5510106 |
| digital_nosnap · 128 | 1 | 1.52e-04 | 5510106 |
| tracr_c1.0 · 8 | 1 | 0.00107 | 2391 |
| tracr_c0.25 · 8 | 1 | 0.003841 | 2391 |
| tracr_c1.0 · 16 | 1 | 6.07e-04 | 5519 |
| tracr_c0.25 · 16 | 1 | 7.93e-04 | 5519 |
| tracr_c1.0 · 32 | 1 | 1.02e-04 | 15615 |
| tracr_c0.25 · 32 | 1 | 1.02e-04 | 15615 |


Hand-built digital design with snapping is nearly flat in N (about 1.3-1.6e-2 from N=16 to 128); removing snapping gives about N^-1.7; Tracr causal is ~1e-4 at N=32. The digital design is larger (parameters grow as N^2).

> **Takeaway.** The digital design with snapping is nearly flat in N (1.3 to 1.6e-2 from N=16 to 128); removing snapping gives about N to the minus 1.7; Tracr causal is about 1e-4 at N=32. The price: parameters grow as N squared.

![Figure D2.](figures/digital_vs_tracr.png)

**Figure D2.** Prefix histogram: the digital design with snapping is nearly flat in N; removing the snapping makes it decay; Tracr causal models are about 1e-4 at N=32.

### D3. Second program: majority-so-far

**Question.** A second program: majority-so-far.

**Code:** `code/second_program.py`  
**Results:** `results/second_program.json`


| item | clean | s50 | params |
|---|---|---|---|
| digital_snap · 8 | 1 | 0.0653 | 3334 |
| digital_nosnap · 8 | 1 | 0.01546 | 3334 |
| digital_snap · 16 | 1 | 0.03521 | 11750 |
| digital_nosnap · 16 | 1 | 0.006315 | 11750 |
| digital_snap · 32 | 1 | 0.03392 | 43942 |
| digital_nosnap · 32 | 1 | 0.001656 | 43942 |
| digital_snap · 64 | 1 | 0.02089 | 169766 |
| digital_nosnap · 64 | 1 | 4.25e-04 | 169766 |
| tracr_c1.0 · 8 | 1 | 0.001033 | 18705 |
| tracr_c0.25 · 8 | 1 | 0.002897 | 18705 |
| tracr_c1.0 · 16 | 1 | 1.90e-04 | 83393 |
| tracr_c0.25 · 16 | 1 | 1.93e-04 | 83393 |
| tracr_c1.0 · 32 | 1 | nan | 504993 |
| tracr_c0.25 · 32 | 1 | nan | 504993 |


Signed one-hot counter with comparison readout. Digital with snapping decays slowly (6.5e-2 to 2.1e-2, N=8-64) and is exact on all 7,776 length-5 sequences; Tracr causal is 1.0e-3 at N=8 and below the smallest tested noise at N=32. Here digital is also smaller (about 44k vs 505k parameters at N=32).

> **Takeaway.** The signed one-hot counter with snapping decays slowly (6.5e-2 to 2.1e-2 over N=8 to 64) and is exact on all 7,776 length-5 sequences; Tracr causal is 1.0e-3 at N=8 and below the smallest noise tested at N=32. Here the digital design is also smaller (about 44k against 505k parameters at N=32).

### D4. Width-matched redundancy baseline for Tracr

**Question.** Is the digital advantage just redundancy? Compare with r copies of the Tracr model at equal parameter count.

**Code:** `code/width_matched.py`  
**Results:** `results/width_matched.json`


| item | s50 | params | clean |
|---|---|---|---|
| N16 · r1 | 6.62e-04 | 5519 | 1 |
| N16 · r4 | 8.35e-04 | 22076 | 1 |
| N16 · r18 | 0.001391 | 99342 | 1 |
| N32 · r1 | 8.23e-05 | 15615 | 1 |
| N32 · r4 | 1.70e-04 | 62460 | 1 |
| N32 · r23 | 5.39e-04 | 359145 | 1 |


r independent compiled Tracr copies with majority-style output summing, sized to match the digital design's parameter count. Redundancy helps (2x at N=16, 6.6x at N=32) but the digital design remains 11-28x better at equal parameters.

> **Takeaway.** Redundancy helps Tracr (2 times at N=16, 6.6 times at N=32) but the digital design remains 11 to 28 times better at equal parameters.

### D5. Digital design under the relative (RMS) noise protocol

**Question.** Does the conclusion survive the relative (scale-free) noise protocol?

**Code:** `code/digital_rel.py`  
**Results:** `results/digital_rel.json`


| item | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| value | 0.107 | 0.0809 | 0.07228 | 0.06114 |


alpha50 of the digital prefix-histogram: 0.107, 0.081, 0.072, 0.061 for N=8, 16, 32, 64, comparable to trained transformers (0.16-0.20 at N=8, 0.034-0.037 at N=32 in the early under-trained runs).

> **Takeaway.** Yes. alpha50 of the digital prefix histogram is 0.107, 0.081, 0.072, 0.061 at N=8, 16, 32, 64, comparable to early under-trained transformers.

### D6. Margin tuning of the digital designs (negative)

**Question.** Can snap thresholds and a global scale be tuned to improve the digital designs?

**Code:** `code/digital_tune.py`  
**Results:** `results/digital_tune.json`


<details><summary>Full table (66 rows)</summary>

| case | value |
|---|---|
| hist · lo0.25 · hi0.75 · k1.0 | 0.01785 |
| hist · lo0.25 · hi0.75 · k2.0 | 0.01772 |
| hist · lo0.25 · hi0.75 · k4.0 | 0.01778 |
| hist · lo0.1 · hi0.5 · k1.0 | 0.005875 |
| hist · lo0.1 · hi0.5 · k2.0 | 0.005891 |
| hist · lo0.1 · hi0.5 · k4.0 | 0.00594 |
| hist · lo0.5 · hi0.9 · k1.0 | 0.01189 |
| hist · lo0.5 · hi0.9 · k2.0 | 0.01265 |
| hist · lo0.5 · hi0.9 · k4.0 | 0.01291 |
| hist · lo0.4 · hi0.6 · k1.0 | 0.01447 |
| hist · lo0.4 · hi0.6 · k2.0 | 0.01436 |
| hist · lo0.4 · hi0.6 · k4.0 | 0.01424 |
| hist · lo0.3 · hi0.7 · k1.0 | 0.01766 |
| hist · lo0.3 · hi0.7 · k2.0 | 0.01778 |
| hist · lo0.3 · hi0.7 · k4.0 | 0.01767 |
| hist · lo0.2 · hi0.8 · k1.0 | 0.01695 |
| hist · lo0.2 · hi0.8 · k2.0 | 0.01728 |
| hist · lo0.2 · hi0.8 · k4.0 | 0.01728 |
| hist · lo0.1 · hi0.9 · k1.0 | 0.01209 |
| hist · lo0.1 · hi0.9 · k2.0 | 0.01319 |
| hist · lo0.1 · hi0.9 · k4.0 | 0.01319 |
| hist · lo0.45 · hi0.55 · k1.0 | 0.007614 |
| hist · lo0.45 · hi0.55 · k2.0 | 0.00769 |
| hist · lo0.45 · hi0.55 · k4.0 | 0.007943 |
| hist · N8 · default | 0.02617 |
| hist · N8 · tuned | 0.02617 |
| hist · N8 · tuned_alpha | 0.09877 |
| hist · N32 · default | 0.01545 |
| hist · N32 · tuned | 0.01545 |
| hist · N32 · tuned_alpha | 0.07128 |
| hist · N64 · default | 0.01127 |
| hist · N64 · tuned | 0.01127 |
| hist · N64 · tuned_alpha | 0.05859 |
| majority · lo0.25 · hi0.75 · k1.0 | 0.0434 |
| majority · lo0.25 · hi0.75 · k2.0 | 0.04677 |
| majority · lo0.25 · hi0.75 · k4.0 | 0.04693 |
| majority · lo0.1 · hi0.5 · k1.0 | 0.01531 |
| majority · lo0.1 · hi0.5 · k2.0 | 0.01698 |
| majority · lo0.1 · hi0.5 · k4.0 | 0.01698 |
| majority · lo0.5 · hi0.9 · k1.0 | 0.04249 |
| majority · lo0.5 · hi0.9 · k2.0 | 0.04529 |
| majority · lo0.5 · hi0.9 · k4.0 | 0.04587 |
| majority · lo0.4 · hi0.6 · k1.0 | 0.02366 |
| majority · lo0.4 · hi0.6 · k2.0 | 0.02399 |
| majority · lo0.4 · hi0.6 · k4.0 | 0.02399 |
| majority · lo0.3 · hi0.7 · k1.0 | 0.03897 |
| majority · lo0.3 · hi0.7 · k2.0 | 0.0406 |
| majority · lo0.3 · hi0.7 · k4.0 | 0.0409 |
| majority · lo0.2 · hi0.8 · k1.0 | 0.04126 |
| majority · lo0.2 · hi0.8 · k2.0 | 0.04646 |
| majority · lo0.2 · hi0.8 · k4.0 | 0.04624 |
| majority · lo0.1 · hi0.9 · k1.0 | 0.03023 |
| majority · lo0.1 · hi0.9 · k2.0 | 0.03194 |
| majority · lo0.1 · hi0.9 · k4.0 | 0.03263 |
| majority · lo0.45 · hi0.55 · k1.0 | 0.01317 |
| majority · lo0.45 · hi0.55 · k2.0 | 0.01334 |
| majority · lo0.45 · hi0.55 · k4.0 | 0.0135 |
| majority · N8 · default | 0.05369 |
| majority · N8 · tuned | 0.06243 |
| majority · N8 · tuned_alpha | 0.149 |
| majority · N32 · default | 0.03365 |
| majority · N32 · tuned | 0.04213 |
| majority · N32 · tuned_alpha | 0.08982 |
| majority · N64 · default | 0.02113 |
| majority · N64 · tuned | 0.02215 |
| majority · N64 · tuned_alpha | 0.08156 |

</details>


Snap thresholds and a global state scale, all exact-preserving. Best gain 1.0x (prefix-hist) and 1.1x (majority): the defaults are already near-optimal for these knobs. NEGATIVE.

> **Takeaway.** No: the best gains are 1.0 times and 1.1 times; the defaults are already near-optimal for these knobs. A negative result.

### D7. Weight-free clip activation instead of four dense snap matrices

**Question.** Replace four dense snap matrices by a weight-free clip activation.

**Code:** `code/clip_activation.py`  
**Results:** `results/clip_activation.json`


| item | clean_new | params_old | params_new | abs / snap_matrices | abs / clip_activation | abs / gain | rel / snap_matrices | rel / clip_activation | rel / gain |
|---|---|---|---|---|---|---|---|---|---|
| maj(0>1) · N16 | 1 | 18975 | 14553 | 0.03676 | 0.07045 | 1.916 | 0.09301 | 0.1434 | 1.542 |
| dyck(0,1) · N16 | 1 | 6030 | 4698 | 0.04848 | 0.08364 | 1.725 | 0.1037 | 0.1417 | 1.366 |
| maj(0>1) · N32 | 1 | 70655 | 53625 | 0.02724 | 0.05298 | 1.945 | 0.0804 | 0.09671 | 1.203 |
| dyck(0,1) · N32 | 1 | 20094 | 15402 | 0.03142 | 0.05859 | 1.865 | 0.08053 | 0.09892 | 1.228 |


Restoring step s' = clip(2z - 0.5, 0, 1) as an activation (no noisy parameters). 20-25% fewer parameters, still exact, and 1.7-1.9x higher sigma50 (1.2-1.5x in relative alpha50).

> **Takeaway.** 20 to 25 percent fewer parameters, still exact, and 1.7 to 1.9 times higher sigma50 (1.2 to 1.5 times in relative alpha50). Removing noisy parameters from the restoring step helps.

### D8. Bilinear (second-order) design and weight-tied composition

**Question.** A bilinear (second-order) form, and weight-tied composition of many programs.

**Code:** `code/ideas_round.py`, `code/bilinear_lib.py`  
**Results:** `results/ideas_round.json`


| item | clean | sigma50 | alpha50 | params | per_head_s50 | joint_s50 |
|---|---|---|---|---|---|---|
| bilinear · maj(0>1) · N16 | 1 | 0.1299 | 0.6149 | 6633 |  |  |
| bilinear · maj(0>1) · N32 | 1 | 0.1003 | 0.6336 | 25545 |  |  |
| bilinear · maj(0>1) · N64 | 1 | 0.09034 | 0.7274 | 100233 |  |  |
| bilinear · dyck(0,1) · N16 | 1 | 0.1691 | 0.6557 | 1998 |  |  |
| bilinear · dyck(0,1) · N32 | 1 | 0.1373 | 0.6732 | 7038 |  |  |
| bilinear · dyck(0,1) · N64 | 1 | 0.1117 | 1.01 | 26334 |  |  |
| tied · k1 | 1 |  |  | 3483 | 0.08439 | 0.08439 |
| tied · k2 | 1 |  |  | 3501 | 0.08245 | 0.06807 |
| tied · k4 | 1 |  |  | 3537 | 0.08063 | 0.05953 |
| tied · k8 | 1 |  |  | 3609 | 0.0792 | 0.05507 |
| tied · k15 | 1 |  |  | 3735 | 0.07956 | 0.05311 |
| tied · k30 | 1 |  |  | 4005 | 0.08054 | 0.05283 |


Bilinear: z = sum_v x_v W_v s with no hidden gates, biases or snap matrices; relative tolerance alpha50 rises from about 0.1 to 0.6-1.0 and stays nearly flat to N=64. Tied: k template programs share one set of transition matrices plus a tiny role table; 30 exact programs fit in about 4,000 parameters with flat per-head tolerance (joint tolerance levels off at about 5e-2).

> **Takeaway.** The bilinear form z = sum over tokens of W_v s has no gates, biases or snap matrices; relative tolerance rises from about 0.1 to 0.6 to 1.0 and stays flat to N=64. The tied variant shares transition matrices between template programs: 30 exact programs fit in about 4,000 parameters with flat per-head tolerance. Note the bilinear form is the second-order recurrent network of Omlin and Giles; the tied design only works for programs that are instances of one template.

![Figure D8.](figures/diagram_tied.png)

**Figure D8.** Weight-tied design: the transition matrices (up, down, hold) are shared; each program only selects which role each token uses.

### D9. Tied + bilinear combined

**Question.** Combine tied and bilinear.

**Code:** `code/tied_bilinear.py`  
**Results:** `results/tied_bilinear.json`


| item | clean | params | abs / per_head | abs / joint | rel / per_head | rel / joint |
|---|---|---|---|---|---|---|
| k1 | 1 | 3384 | 0.1229 | 0.1229 | 0.4277 | 0.4277 |
| k4 | 1 | 3438 | 0.1206 | 0.09495 | 0.4516 | 0.3464 |
| k15 | 1 | 3636 | 0.1165 | 0.08996 | 0.4538 | 0.2418 |
| k30 | 1 | 3906 | 0.1195 | 0.08917 | 0.4638 | 0.227 |


Role-gated multiplicative update, no hidden gates and no biases. About 3,400-3,900 parameters for k=1-30 exact programs; per-head sigma50 about 0.12 regardless of k.

> **Takeaway.** A role-gated multiplicative update with no hidden gates or biases: about 3,400 to 3,900 parameters for 1 to 30 exact programs, per-head sigma50 about 0.12 regardless of k.

### D10. Matched comparison: bilinear designs vs trained GRUs

**Question.** Matched comparison with trained GRUs: same programs, distribution, evaluation sequences and noise protocols.

**Code:** `code/matched_gru.py`  
**Results:** `results/matched_gru.json`


| item | clean | sigma50 | alpha50 | params |
|---|---|---|---|---|
| majority · N16 · bilinear (compiled) | 1 | 0.1216 | 0.5987 | 6633 |
| majority · N16 · GRU32 seed0 | 1 | 0.175 | 0.6113 | 3810 |
| majority · N16 · GRU32 seed1 | 1 | 0.1512 | 0.5529 | 3810 |
| majority · N16 · GRU64 seed0 | 1 | 0.1908 | 0.8588 | 13762 |
| majority · N16 · GRU64 seed1 | 1 | 0.1987 | 0.8769 | 13762 |
| majority · N32 · bilinear (compiled) | 1 | 0.1147 | 0.6936 | 25545 |
| majority · N32 · GRU32 seed0 | 0.995 | 0.08171 | 0.3253 | 3810 |
| majority · N32 · GRU32 seed1 | 0.995 | 0.07107 | 0.336 | 3810 |
| majority · N32 · GRU64 seed0 | 1 | 0.1053 | 0.5098 | 13762 |
| majority · N32 · GRU64 seed1 | 0.995 | 0.1095 | 0.487 | 13762 |
| dyck · N16 · bilinear (compiled) | 1 | 0.2409 | 0.6551 | 1998 |
| dyck · N16 · GRU32 seed0 | 1 | 0.378 | 1.131 | 3810 |
| dyck · N16 · GRU32 seed1 | 1 | 0.3052 | 1.059 | 3810 |
| dyck · N16 · GRU64 seed0 | 1 | 0.3437 | 1.474 | 13762 |
| dyck · N16 · GRU64 seed1 | 1 | 0.2874 | 1.338 | 13762 |
| dyck · N32 · bilinear (compiled) | 1 | 0.1429 | 0.8983 | 7038 |
| dyck · N32 · GRU32 seed0 | 1 | 0.2441 | 0.8614 | 3810 |
| dyck · N32 · GRU32 seed1 | 1 | 0.2354 | 0.8394 | 3810 |
| dyck · N32 · GRU64 seed0 | 1 | 0.2628 | 1.132 | 13762 |
| dyck · N32 · GRU64 seed1 | 1 | 0.2281 | 1.008 | 13762 |


Same programs, token distribution, eval sequences and noise protocols (sigma50 absolute, alpha50 relative); GRUs have 32 or 64 hidden units, 2 seeds each, 2,500 training steps. Majority: bilinear is below the GRUs at N=16 (sigma50 0.12 vs 0.15-0.20) but above them at N=32 (0.115 vs 0.07-0.11, where the GRUs decay and are 99.5% accurate). Dyck: GRUs are 1.2-1.9x higher in sigma50 (N=16: 0.29-0.38 vs 0.24; N=32: 0.23-0.26 vs 0.14) and similar or higher in alpha50. So the bilinear compiled design is competitive with small trained GRUs, not uniformly better. Single token distribution per task; 6 noise draws.

> **Takeaway.** The bilinear design is competitive with small trained GRUs but not uniformly better: on majority it is below them at N=16 and above them at N=32 (where the GRUs decay); on dyck the GRUs are 1.2 to 1.9 times higher in sigma50. One token distribution per task and six noise draws.

### D11. Trained GRU baselines

**Question.** Trained GRU baselines.

**Code:** `code/trained_gru.py`, `code/gru_long.py`  
**Results:** `results/trained_gru_128.json`, `results/trained_gru_long_256.json`

`trained_gru_128.json`


| item | clean | params | abs | rel |
|---|---|---|---|---|
| prefix_hist · 8 · 0 | 0.995 | 53001 | 0.1167 | 0.5471 |
| prefix_hist · 8 · 1 | 0.99 | 53001 | 0.1095 | 0.5511 |
| prefix_hist · 16 · 0 | 0.945 | 54033 | 0.03725 | 0.2067 |
| prefix_hist · 16 · 1 | 0.965 | 54033 | 0.0377 | 0.1996 |
| prefix_hist · 32 · 0 | 0.705 | 56097 | 0.01163 | 0.04997 |
| prefix_hist · 32 · 1 | 0.575 | 56097 | 0.007904 | 0.03773 |
| majority · 8 · 0 | 1 | 52098 | 0.3725 | inf |
| majority · 8 · 1 | 1 | 52098 | 0.3799 | 1.88 |
| majority · 16 · 0 | 1 | 52098 | 0.1659 | 0.9449 |
| majority · 16 · 1 | 1 | 52098 | 0.223 | 1.089 |
| majority · 32 · 0 | 1 | 52098 | 0.08344 | 0.519 |
| majority · 32 · 1 | 1 | 52098 | 0.126 | 0.7725 |


`trained_gru_long_256.json`


| item | clean | params | abs | rel |
|---|---|---|---|---|
| prefix_hist · 16 · 0 | 0.995 | 206353 | 0.06578 | 0.4258 |
| prefix_hist · 32 · 0 | 0.98 | 210465 | 0.03145 | 0.1979 |


GRU(128) trained 3,000 steps on prefix-hist and majority; prefix-hist is not learned exactly at N=32 (clean 0.58-0.71) in that budget, majority is exact at all N and more tolerant than the early digital designs. `trained_gru_long_256.json`: GRU(256), 7,000 steps, prefix-hist N=16 and 32 reach 0.995 and 0.98 accuracy with alpha50 0.43 and 0.20.

> **Takeaway.** GRU(128) does not learn the prefix histogram exactly at N=32 in 3,000 steps (clean 0.58 to 0.71) but learns majority exactly at all N; GRU(256), 7,000 steps, reaches 0.995 and 0.98 accuracy at N=16 and 32 with alpha50 of 0.43 and 0.20.
