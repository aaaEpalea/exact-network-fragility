
## Part F. Zeros, crosstalk and composition

Two structural questions about the digital designs: how much of the fragility comes from noise on the (exactly zero) unused weights, and what happens when several programs are combined in one network or when trained networks are merged.

### F1. How much fragility comes from the zero entries?

**Question.** How much fragility comes from the exactly-zero entries?

**Code:** `code/zeros_empirical.py`  
**Results:** `results/zeros_empirical.json`


| item | all | zero | nonzero |
|---|---|---|---|
| digital · majority · 16 | 0.03989 | 0.04527 | 0.08326 |
| digital · majority · 32 | 0.02851 | 0.04161 | 0.06617 |
| digital · majority · 64 | 0.02189 | 0.03236 | 0.03946 |
| digital · dyck · 16 | 0.04455 | 0.06032 | 0.1042 |
| digital · dyck · 32 | 0.0367 | 0.04299 | 0.07223 |
| digital · dyck · 64 | 0.02692 | 0.04189 | 0.07838 |
| composition · k1 | 0.0387 | 0.04667 | 0.07099 |
| composition · k2 | 0.02231 | 0.02603 | 0.06124 |
| composition · k4 | 0.01417 | 0.01799 | 0.04467 |
| composition · k8 | 0.009101 | 0.01122 | 0.02328 |
| tracr · hist · 8 | 0.001408 | 0.001746 | 0.002733 |
| tracr · hist · 16 | 7.42e-04 | 8.37e-04 | 0.001475 |
| tracr · hist · 32 | 1.45e-04 | 1.38e-04 | 3.33e-04 |


sigma50 with noise on all / only zero / only nonzero entries for digital designs, composites, and Tracr hist. Protecting zeros gives about 1.5-2.9x (digital) and 1.9-2.3x (Tracr, roughly constant in N): zeros are not what drives Tracr's decay with length.

> **Takeaway.** Protecting zeros gains about 1.5 to 2.9 times for digital designs and 1.9 to 2.3 times for Tracr, roughly constant in N: zeros are not what drives Tracr's decay with length.

### F2. Parallel composition of k compiled programs (ReLU-snap design)

**Question.** Parallel composition of k compiled programs in one dense network.

**Code:** `code/composition_exp.py`  
**Results:** `results/composition_exp.json`


| item | k | D | params | exact_cert | clean | s50 | certified_radius |
|---|---|---|---|---|---|---|---|
| k=1 | 1 | 33 | 18975 | 198, 198 | 1 | 0.03631 | 5.62e-04 |
| k=2 | 2 | 51 | 44115 | 306, 306 | 1 | 0.01995 | 5.62e-04 |
| k=4 | 4 | 102 | 171870 | 612, 612 | 1 | 0.01531 | 3.16e-04 |
| k=8 | 8 | 204 | 678300 | 1224, 1224 | 1 | 0.009533 |  |


k programs in one dense network: all composites exact (certificate 1224/1224 at k=8), parameters grow roughly as k^2, joint tolerance falls about k^-0.65.

> **Takeaway.** All composites are exact (certificate 1224/1224 at k=8), parameters grow roughly as k squared and joint tolerance falls about k to the minus 0.65.

### F3. Composition: per-head tolerance and composite certificates

**Question.** Composition: per-head tolerance and composite certificates.

**Code:** `code/comp_perhead.py`, `code/comp_cert.py`  
**Results:** `results/comp_perhead.json`, `results/comp_cert.json`

`comp_perhead.json`


| item | per_head_s50 | joint_s50 |
|---|---|---|
| k1 · all | 0.03872 | 0.03872 |
| k1 · nonzero | 0.07564 | 0.07564 |
| k2 · all | 0.02502 | 0.02298 |
| k2 · nonzero | 0.08711 | 0.05653 |
| k4 · all | 0.01729 | 0.01496 |
| k4 · nonzero | 0.08579 | 0.04569 |
| k8 · all | 0.01355 | 0.009503 |
| k8 · nonzero | 0.07739 | 0.02675 |


`comp_cert.json`


| item | D | dense | protected |
|---|---|---|---|
| k1 | 33 | 0.002788 | 0.003995 |
| k2 | 51 | 0.001685 | 0.003924 |
| k4 | 102 | 9.14e-04 | 0.003924 |


With zeros protected, per-head sigma50 is flat in k (7.6e-2 to 8.7e-2) and the certified radius is the same for k=1, 2, 4 (about 3.9e-3 to 4.0e-3); with dense noise both fall. The decay of the joint (all-heads-exact) number is largely the statistical cost of requiring every head to be right.

> **Takeaway.** With zeros protected, per-head sigma50 is flat in k (7.6e-2 to 8.7e-2) and the certified radius is the same for k=1, 2, 4 (about 3.9e-3 to 4.0e-3); with dense noise both fall. The decay of the joint number is largely the statistical cost of requiring every head to be right.

### F4. Composing trained GRUs: merging vs stacking

**Question.** Composing trained GRUs: merging weights vs stacking blocks.

**Code:** `code/gru_merge.py`  
**Results:** `results/gru_merge.json`


| item | majority alone | dyck alone | composite (both heads exact) |
|---|---|---|---|
| value | 1.023 | 1.234 | 0.9512 |


Weight averaging destroys both trained programs (accuracy 0.04 and 0.15, printed in the log). Block-diagonal stacking is exact with no tolerance cost (alpha50 about 1). Block-diagonal composition is easy in any architecture; the open problem is shared-capacity composition.

> **Takeaway.** Averaging weights destroys both programs (accuracy 0.04 and 0.15). Block-diagonal stacking is exact with no tolerance cost (alpha50 about 1). Block-diagonal composition is easy in any architecture; the open problem is shared-capacity composition (see Part I).
