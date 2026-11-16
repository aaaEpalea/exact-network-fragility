
## Part E. Compiler and certificates

A restoring-state network can be accompanied by proofs. We build a small compiler from finite-state specifications, prove exactness for sequences of any length, certify a noise radius with interval bounds, and then attack the certificate to see how tight it is.

### E1. Compiler prototype: exactness certificates and empirical tolerance

**Question.** A compiler from finite-state specifications to a restoring-state network, with exactness and noise certificates.

**Code:** `code/digital_compiler.py`, `code/run_compiler_tests.py`, `code/exhaustive.py`  
**Results:** `results/compiler_tests.json`


| item | exact_cert | random_agree | params | certified_radius | tube | clean | s50 |
|---|---|---|---|---|---|---|---|
| prefix_hist(N=4) | 180, 180 | 1 | 17010 | 3.16e-04 | 0.04 |  |  |
| prefix_hist(N=8) | 324, 324 | 1 | 52866 |  |  |  |  |
| prefix_hist(N=16) | 612, 612 | 1 | 183906 |  |  |  |  |
| majority(N=4) | 54, 54 | 1 | 1719 | 0.001 | 0.04 |  |  |
| majority(N=8) | 102, 102 | 1 | 5423 | 0.001 | 0.04 |  |  |
| majority(N=16) | 198, 198 | 1 | 18975 | 5.62e-04 | 0.04 |  |  |
| majority(N=32) | 390, 390 | 1 | 70655 |  |  |  |  |
| dyck_valid_prefix(N=4) | 36, 36 | 1 | 858 | 0.001 | 0.04 |  |  |
| dyck_valid_prefix(N=8) | 60, 60 | 1 | 2070 | 0.001 | 0.04 |  |  |
| dyck_valid_prefix(N=16) | 108, 108 | 1 | 6030 | 0.001 | 0.04 |  |  |
| dyck_valid_prefix(N=32) | 204, 204 | 1 | 20094 | 5.62e-04 | 0.04 |  |  |
| parity_of_token0 | 12, 12 | 1 | 158 | 0.001 | 0.04 |  |  |
| count_mod3 | 18, 18 | 1 | 288 | 0.001 | 0.04 |  |  |
| emp · prefix_hist · 8 |  |  |  |  |  | 1 | 0.0158 |
| emp · prefix_hist · 16 |  |  |  |  |  | 1 | 0.009608 |
| emp · prefix_hist · 32 |  |  |  |  |  | 1 | 0.008194 |
| emp · prefix_hist · 64 |  |  |  |  |  | 1 | 0.006003 |
| emp · majority · 8 |  |  |  |  |  | 1 | 0.05949 |
| emp · majority · 16 |  |  |  |  |  | 1 | 0.03667 |
| emp · majority · 32 |  |  |  |  |  | 1 | 0.02833 |
| emp · majority · 64 |  |  |  |  |  | 1 | 0.02332 |
| emp · dyck_valid_prefix · 8 |  |  |  |  |  | 1 | 0.07276 |
| emp · dyck_valid_prefix · 16 |  |  |  |  |  | 1 | 0.04467 |
| emp · dyck_valid_prefix · 32 |  |  |  |  |  | 1 | 0.03187 |
| emp · dyck_valid_prefix · 64 |  |  |  |  |  | 1 | 0.02864 |


FSM-bundle specs compiled to the original ReLU-snap digital network. `exact_cert` = verified (state, token) transitions out of total; valid for sequences of any length by induction. `certified_radius` = interval-bound certified worst-case parameter perturbation (any length). Empirical tolerance (rows starting `emp|`) decays about N^-0.45. Prior art: Omlin and Giles (1996) compile automata into recurrent nets with stability proofs, so this is not a new compiler concept.

> **Takeaway.** Every transition is verified exactly (valid for any length by induction) and a certified radius is computed by interval bounds; empirical tolerance decays about N to the minus 0.45. Omlin and Giles (1996) compile automata into recurrent networks with stability proofs, so this is not a new compiler concept; the contribution here is measurement.

### E2. Attacking the noise certificate

**Question.** Attack the certificate: is it tight?

**Code:** `code/cert_attack.py`, `code/cert_attack2.py`, `code/cert_attack3.py`  
**Results:** `results/cert_attack.json`, `results/cert_attack2.json`, `results/cert_attack3.json`

`cert_attack.json`


| item | eps_cert | first_break_multiple |
|---|---|---|
| majority(N=8) | 0.001 | 2 |
| majority(N=16) | 5.62e-04 | 2 |
| dyck_valid_prefix(N=16) | 0.001 | 2 |
| dyck_valid_prefix(N=32) | 5.62e-04 | 2 |
| parity_of_token0 | 0.001 | 2 |
| count_mod3 | 0.001 | 2 |


`cert_attack2.json`


| item | eps | tube | status / 1 | status / 1.5 | status / 2 | status / 3 | status / 5 | status / 8 |
|---|---|---|---|---|---|---|---|---|
| majority(N=8) | 0.003278 | 0.03 | mixed | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) |
| majority(N=16) | 0.002788 | 0.03 | mixed | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) |
| dyck_valid_prefix(N=16) | 0.003191 | 0.03 | mixed | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) |
| dyck_valid_prefix(N=32) | 0.002738 | 0.03 | mixed | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) |
| parity_of_token0 | 0.003854 | 0.03 | mixed | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) |
| count_mod3 | 0.00382 | 0.03 | mixed | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) | BREAKS(all tubes) |


`cert_attack3.json`


| item | 1.0 | 1.1 | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| majority(N=8) | -2.57e-04 | 0.01535 | 0.04134 | 0.08504 | 0.1739 |
| majority(N=16) | -3.51e-04 | 0.0119 | 0.03757 | 0.08096 | 0.1701 |
| dyck_valid_prefix(N=16) | -2.60e-04 | 0.01208 | 0.03761 | 0.08041 | 0.1679 |
| dyck_valid_prefix(N=32) | -2.02e-04 | 0.01003 | 0.03551 | 0.07852 | 0.1659 |
| parity_of_token0 | -0.001126 | 0.01597 | 0.04179 | 0.08488 | 0.1718 |
| count_mod3 | -7.60e-04 | 0.01639 | 0.04219 | 0.08554 | 0.1728 |


Gradient (PGD) attack on parameters and tube states against the certified (eps, tube) pairs. `cert_attack3.json` rows: attack violation at multiples of the certified radius (negative = guarantee holds). All programs hold at 1x and break at 1.1x, so the ReLU-snap certificate is tight. (An earlier attack with a mismatched tube width suggested the opposite; superseded.)

> **Takeaway.** All programs hold at 1 times the certified radius and break at 1.1 times, so the ReLU-snap certificate is tight. An earlier attack with a mismatched tube width suggested the opposite and is superseded.

### E3. Certified radius vs number of states (ReLU-snap design)

**Question.** How does the certified radius scale with the number of states?

**Code:** `code/cert_scaling.py`  
**Results:** `results/cert_scaling.json`


| item | D | eps | tube |
|---|---|---|---|
| majority · 2 | 5 | 0.003718 | 0.03 |
| majority · 4 | 9 | 0.003587 | 0.03 |
| majority · 8 | 17 | 0.003278 | 0.03 |
| majority · 16 | 33 | 0.002788 | 0.03 |
| majority · 32 | 65 | 0.002148 | 0.03 |
| majority · 64 | 129 | 0.001472 | 0.03 |
| dyck · 2 | 4 | 0.003718 | 0.03 |
| dyck · 4 | 6 | 0.003619 | 0.03 |
| dyck · 8 | 10 | 0.00346 | 0.03 |
| dyck · 16 | 18 | 0.003191 | 0.03 |
| dyck · 32 | 34 | 0.002738 | 0.03 |
| dyck · 64 | 66 | 0.00211 | 0.03 |


Refined radius (tube 0.03): majority 3.28e-3 -> 1.47e-3 and dyck 3.46e-3 -> 2.11e-3 for N=8 -> 64; scales roughly D^-0.4 when all entries may be perturbed.

> **Takeaway.** When all entries may be perturbed, the radius falls roughly as D to the minus 0.4 (majority 3.3e-3 to 1.5e-3 and dyck 3.5e-3 to 2.1e-3 from N=8 to 64).

### E4. Zero-protected certificate (ReLU-snap design)

**Question.** What if exactly-zero weights are protected from noise?

**Code:** `code/zeros_cert.py`  
**Results:** `results/zeros_cert.json`


| item | D | dense_eps | dense_tube | protected_eps | protected_tube | 1.0 | 1.1 | 1.25 | 1.5 |
|---|---|---|---|---|---|---|---|---|---|
| majority · 8 | 17 | 0.003278 | 0.03 | 0.004032 | 0.03 |  |  |  |  |
| majority · 16 | 33 | 0.002788 | 0.03 | 0.004032 | 0.03 |  |  |  |  |
| majority · 32 | 65 | 0.002148 | 0.03 | 0.004032 | 0.03 |  |  |  |  |
| majority · 64 | 129 | 0.001472 | 0.03 | 0.004032 | 0.03 |  |  |  |  |
| dyck · 8 | 10 | 0.00346 | 0.03 | 0.003924 | 0.03 |  |  |  |  |
| dyck · 16 | 18 | 0.003191 | 0.03 | 0.003924 | 0.03 |  |  |  |  |
| dyck · 32 | 34 | 0.002738 | 0.03 | 0.003924 | 0.03 |  |  |  |  |
| dyck · 64 | 66 | 0.00211 | 0.03 | 0.003924 | 0.03 |  |  |  |  |
| attack · majority(N=16) |  |  |  |  |  | -2.47e-04 | 0.002862 | 0.007571 | 0.01554 |
| attack · dyck(N=16) |  |  |  |  |  | -1.74e-04 | 0.002957 | 0.007703 | 0.01575 |
| attack · majority(N=32) |  |  |  |  |  | -2.47e-04 | 0.002862 | 0.007571 | 0.01554 |


If exactly-zero weights stay zero, the certified radius is constant in network size (4.03e-3 majority, 3.92e-3 dyck, N=8-64) while the all-entries-noisy radius falls. Attack check: holds at 1x, breaks at 1.1x.

> **Takeaway.** The certified radius is then constant in network size (4.03e-3 majority, 3.92e-3 dyck, N=8 to 64): the size dependence of the dense case is entirely crosstalk through the zero entries. The attack again holds at 1 times and breaks at 1.1 times.

### E5. Certificates for the bilinear and weight-tied designs

**Question.** Certificates for the bilinear and weight-tied designs.

**Code:** `code/cert_bilinear.py`, `code/bilinear_lib.py`  
**Results:** `results/cert_bilinear.json`


| case | D | dense_eps | dense_tube | protected_eps | protected_tube | n | 1.0 | 1.1 | 1.25 | 1.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| bilinear · majority · 8 | 17 | 0.06043 | 0.075 | 0.1 | 0.1 |  |  |  |  |  |
| bilinear · majority · 16 | 33 | 0.05 | 0.05 | 0.1 | 0.1 |  |  |  |  |  |
| bilinear · majority · 32 | 65 | 0.04117 | 0.05 | 0.1 | 0.1 |  |  |  |  |  |
| bilinear · majority · 64 | 129 | 0.03 | 0.03 | 0.1 | 0.1 |  |  |  |  |  |
| bilinear · dyck · 8 | 10 | 0.075 | 0.075 | 0.1 | 0.1 |  |  |  |  |  |
| bilinear · dyck · 16 | 18 | 0.05851 | 0.075 | 0.1 | 0.1 |  |  |  |  |  |
| bilinear · dyck · 32 | 34 | 0.05 | 0.05 | 0.1 | 0.1 |  |  |  |  |  |
| bilinear · dyck · 64 | 66 | 0.0407 | 0.05 | 0.1 | 0.1 |  |  |  |  |  |
| tied · N16 |  | 0.038 | 0.05 | 0.075 | 0.075 | 33 |  |  |  |  |
| tied · N32 |  | 0.03 | 0.03 | 0.075 | 0.075 | 65 |  |  |  |  |
| tied · N64 |  | 0.0296 | 0.03 | 0.075 | 0.075 | 129 |  |  |  |  |
| attack · bilinear majority N=16 · dense |  |  |  |  |  |  | -0.05 | -0.05 | -0.05 | -0.05 |
| attack · bilinear majority N=16 · protected |  |  |  |  |  |  | -0.1 | -0.1 | -0.1 | -0.1 |
| attack · bilinear dyck N=16 · dense |  |  |  |  |  |  | -2.24e-05 | 0.02748 | 0.06872 | 0.09797 |
| attack · bilinear dyck N=16 · protected |  |  |  |  |  |  | 1.42e-07 | 0.02 | 0.05 | 0.1 |
| attack · tied N=16 · dense |  |  |  |  |  |  | -0.05 | -0.05 | -0.05 | 0.07179 |
| attack · tied N=16 · protected |  |  |  |  |  |  | -0.00875 | 0.01664 | 0.0543 | 0.1159 |


Interval-bound certificates (exact for any length; the tied certificate is valid for any number of instances k) plus PGD attacks. Radii are 15-40x larger than the ReLU-snap design. Caveats: the zero-protected value 0.1 may be limited by the tube grid and the bilinear-majority certificate could not be broken by the attack even at 1.5x (conservative there); dyck N=16 and tied N=16 are close to tight.

> **Takeaway.** Radii are 15 to 40 times larger than the ReLU-snap design; the tied certificate is valid for any number of instances. Caveats: the zero-protected value may be limited by the tube grid, and the bilinear majority certificate could not be broken by the attack even at 1.5 times (conservative there). Superseded in value by E6.

### E6. Tightened bilinear/tied certificates and function-level adversarial attack

**Question.** Tighten the certificates and attack with a function-level adversary.

**Code:** `code/cert_bilinear2.py`, `code/cert_bilinear.py`, `code/bilinear_lib.py`  
**Results:** `results/cert_bilinear2.json`


| item | fine_eps | tube | certified_eps | min_margin_found / 1 | min_margin_found / 2 | first_break_multiple | min_margin_found / 3 |
|---|---|---|---|---|---|---|---|
| bilinear · majority · 8 · dense | 0.06827 | 0.06827 |  |  |  |  |  |
| bilinear · majority · 8 · protected | 0.09998 | 0.09998 |  |  |  |  |  |
| bilinear · majority · 16 · dense | 0.05704 | 0.05704 |  |  |  |  |  |
| bilinear · majority · 16 · protected | 0.09998 | 0.09998 |  |  |  |  |  |
| bilinear · majority · 32 · dense | 0.04569 | 0.04569 |  |  |  |  |  |
| bilinear · majority · 32 · protected | 0.09998 | 0.09998 |  |  |  |  |  |
| bilinear · dyck · 8 · dense | 0.07655 | 0.07655 |  |  |  |  |  |
| bilinear · dyck · 8 · protected | 0.09998 | 0.09998 |  |  |  |  |  |
| bilinear · dyck · 16 · dense | 0.06733 | 0.06733 |  |  |  |  |  |
| bilinear · dyck · 16 · protected | 0.09998 | 0.09998 |  |  |  |  |  |
| bilinear · dyck · 32 · dense | 0.05652 | 0.05652 |  |  |  |  |  |
| bilinear · dyck · 32 · protected | 0.09998 | 0.09998 |  |  |  |  |  |
| tied · N16 · dense | 0.0425 | 0.0425 |  |  |  |  |  |
| tied · N16 · protected | 0.07643 | 0.07643 |  |  |  |  |  |
| tied · N32 · dense | 0.03634 | 0.03634 |  |  |  |  |  |
| tied · N32 · protected | 0.07643 | 0.07643 |  |  |  |  |  |
| funcattack · majority · 8 · dense |  |  | 0.06827 | 0.8635 | -11.46 | 2 |  |
| funcattack · majority · 8 · protected |  |  | 0.09998 | 0.9 | 0 | 2 |  |
| funcattack · majority · 16 · dense |  |  | 0.05704 | 0.8859 | -20.88 | 2 |  |
| funcattack · majority · 16 · protected |  |  | 0.09998 | 0.9 | 0 | 2 |  |
| funcattack · dyck · 16 · dense |  |  | 0.06733 | 0.8653 | -21.58 | 2 |  |
| funcattack · dyck · 16 · protected |  |  | 0.09998 | 0.9 | 0.8 | 3 | 0 |


`fine_eps` = largest certified per-parameter radius when the tube width is searched finely (supersedes the coarse values in E5). Bilinear dense: majority 0.068/0.057/0.046 and dyck 0.077/0.067/0.057 for N=8/16/32; zeros-protected: exactly 0.100 for every size (an analytic limit from the saturating rows of the shift matrices, not a grid artifact); tied: dense 0.043/0.036 and protected 0.076 (valid for any k). `funcattack` rows: gradient attack on a single fixed perturbation (same at every time step) over length-4N sequences; `min_margin_found` is the smallest correct-minus-best-other output margin found (about 0.86-0.90 at the certified radius means the network is still correct; 0 or negative means a real wrong output). A real wrong output is found at 2x the certified radius for all dense cases and at 2x-3x for zeros-protected ones, so the certificate is within about 2-3x of a demonstrated counterexample (the true radius lies between the certificate and that point). Attack search is limited, so these are upper bounds on the true radius.

> **Takeaway.** Searching the tube width finely gives bilinear dense radii of 0.068, 0.057, 0.046 (majority) and 0.077, 0.067, 0.057 (dyck) for N=8, 16, 32, and exactly 0.100 when zeros are protected (an analytic limit from the saturating rows of the shift matrices). A real wrong output is found at 2 times the certified radius for every dense case and at 2 to 3 times for zero-protected cases, so the certificates are within about 2 to 3 times of a demonstrated counterexample.

### E7. Function-level attack: bilinear N=32 and weight-tied design with k = 1, 4, 15

**Question.** Extend the function-level attack to N=32 and to the tied design with k=1, 4, 15.

**Code:** `code/func_attack_round2.py`, `code/cert_bilinear2.py`  
**Results:** `results/func_attack_round2.json`


| item | certified_eps | min_margin_found / 1 | min_margin_found / 2 | first_break_multiple | min_margin_found / 3 |
|---|---|---|---|---|---|
| bilinear · majority · 32 · dense | 0.04569 | 0.9086 | -39.03 | 2 |  |
| bilinear · majority · 32 · protected | 0.09998 | 0.9 | 0.8 | 3 | 0 |
| bilinear · dyck · 32 · dense | 0.05652 | 0.887 | -40.46 | 2 |  |
| bilinear · dyck · 32 · protected | 0.09998 | 0.9 | 0 | 2 |  |
| tied · N16 · k1 · dense | 0.0425 | 0.915 | -18.72 | 2 |  |
| tied · N16 · k1 · protected | 0.07643 | 0.9236 | 0 | 2 |  |
| tied · N16 · k4 · dense | 0.0425 | 0.915 | -18.72 | 2 |  |
| tied · N16 · k4 · protected | 0.07643 | 0.9236 | 0 | 2 |  |
| tied · N16 · k15 · dense | 0.0425 | 0.915 | -18.72 | 2 |  |
| tied · N16 · k15 · protected | 0.07643 | 0.9236 | 0 | 2 |  |


Same single-fixed-perturbation attack as E6 on length-4N sequences, extended to N=32 and to the tied design (one shared perturbation of the shared matrices and role tables, k instances at once). `min_margin_found` at 1x the certified radius is 0.89-0.92 in all ten cases (network still correct). Real wrong outputs are found at 2x the certified radius for every dense case (margins -19 to -40) and at 2x (tie, margin 0.000) or 3x for the zeros-protected cases; results for k=1, 4 and 15 are identical, so the tied certificate and its tightness do not depend on k. Upper bounds only: the attack search is limited, ties count as failures. (The first run of the k=15 cases died and was resumed with the same script.)

> **Takeaway.** At 1 times the certified radius the network is still correct (margin 0.89 to 0.92) in all ten cases; wrong outputs appear at 2 times (dense) and 2 to 3 times (protected). Results are identical for k=1, 4 and 15, so the tied certificate and its tightness do not depend on k. The k=15 run died once and was resumed with the same script.
