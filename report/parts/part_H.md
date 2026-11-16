
## Part H. Wild ideas

Three out-of-the-box ideas were tried at small scale before the systematic study of Part I: replicate state units, superpose codes to compress the state, and store several programs in one associative memory with sparse error-correcting codes. All three are negative at the sizes tested, and the third was redone properly in Part I (where one error of the first attempt is corrected).

### H1. Wild ideas: replication, superposition, noise-trained bilinear

**Question.** Replication, superposition, and noise-trained bilinear weights.

**Code:** `code/wild_round.py`  
**Results:** `results/wild_round.json`


| case | clean | sigma50 | params | exact_decode_fraction | loss | all_configs_exact | length400_acc | sigma50_before | sigma50_after | alpha50_before | alpha50_after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| replication · r1 | 1 | 0.1202 | 6633 |  |  |  |  |  |  |  |  |
| replication · r2 | 1 | 0.1931 | 26334 |  |  |  |  |  |  |  |  |
| replication · r4 | 1 | 0.3149 | 104940 |  |  |  |  |  |  |  |  |
| replication · r8 | 1 | 0.3561 | 418968 |  |  |  |  |  |  |  |  |
| superposition · d24 |  |  |  | 0.1751 |  |  |  |  |  |  |  |
| superposition · d48 |  |  |  | 0.7497 |  |  |  |  |  |  |  |
| superposition · d96 |  |  |  | 0.9905 |  |  |  |  |  |  |  |
| superposition · d192 |  |  |  | 0.9994 |  |  |  |  |  |  |  |
| superposition · d384 |  |  |  | 1 |  |  |  |  |  |  |  |
| superposition · d768 |  |  |  | 1 |  |  |  |  |  |  |  |
| superposition · d1536 |  |  |  | 1 |  |  |  |  |  |  |  |
| noise_trained_bilinear | 0.9167 |  |  |  | 0.001032 | 0.6465 | 0.2344 | 0.1202 | 0.1059 | 0.607 | 0.4457 |


Replication of every state unit with unscaled weights gives about sqrt(r) tolerance gain at r^2 parameters (inefficient). Superposition codes need d of about 96-192 dimensions to decode 51 one-hot states exactly: no compression (negative). Noise-injection training of the bilinear weights broke exactness (65% of one-step transitions correct, length-400 accuracy 0.23) and did not raise tolerance (negative).

> **Takeaway.** Replicating every state unit with unscaled weights gives only about sqrt(r) tolerance gain at r squared parameters. Superposition codes need 96 to 192 dimensions to decode 51 one-hot states exactly (no compression). Noise-injection training of the bilinear weights broke exactness (65 percent of one-step transitions correct; length-400 accuracy 0.23) and did not raise tolerance. All negative.

### H2. Error-correcting sparse state codes (Willshaw transitions) and a shared-memory universal interpreter

**Question.** Error-correcting sparse state codes (Willshaw transitions), and one memory for several programs.

**Code:** `code/willshaw_fsm.py`  
**Results:** `results/willshaw_fsm.json`


| case | states_n | code_len_m | code_weight_w | params_willshaw | params_onehot_bilinear | param_ratio | clean | sigma50_willshaw | alpha50_willshaw | sigma50_onehot | alpha50_onehot |
|---|---|---|---|---|---|---|---|---|---|---|---|
| single · majority · N16 | 33 | 128 | 3 | 98688 | 6633 | 0.06721 | 1 | 0.1113 | 0.783 | 0.1176 | 0.6079 |
| single · dyck · N16 | 18 | 28 | 3 | 4788 | 1998 | 0.4173 | 1 | 0.1041 | 0.252 | 0.196 | 0.6218 |
| single · dyck · N32 | 34 | 64 | 3 | 24768 | 7038 | 0.2842 | 1 | 0.1042 | 0.3342 | 0.1135 | 0.6591 |
| single · dyck · N64 | 66 | 96 | 3 | 55584 | 26334 | 0.4738 | 1 | 0.08414 | 0.3236 | 0.1187 | 0.884 |


(A) Each state is a sparse constant-weight codeword (pairwise overlap <= 1), transitions are binary Willshaw associations recalled by thresholding; every transition was verified exactly. It needs MORE parameters than the one-hot bilinear design (2-15x more: e.g. dyck N=64, n=66 states: 55.6k vs 26.3k), found no exact code for majority at N=32 and 64 within code length 128, and gives similar sigma50 (0.08-0.11 vs 0.11-0.20) with lower relative alpha50. (B) Several DIFFERENT programs (majority, dyck, mod counters, random FSMs with 48 states) stored in ONE associative memory with a program code in the key: no exact shared memory was found up to 256 state units for P = 2, 4, 8 programs, and a memory of that size would exceed the separate one-hot tables (27k, 42k, 83k parameters). NEGATIVE at these sizes. Caveats: random codes, weight 3, 4-5 seeds, strict exactness requirement; theory suggests some compression is possible only for much larger state spaces or with optimized codes.

> **Takeaway.** (A) Sparse constant-weight codes with binary Willshaw associations were verified exact but needed 2 to 15 times more parameters than the one-hot design and gave similar sigma50. The statement that no exact code exists for majority at N=32 and 64 was later shown to be an artifact of the readout used (see Part I, I2). (B) Storing several different programs in one associative memory found no exact shared memory up to 256 state units for 2, 4, 8 programs. Negative at these sizes.
