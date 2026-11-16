
## Part J. LLM-synthesized transformers: protocol, harness and first (contaminated) run

From here the question changes from *how do we build a robust network* to *how robust are networks built by a language model, compared with other construction methods*. Part J describes the frozen protocol, the shared evaluation harness and a first run by the same Claude instance that wrote the harness (therefore *contaminated*). It is a pointer, not a result.

**Status: preliminary pointer data, one LLM, one attempt per cell, CONTAMINATED context. Not for general claims; reserved as held-out data for the theory stage.**
Code and protocol: `llm_synthesis/` (`PROTOCOL.md` frozen before any attempt, package sha256 `a528b01d96fb9158dbd42af1ca260c1336866e7f191d0f5be1ebc42866fdc8e1`; harness files `tasks.py`, `family.py`, `evaluate.py`, `run_eval.py`, `feedback.py`, `tracr_baseline.py`; prompts `prompts/rendered/*.md` (32), run sheet `prompts/run_sheet.csv` (60 chats for the fresh session)); results `llm_synthesis/results/contam__*.json`, `contaminated_summary.json`, `contaminated_tables.md`, `tracr_baseline_same_harness.json`; records `llm_synthesis/records/`. Prior-art notes: `PRIOR_ART_NOTES.md`; theory inputs: `THEORY_CANDIDATES.md`.

### J0. Prior-art check (3 queries, not an audit)

**Question.** What prior work exists on LLMs synthesizing network weights?

Direct LLM synthesis of network weights without gradient descent exists in early form (blog-level "weight-space programming"; trained generators NeuroGen and Drag-and-Drop LLMs); LLM-written RASP compiled by Tracr exists (TracrBench: 57/101 programs correct for GPT-4-turbo); weight-from-graph compilers exist (torchwright). I found no study comparing parameter-perturbation robustness across compiler-built, analytic, LLM-synthesized and trained networks (weak evidence of absence). Two earlier claims overlap known work and go on the audit list: Tracr numerical instability at long lengths (B1/B2) and noise-aware training / error accumulation with depth in analog transformers (C2, G1).

> **Takeaway.** A three-query check (not an audit). Direct LLM weight synthesis exists in early forms; we found no robustness comparison.

### J1. What was built

**Question.** What was built?

Six tasks with the project's semantics (Tracr `length`, `hist`, `reverse`; digital `parity`, `majority(N=8)`, `dyck_valid_prefix(N=8)`), one transformer family (no LayerNorm, ReLU MLP, softmax attention) with every parameter a named perturbable tensor, and one evaluation pipeline: clean-correctness gate (hidden test set AND all 335,922 sequences of length 1-7), static Gaussian noise (absolute, relative, zero-only, nonzero-only, 6 parameter groups; sigma 1e-5..1 in 16 steps, 5 seeds), a new interval-bound certificate for the family (absolute box on all parameters, evaluated on the 200 test sequences = sample-certified radius, soundness checked on a random model), and an L-inf attack with doubling+bisection. The Tracr baselines run through the same noise procedure reproduce the earlier results (hist abs 1.4e-3, reverse 6.1e-4; A1: 1.4e-3, 5.8e-4).

> **Takeaway.** Six tasks, one transformer family in which every parameter is a named perturbable tensor, and one evaluation pipeline used for every network: a clean-correctness gate (exhaustive and hidden test set), static noise protocols, a new sample-certified interval bound for transformers, and an attack.

### J2. Contaminated run (category 3b and 4; no feedback)

**Question.** The contaminated run: 16 constructions written before the checker was run.

16 submissions were written before the checker was run on any of them (12 core cells = 6 tasks x constrained/unconstrained, category 3b; 4 category-4 cells = literal weight values for `length` and `parity`). **All 16 passed the clean-correctness gate** (exhaustive, 0 wrong of 335,922) on the first try; there were no synthesis failures and no feedback-condition attempts were needed. The author of the submissions also wrote the harness and had seen the project's designs, so this says little about LLM capability in general. Category-4 attempts use the same weights as their 3b counterparts, so they are not independent observations.

Per attempt:


| task | condition | cat | params | nonzero | max abs w | sigma50 abs | rel | nonzero-only | cert radius | attack first break | attack/cert | abs/cert |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dyck8 | constrained | 3b | 25698 | 0.001 | 240 | 2.43e-03 | 7.57e-03 | 4.69e-02 | 6.78e-06 | 1.02e-05 | 1.5 | 359 |
| dyck8 | unconstrained | 3b | 378 | 0.071 | 60 | 5.63e-03 | 3.02e-02 | 2.44e-02 | 6.77e-06 | 1.02e-05 | 1.5 | 832 |
| hist | constrained | 3b | 1881 | 0.023 | 170 | 8.03e-04 | 1.09e-03 | 9.85e-04 | 1.68e-05 | 1.19e-04 | 7.1 | 48 |
| hist | unconstrained | 3b | 550 | 0.078 | 159 | 7.48e-04 | 1.25e-03 | 1.44e-03 | 1.68e-05 | 1.19e-04 | 7.1 | 44 |
| length | constrained | 3b | 1881 | 0.012 | 56 | 4.58e-03 | 2.09e-02 | 9.47e-03 | 6.32e-04 | 1.17e-03 | 1.9 | 7 |
| length | constrained | 4 | 1881 | 0.012 | 56 | 4.58e-03 | 2.09e-02 | 9.47e-03 | 6.32e-04 | 1.17e-03 | 1.9 | 7 |
| length | unconstrained | 3b | 146 | 0.158 | 56 | 4.28e-03 | 4.87e-02 | 4.51e-03 | 6.36e-04 | 1.18e-03 | 1.9 | 7 |
| length | unconstrained | 4 | 146 | 0.158 | 56 | 4.28e-03 | 4.87e-02 | 4.51e-03 | 6.36e-04 | 1.18e-03 | 1.9 | 7 |
| majority8 | constrained | 3b | 25698 | 0.001 | 240 | 2.91e-02 | 5.42e-02 | 6.40e-02 | 3.50e-05 | 1.00e-04 | 2.9 | 832 |
| majority8 | unconstrained | 3b | 196 | 0.112 | 60 | 6.84e-03 | 5.27e-02 | 1.82e-02 | 3.49e-05 | 9.97e-05 | 2.9 | 196 |
| parity | constrained | 3b | 6706 | 0.005 | 170 | 2.44e-03 | 5.03e-03 | 5.35e-03 | 7.86e-06 | 1.29e-04 | 16.4 | 311 |
| parity | constrained | 4 | 6706 | 0.005 | 170 | 2.44e-03 | 5.03e-03 | 5.35e-03 | 7.86e-06 | 1.29e-04 | 16.4 | 311 |
| parity | unconstrained | 3b | 160 | 0.194 | 98 | 2.11e-03 | 3.46e-03 | 3.16e-03 | 7.83e-06 | 1.87e-04 | 23.9 | 270 |
| parity | unconstrained | 4 | 160 | 0.194 | 98 | 2.11e-03 | 3.46e-03 | 3.16e-03 | 7.83e-06 | 1.87e-04 | 23.9 | 270 |
| reverse | constrained | 3b | 17543 | 0.004 | 320 | 1.03e-03 | 1.53e-02 | 4.53e-03 | 9.69e-06 | 1.22e-04 | 12.6 | 106 |
| reverse | unconstrained | 3b | 1967 | 0.033 | 196 | 6.08e-04 | 1.71e-02 | 8.80e-03 | 9.69e-06 | 1.22e-04 | 12.6 | 63 |


Per parameter group (sigma50, category 3b):


| task | condition | embeddings | attn q/k | attn v/o | mlp in | mlp out | unembed | zero-only |
|---|---|---|---|---|---|---|---|---|
| dyck8 | constrained | 4.10e-03 | inf | 7.90e-02 | 1.22e-01 | 4.48e-02 | 3.08e-01 | 2.60e-03 |
| dyck8 | unconstrained | 2.93e-03 | 9.41e-01 | 8.86e-02 | 1.59e-01 | 3.05e-02 | 3.08e-01 | 7.18e-03 |
| hist | constrained | 1.09e-03 | 1.53e-03 | 2.63e-02 | 3.64e-03 | inf | 3.07e-01 | 1.23e-03 |
| hist | unconstrained | 1.06e-03 | 2.10e-03 | 2.94e-02 | 4.23e-03 | 6.95e-01 | 3.07e-01 | 1.08e-03 |
| length | constrained | 7.46e-03 | 5.21e-01 | 1.28e-02 | 9.47e-03 | 7.26e-01 | 2.33e-01 | 5.62e-03 |
| length | unconstrained | 8.44e-03 | inf | 1.21e-02 | 4.51e-03 | 7.26e-01 | 2.33e-01 | 5.38e-03 |
| majority8 | constrained | 3.16e-02 | inf | 1.27e-01 | 5.61e-01 | 7.91e-01 | 4.05e-01 | 3.36e-02 |
| majority8 | unconstrained | 1.31e-02 | 3.66e-01 | 2.81e-02 | 6.45e-01 | 5.71e-01 | 5.63e-01 | 7.84e-03 |
| parity | constrained | 7.07e-03 | 4.93e-02 | 7.46e-02 | 5.57e-03 | 2.47e-01 | 3.00e-01 | 4.14e-03 |
| parity | unconstrained | 8.22e-03 | 7.24e-02 | 2.10e-01 | 2.99e-03 | 9.16e-01 | 8.00e-01 | 3.53e-03 |
| reverse | constrained | 7.92e-03 | 1.58e-02 | 2.85e-03 | 1.24e-03 | 3.24e-01 | 2.44e-01 | 1.14e-03 |
| reverse | unconstrained | 8.62e-03 | 1.63e-02 | 7.00e-03 | 7.56e-04 | 2.85e-01 | 2.44e-01 | 6.32e-04 |


Against the compiler (category 1) on the three shared tasks:


| task | Tracr (compiler, category 1) abs | LLM constrained abs | LLM unconstrained abs | Tracr rel | LLM constrained rel | LLM unconstrained rel | Tracr params | LLM constrained / unconstrained params |
|---|---|---|---|---|---|---|---|---|
| length | 2.69e-04 | 4.58e-03 | 4.28e-03 | 9.90e-04 | 2.09e-02 | 4.87e-02 | 2391 | 1881 / 146 |
| hist | 1.38e-03 | 8.03e-04 | 7.48e-04 | 2.78e-03 | 1.09e-03 | 1.25e-03 | 2391 | 1881 / 550 |
| reverse | 6.10e-04 | 1.03e-03 | 6.08e-04 | 1.65e-03 | 1.53e-02 | 1.71e-02 | 58289 | 17543 / 1967 |

> **Takeaway.** All 16 passed the exhaustive correctness gate on the first try. Because the author also wrote the harness and had seen the project, this says little about LLM capability in general.

### J3. Pre-registered expectations vs this run

**Question.** Pre-registered expectations versus this run.

* **E1** (failures common, worse without feedback): **not supported** here: 0/16 failures (contaminated, blind).
* **E2** (>=10x spread across attempts): **untested** (one attempt per cell). Measurement floor: the same construction at different widths differs up to ~2.3x in absolute sigma50 (extra zero entries inject noise, cf. F1) or in relative sigma50 (draw variation), so differences below ~2-3x are not meaningful.
* **E3** (sigma50 near the Tracr range 1e-4..1e-2, far below digital >= 0.1): **supported**: absolute sigma50 of the 12 core attempts spans 6.1e-04..2.9e-02; the three shared tasks lie within a factor ~17 of Tracr (length 17x more tolerant, hist 0.6x, reverse 1.0-1.7x), the largest value (majority8, constrained) is 2.9e-2.
* **E4** (constrained vs unconstrained differ via size, not margin structure): **partly supported**: certified and attack radii are identical for the two conditions of a task (same construction), while absolute sigma50 differs by up to ~2.3x through the number of zero entries.
* **E5** (category 4 feasible only for the smallest tasks; sparse, integer-valued): **partly**: feasible for `length` and `parity` (146-6706 parameters, 0.5-19% non-zeros), but the values are not integer (thresholds 1/k, coefficients up to 98).
* **E6** (certified radius orders of magnitude below sigma50; attack within ~10x of it): **partly**: sigma50/cert is 1e2-1e3 and above (cert 6.8e-06..6.4e-04); attack/cert (category 3b, range per task): dyck8 1.5-1.5x; hist 7.1-7.1x; length 1.9-1.9x; majority8 2.9-2.9x; parity 16.4-23.9x; reverse 12.6-12.6x; i.e. within 10x for dyck, hist, length, majority, not for parity and reverse.

> **Takeaway.** Failures were not common (E1 not supported), sigma50 fell near the compiled range (E3 supported), certificate and attack relations were partly as expected (E6).

### J4. Observations and competing explanations (descriptive; n = 1 per cell)

**Question.** Observations and competing explanations.

* All constructions use hard attention (maximum |w| 56-320) and BOS-based counting plus piecewise-linear ReLU lookups; `hist` and `reverse` therefore use the same mechanism as the Tracr compilation, which may explain why their tolerance is close to Tracr's. `length` is 17x more tolerant than Tracr: possible reasons are a smaller number of components in the critical path or Tracr's wider one-hot encodings; not tested.
* Noise on attention query/key parameters is often harmless (sigma50 up to > 0.9 or `inf` in several attempts) while the embeddings, the value/output path and zero-entry noise limit the networks; the group ordering resembles A1. Large weights do not imply fragility by themselves: robustness is set by the least-margin step (candidate 15 in `THEORY_CANDIDATES.md`).
* The sample-certified radius is 1e-6..1e-3, far below sigma50; the attack finds failures within 1.5-24x of it.

> **Takeaway.** Constructions share a mechanism (hard attention and BOS-based counting with piecewise-linear lookups), which may explain why `hist` and `reverse` match Tracr. The attention query/key groups are often harmless; embeddings, the value/output path and zero-entry noise limit the networks.

### J5. Limitations

**Question.** Limitations of the contaminated run.

One LLM, one attempt per cell, contaminated context, tasks are small, hidden-test seeds fixed, certificate sample-based and loose, attack weak (30 steps, 1 restart), category 3b/4 boundary relies on transcript review, Tracr certificates/attacks not applicable, no trained same-family baselines yet. The fresh-session run (`llm_synthesis/README_FRESH_SESSION.md`) is still to be done by the project owner and will be added as a separate, never-pooled run.

> **Takeaway.** One LLM, one attempt per cell, small tasks, sample-based certificate, weak attack.
