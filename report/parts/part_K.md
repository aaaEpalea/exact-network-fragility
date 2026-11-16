
## Part K. LLM-synthesized transformers: fresh-session run (uncontaminated), first pass

The same prompts, given to fresh chat sessions that had seen nothing about the project. The project owner ran 60 chats; we re-evaluated every submission with the frozen harness. **One of the four conditions (feedback) was compromised by an operator error, and the model identity of five chats is unresolved; both are described in K0.**

**Status: preliminary pointer data; one LLM family in chat UI, up to 3 attempts per cell. Never pooled with the contaminated run (J). Protocol deviations are listed in K0.** Frozen package unchanged (`freeze.py --check`: package unchanged, sha256 `a528b01d...`). Files: `llm_synthesis/fresh_run/` (submissions exactly as received, records, per-attempt results `*.json`, `fresh_summary.json`), `harness/batch_fresh.py`, `harness/aggregate_fresh.py`, `harness/splice_K.py`, `HARNESS_CHANGELOG.md`.

### K0. Deviations and open items (read first)

**Question.** Deviations and open items (read first).

* **Priority 3 (feedback) is compromised.** For all six unconstrained feedback cells the first answer is valid and correct, yet `feedback.py` run with `--condition constrained` returns a validation error on it (with the matching condition it prints 200/200); the project owner reports the chats ended with the model calling the fix impossible after a validation error. The most likely cause is that the wrong `--condition` was passed; not confirmed. In the constrained cells `hist` and `parity` the later revisions changed the architecture and are invalid under the constrained condition, although the first answers were correct. These 12 cells are reported but excluded from any feedback-vs-no-feedback comparison.
* **Model identity.** The project owner reports that the last five priority-4 chats used GPT instead of Claude. Exact attempt ids were not provided and no records/`meta.json` were included; priority-4 results below are therefore NOT attributed to a model. They must not be pooled with the Claude results until the ids are confirmed.
* **Missing material.** No transcripts for the 12 feedback chats and 12 category-4 chats, no per-attempt records, no sampling settings (chat defaults).
* **Harness v2.** `evaluate.py` was patched for speed (attack functions compiled once per model); the attack first-failure value is bit-identical to v1 on a test model. Group-wise tools and the frozen files are otherwise unchanged (`HARNESS_CHANGELOG.md`).

> **Takeaway.** The feedback condition was compromised (the feedback checker was most likely run with the wrong condition flag for the unconstrained cells), five chats used a different model whose identity was not recorded, and transcripts and per-attempt records are missing for the feedback and literal-weight chats. We report these results but keep them out of the feedback comparison and from any model attribution.

### K1. Clean-correctness gate (exhaustive over all 335,922 sequences + hidden test set)

**Question.** Clean-correctness gate.

First answers: **57 of 60 correct**. Priority 1-2 (3b, no feedback): 35 of 36 correct (failure: `hist` constrained a3, 0% accuracy). Priority 3 (feedback cells): 12 of 12 first answers correct. Priority 4 (literal weights): 10 of 12 correct (failures: `parity` constrained a3 and `parity` unconstrained a3, both 59% accuracy / 106,431 wrong sequences).

> **Takeaway.** 57 of 60 first answers were exactly correct on all 335,922 inputs.

### K2. Category 3b, no feedback: robustness per attempt (absolute sigma50)

**Question.** Category 3b (LLM code computes the weights), no feedback: robustness per attempt.

| task | condition | a1 | a2 | a3 | spread (max/min) | contaminated | params (a1/a2/a3) | sample-cert radius (a1/a2/a3) | attack/cert (a1/a2/a3) | identical weights |
|---|---|---|---|---|---|---|---|---|---|---|
| length | constrained | 4.58e-03 | 4.58e-03 | 4.58e-03 | 1.0x | 4.58e-03 | 1881/1881/1881 | 6.32e-04/6.32e-04/6.32e-04 | 1.9/1.9/1.9 | 3 of 3 identical to each other and to the contaminated attempt |
| length | unconstrained | 4.13e-03 | 4.13e-03 | 4.07e-03 | 1.0x | 4.28e-03 | 112/112/112 | 6.36e-04/6.36e-04/6.35e-04 | 1.9/1.9/1.9 | a1 = a2 identical; a3 differs |
| hist | constrained | 3.16e-03 | 3.21e-03 | synthesis_failure | 1.0x | 8.03e-04 | 1881/1881/- | 2.24e-04/8.83e-05/- | 1.4/6.2/- | all differ |
| hist | unconstrained | 1.73e-03 | 1.75e-03 | 1.51e-03 | 1.2x | 7.48e-04 | 511/550/550 | 3.33e-05/4.69e-05/2.94e-05 | 9.4/6.6/7.6 | all differ |
| reverse | constrained | 9.51e-04 | 1.06e-03 | 9.68e-04 | 1.1x | 1.03e-03 | 17543/17543/17543 | 1.20e-05/2.13e-05/2.05e-05 | 9.9/5.5/5.8 | all differ |
| reverse | unconstrained | 1.30e-02 | 9.26e-04 | 5.28e-03 | 14.1x | 6.08e-04 | 1215/11259/10675 | 2.60e-04/6.24e-05/2.67e-04 | 8.4/5.4/3.6 | all differ |
| parity | constrained | 6.50e-04 | 2.48e-02 | 2.68e-03 | 38.2x | 2.44e-03 | 6706/6706/6706 | 1.52e-05/5.46e-05/1.05e-04 | 10.3/33.1/11.0 | all differ |
| parity | unconstrained | 7.18e-03 | 9.20e-03 | 1.27e-02 | 1.8x | 2.11e-03 | 160/1546/1546 | 8.23e-05/3.12e-04/7.78e-05 | 1.2/10.6/24.1 | all differ |
| majority8 | constrained | 1.20e-01 | 2.84e-02 | 6.45e-02 | 4.2x | 2.91e-02 | 25698/25698/25698 | 4.33e-03/1.68e-03/1.69e-03 | 1.2/2.5/2.8 | all differ |
| majority8 | unconstrained | 8.17e-02 | 7.80e-02 | 7.80e-02 | 1.0x | 6.84e-03 | 56/81/81 | 5.13e-03/3.72e-03/3.72e-03 | 1.1/1.1/1.1 | a2 = a3 identical |
| dyck8 | constrained | 5.30e-02 | 1.10e-01 | 2.55e-02 | 4.3x | 2.43e-03 | 25698/25698/25698 | 1.01e-03/1.01e-03/8.03e-04 | 1.0/1.0/4.5 | all differ |
| dyck8 | unconstrained | 4.82e-02 | 2.98e-02 | 4.61e-02 | 1.6x | 5.63e-03 | 160/197/1156 | 1.03e-03/2.20e-04/9.96e-04 | 1.0/4.7/2.6 | all differ |


Full per-attempt metrics (relative, nonzero-only, zero-only, parameter groups, certificate, attack) are in `fresh_summary.json` and the per-attempt result files.

> **Takeaway.** Absolute sigma50 spans 6e-4 to 0.12. Attempts of one cell differ by up to 38 times; several cells are tightly reproducible.

![Figure K1.](figures/K1_sigma50_by_task.png)

**Figure K1.** Absolute sigma50 for every attempt, by task and condition: fresh run (dots), contaminated run (diamonds) and the Tracr compiled baseline (squares, shared tasks only).

### K3. Category 4 (literal weights; model attribution pending, see K0)

**Question.** Category 4 (literal weights).

| attempt | status | params | nonzero | sigma50 abs | rel | nonzero-only | cert | attack |
|---|---|---|---|---|---|---|---|---|
| length__constrained__4__none__a1 | correct | 1881 | 0.012 | 4.71e-03 | 2.28e-02 | 9.47e-03 | 6.31e-04 | 1.18e-03 |
| length__constrained__4__none__a2 | correct | 1881 | 0.012 | 4.58e-03 | 2.09e-02 | 9.47e-03 | 6.32e-04 | 1.17e-03 |
| length__constrained__4__none__a3 | correct | 1881 | 0.011 | 3.56e-03 | 3.78e-02 | 3.44e-02 | 6.33e-04 | 1.21e-03 |
| length__unconstrained__4__none__a1 | correct | 53 | 0.189 | 7.36e-02 | 6.07e-02 | 7.64e-02 | 1.18e-02 | 1.38e-02 |
| length__unconstrained__4__none__a2 | correct | 78 | 0.282 | 3.49e-03 | 2.61e-02 | 3.37e-02 | 6.28e-04 | 1.18e-03 |
| length__unconstrained__4__none__a3 | correct | 53 | 0.189 | 7.36e-02 | 6.07e-02 | 7.64e-02 | 1.18e-02 | 1.38e-02 |
| parity__constrained__4__none__a1 | correct | 6706 | 0.019 | 1.76e-02 | 1.68e-02 | 2.40e-02 | 2.95e-04 | 1.58e-03 |
| parity__constrained__4__none__a2 | correct | 6706 | 0.019 | 1.86e-02 | 1.98e-02 | 2.95e-02 | 2.95e-04 | 1.08e-03 |
| parity__constrained__4__none__a3 | synthesis_failure | - | - | - | - | - | - | - |
| parity__unconstrained__4__none__a1 | correct | 877 | 0.173 | 2.11e-02 | 1.44e-02 | 2.36e-02 | 2.95e-04 | 1.12e-03 |
| parity__unconstrained__4__none__a2 | correct | 970 | 0.143 | 1.57e-02 | 1.69e-02 | 2.83e-02 | 2.89e-04 | 1.06e-03 |
| parity__unconstrained__4__none__a3 | synthesis_failure | - | - | - | - | - | - | - |

> **Takeaway.** Feasible for the two smallest tasks (10 of 12 correct); attribution to a model is pending, see K0.

### K4. Feedback cells (compromised, see K0)

**Question.** Feedback cells.

| attempt file (revision) | status | params | sigma50 abs |
|---|---|---|---|
| dyck8__constrained__3b__clean_check__a1_rev0 | correct | 25698 | 4.17e-02 |
| dyck8__unconstrained__3b__clean_check__a1_rev1 | correct | 25698 | 3.60e-02 |
| hist__constrained__3b__clean_check__a1_rev0 | correct | 1881 | 2.46e-03 |
| hist__constrained__3b__clean_check__a1_rev2 | invalid_submission | - | - |
| hist__unconstrained__3b__clean_check__a1_rev0 | correct | 550 | 2.83e-03 |
| length__constrained__3b__clean_check__a1_rev0 | correct | 1881 | 4.58e-03 |
| length__unconstrained__3b__clean_check__a1_rev0 | correct | 105 | 3.74e-03 |
| majority8__constrained__3b__clean_check__a1_rev0 | correct | 25698 | 5.52e-02 |
| majority8__unconstrained__3b__clean_check__a1_rev0 | correct | 56 | 8.17e-02 |
| parity__constrained__3b__clean_check__a1_rev0 | correct | 6706 | 5.85e-04 |
| parity__constrained__3b__clean_check__a1_rev1 | invalid_submission | - | - |
| parity__unconstrained__3b__clean_check__a1_rev1 | correct | 25698 | 1.24e-02 |
| reverse__constrained__3b__clean_check__a1_rev0 | correct | 17543 | 9.85e-04 |
| reverse__unconstrained__3b__clean_check__a1_rev0 | correct | 11259 | 1.42e-03 |

> **Takeaway.** Reported but compromised: the first answers were all correct, and later revisions in two cells changed the architecture and became invalid.

### K5. Pre-registered expectations vs the fresh run

**Question.** Pre-registered expectations versus the fresh run.

* **E1** (synthesis failures common): **not supported** (57/60 first answers correct; 3 failures).
* **E2** (>= 10x spread between attempts of one cell): **supported in 2 of 12 cells** (reverse unconstrained 14x, parity constrained 38x); the other cells vary by 4.3x or less.
* **E3** (sigma50 nearer the Tracr range than the digital range >= 0.1): **not supported for the counter tasks**: 2 of 35 correct 3b attempts have absolute sigma50 >= 0.1 (dyck8__constrained a2, majority8__constrained a1), and `majority8`/`dyck8` span 0.026-0.12. `length`, `hist`, `reverse`, `parity` stay at 6.5e-4-2.5e-2.
* **E4** (constrained vs unconstrained differ via size, not margin structure): **mixed**; median unconstrained/constrained sigma50 per task: length 0.9; hist 0.5; reverse 5.5; parity 3.4; majority8 1.2; dyck8 0.9. 
* **E5** (category 4 only on the smallest tasks, sparse, integer-valued): feasible on `length` and `parity` (10/12 correct, non-zero fraction 1-28%), values not integer.
* **E6** (certificate orders of magnitude below sigma50; attack within ~10x of it): sigma50/cert ranges 6-455x (median 36x), i.e. as low as 6x; attack/cert is <= 10x in 86% of attempts (max 33x).

> **Takeaway.** E1 not supported; E2 (large spread) supported in 2 of 12 cells; E3 not supported for the counter tasks; E6 partly.

### K6. Contaminated vs fresh, and convergence

**Question.** Contaminated versus fresh, and convergence.

* Fresh medians vs the contaminated attempt: within 1.1x for `length` (both conditions), `reverse` constrained and `parity` constrained; 2-4.4x for `hist` (both) and `parity` unconstrained; 8.7x for `reverse` unconstrained. `length` constrained: **the three fresh attempts and the contaminated attempt have bit-identical weights**. Other cells: only `length` unconstrained (a1 = a2) and `majority8` unconstrained (a2 = a3) repeat; all other attempts have different weights even when sigma50 is similar. So convergence to one design is the exception except for the simplest task, not the rule.
* `majority8` and `dyck8`: fresh attempts are mostly more tolerant than the contaminated ones (fresh medians/contaminated: `majority8` constrained 2.2x, unconstrained 11x; `dyck8` constrained 22x, unconstrained 8x; one fresh `majority8` constrained attempt equals it). A plausible explanation (untested) is a different counting design (signed averaging instead of the BOS-weight trick); the contaminated design was the less robust one here.
* Candidate explanation to test in Stage 5 (not a conclusion): robustness is set by the least-margin step of the chosen construction; the same LLM produces constructions whose sigma50 differs by up to 38x within a cell.

> **Takeaway.** The two runs agree on the simplest tasks (for `length` constrained the three fresh attempts and the contaminated one have bit-identical weights) and differ for the counters. Convergence to one design is the exception except for the simplest task.

### K7. Limitations

**Question.** Limitations of the fresh run.

Chat UI with unknown sampling; one model family (plus unattributed GPT chats in priority 4); at most 3 attempts per cell; contaminated and fresh runs differ in what the model had seen; tasks small; certificate sample-based; attack weak; priority 3 compromised; transcripts and records incomplete.

> **Takeaway.** Chat interface with unknown sampling, one model family, at most 3 attempts per cell, feedback compromised.
