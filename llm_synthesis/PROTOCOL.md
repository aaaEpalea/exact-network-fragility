# Protocol: LLM-synthesized transformers vs compiled / analytic / trained networks (frozen before any attempt)

## 1. Purpose
Compare the parameter-perturbation robustness of functionally exact transformers whose weights were produced by different construction methods, using the project's tasks, noise protocols, metrics, attack and certification procedures. This first run uses ONE LLM (the Claude instance in the project session = *contaminated* run; a fresh Claude session given only this package = *fresh* run). It produces pointers for the later multi-model AWS run; it is not a general claim about LLMs.

## 2. Construction categories (recorded separately; never pooled)
| id | category | in this package |
|---|---|---|
| 1 | compiler-generated (Tracr; other compilers e.g. torchwright are future work) | Tracr `length`, `hist`, `reverse` evaluated through the same harness (`harness/tracr_baseline.py`) |
| 2 | analytically constructed (project digital / bilinear designs) | evaluated with the project's existing code; not re-run here |
| 3a | LLM writes a *specification* for a compiler (e.g. RASP for Tracr) | not run (TracrBench-style); future work |
| 3b | LLM writes implementation code that computes analytic weights | **run** |
| 4 | LLM writes the weight values directly as literals (no code computes them) | **run** on `length`, `parity` |
| 5 | conventionally trained networks | project GRU/transformer baselines; same-family trained baselines are future work |
Attestation: for category 4 the transcript must show the values as literals; for 3b no training/fitting/search is allowed. Transcripts are reviewed and any violation is recorded in `rule_violations`; violating attempts are reported but excluded from the primary comparison.

## 3. Synthesis conditions
* **Constrained**: exact architecture given per task (`harness/family.py: CONSTRAINED`).
* **Unconstrained**: architecture chosen by the LLM within `LIMITS` (layers <= 4, heads <= 4, d_model <= 64, d_head <= 32, d_mlp <= 256, <= 60,000 parameters).
* **Feedback**: `none` (single shot, no information) or `clean_check` (up to 3 revisions; after each: validation errors, dev-set accuracy on 200 sequences (seed 100), up to 3 failing dev examples). The hidden test set (seed 0) and all robustness information are never shown. No prompt mentions noise, perturbation or robustness.

## 4. Tasks and model family
Six tasks with the semantics, token conventions (BOS=0, tokens 1..6, outputs read at positions 1..L) and input distributions of the existing project: `length`, `hist`, `reverse` (Tracr programs of `fragility.py`; numeric atol 0.5 for length/hist) and `parity`, `majority8`, `dyck8` (digital specs; token t of the digital spec = id t+1; probabilities [.3,.3,.1,.1,.1,.1]). Model family: no-LayerNorm transformer with ReLU MLP and softmax attention (`harness/family.py`), flat dict of named float32 tensors so that every parameter is perturbable by name and group.

## 5. Evaluation pipeline (identical for every network)
1. **Clean-correctness gate**: validation of shapes/limits; exact correctness on the hidden test set (lengths 3..7 x 40, seed 0 = project protocol) AND on all 335,922 sequences of lengths 1..7. Only fully correct submissions enter robustness comparisons; the rest are reported as synthesis failures (with failure type).
2. **Noise protocols** (static Gaussian, as in RESULTS.md A1/A2): absolute, relative (multiplicative), zero-entries-only, nonzero-entries-only, and per parameter group (embeddings, attn query/key, attn value/out, mlp in, mlp out, unembed); sigma grid 10^-5..10^0 in 16 log steps (superset of the project grid), 5 seeds; metric = exact-sequence accuracy; sigma50 = log-interpolated 50% crossing.
3. **Certificate**: interval-bound propagation with an absolute box +-eps on ALL parameters, applied to the 200 test sequences (sample-certified radius; an upper bound on a domain-wide certificate). NEW for transformers (the project had certificates only for the digital / bilinear designs); it is applicable to every network of the family.
4. **Attack**: function-level L-inf sign-gradient attack on the same box, doubling then 6 bisection steps (first failure found; upper bound on the true radius; 30 steps, 1 restart).
5. **Descriptors** (recorded for the later theory stage): parameter count, nonzero fraction, max |w|, mean |w| of nonzeros, architecture, per-group sigma50.
Tracr baselines go through steps 2 with the project's own model (adapter `tracr_baseline.py`); certificates/attacks for Tracr models are not applicable because their forward pass is outside the family (to be added by an adapter for the AWS run).

## 6. Recorded for every attempt
LLM identity and version; interface; sampling settings; prompt (sha256); independent attempt number; execution feedback; revisions allowed/used; whether the LLM could run code; architecture; parameter count; weight-generation procedure; clean correctness (+ failure type); raw transcript; submission file (`records/record_template.json`).

## 7. Design
Cells = task x condition x category x feedback. Priority 1: attempt 1 of the 12 core cells (3b, no feedback); 2: attempts 2-3; 3: feedback variants; 4: category 4 on `length` and `parity`. The contaminated run covers priority 1, the feedback variant for failed cells, and category 4 for 4 cells (1 attempt each). The fresh run covers as many of `prompts/run_sheet.csv` as the user chooses, in priority order, each attempt in a NEW chat with memory/project context off.

## 8. Pre-registered expectations (written before any attempt; guesses, not results)
* E1 Synthesis failure rate rises with task complexity (`dyck8`, `reverse` hardest), and is higher in the no-feedback condition than with clean_check.
* E2 Correct LLM-built networks will show large attempt-to-attempt spread in sigma50 for the same task and condition (>= 10x between best and worst), because exactness does not constrain attention-score scale or MLP margins.
* E3 Typical sigma50 of correct 3b networks will lie nearer the Tracr range (1e-4..1e-2, absolute noise) than the digital restoring-state range (>= 0.1), because natural constructions use softmax attention and ReLU lookups without restoring non-linearities.
* E4 Constrained and unconstrained results differ mainly through architecture size, not through a systematic difference in margin structure.
* E5 Category 4 (literal weights) is feasible only for the smallest tasks, and its correct networks are sparse and integer-valued.
* E6 The sample-certified radius is orders of magnitude below sigma50 (as for the other designs), and the attack-found radius is within ~10x of it.

## 9. Analysis plan
Report all attempts. Per cell: failure rate with failure type, sigma50 (abs/rel/groups), certified and attack radii, descriptors. Compare with the Tracr baselines on the three shared tasks. No significance tests (n is tiny); the contaminated and fresh runs are reported side by side, never pooled. Competing explanations for any difference are listed in the results; the data of this run is NOT used to develop the Stage 5 theory (it is reserved as a held-out set).

## 10. Known limitations
One LLM; chat sampling not controllable; the contaminated run has seen the project's designs (and RESULTS.md); small tasks; hidden test set seeds are fixed; certificate is sample-based and loose; attack is weak; category 4/3b boundary relies on transcript review; Tracr certificates/attacks not applicable.
