import os
H = os.path.dirname(os.path.abspath(__file__))
import json, re
R = H + "/../results/"; rows = json.load(open(R + "contaminated_summary.json")); tabs = open(R + "contaminated_tables.md").read(); tr = json.load(open(R + "tracr_baseline_same_harness.json"))
rat = {}
for d in rows:
    if d["category"] == "3b": rat.setdefault(d["task"], []).append(d["attack"] / d["cert"])
ra = "; ".join(f"{t} {min(v):.1f}-{max(v):.1f}x" for t, v in rat.items()); sig = [d["s50"]["abs"] for d in rows if d["category"] == "3b"]; cert = [d["cert"] for d in rows if d["category"] == "3b"]; mx = [d["params"]["max_abs"] for d in rows]
sec = f"""## J. LLM-synthesized transformers: protocol, harness and first (contaminated) run

**Status: preliminary pointer data, one LLM, one attempt per cell, CONTAMINATED context. Not for general claims; reserved as held-out data for the theory stage.**
Code and protocol: `llm_synthesis/` (`PROTOCOL.md` frozen before any attempt, package sha256 `a528b01d96fb9158dbd42af1ca260c1336866e7f191d0f5be1ebc42866fdc8e1`; harness files `tasks.py`, `family.py`, `evaluate.py`, `run_eval.py`, `feedback.py`, `tracr_baseline.py`; prompts `prompts/rendered/*.md` (32), run sheet `prompts/run_sheet.csv` (60 chats for the fresh session)); results `llm_synthesis/results/contam__*.json`, `contaminated_summary.json`, `contaminated_tables.md`, `tracr_baseline_same_harness.json`; records `llm_synthesis/records/`. Prior-art notes: `PRIOR_ART_NOTES.md`; theory inputs: `THEORY_CANDIDATES.md`.

### J0. Prior-art check (3 queries, not an audit)
Direct LLM synthesis of network weights without gradient descent exists in early form (blog-level "weight-space programming"; trained generators NeuroGen and Drag-and-Drop LLMs); LLM-written RASP compiled by Tracr exists (TracrBench: 57/101 programs correct for GPT-4-turbo); weight-from-graph compilers exist (torchwright). I found no study comparing parameter-perturbation robustness across compiler-built, analytic, LLM-synthesized and trained networks (weak evidence of absence). Two earlier claims overlap known work and go on the audit list: Tracr numerical instability at long lengths (B1/B2) and noise-aware training / error accumulation with depth in analog transformers (C2, G1).

### J1. What was built
Six tasks with the project's semantics (Tracr `length`, `hist`, `reverse`; digital `parity`, `majority(N=8)`, `dyck_valid_prefix(N=8)`), one transformer family (no LayerNorm, ReLU MLP, softmax attention) with every parameter a named perturbable tensor, and one evaluation pipeline: clean-correctness gate (hidden test set AND all 335,922 sequences of length 1-7), static Gaussian noise (absolute, relative, zero-only, nonzero-only, 6 parameter groups; sigma 1e-5..1 in 16 steps, 5 seeds), a new interval-bound certificate for the family (absolute box on all parameters, evaluated on the 200 test sequences = sample-certified radius, soundness checked on a random model), and an L-inf attack with doubling+bisection. The Tracr baselines run through the same noise procedure reproduce the earlier results (hist abs 1.4e-3, reverse 6.1e-4; A1: 1.4e-3, 5.8e-4).

### J2. Contaminated run (category 3b and 4; no feedback)
16 submissions were written before the checker was run on any of them (12 core cells = 6 tasks x constrained/unconstrained, category 3b; 4 category-4 cells = literal weight values for `length` and `parity`). **All 16 passed the clean-correctness gate** (exhaustive, 0 wrong of 335,922) on the first try; there were no synthesis failures and no feedback-condition attempts were needed. The author of the submissions also wrote the harness and had seen the project's designs, so this says little about LLM capability in general. Category-4 attempts use the same weights as their 3b counterparts, so they are not independent observations.

Per attempt:

{tabs.split(chr(10)*2)[0]}

Per parameter group (sigma50, category 3b):

{tabs.split(chr(10)*2)[1]}

Against the compiler (category 1) on the three shared tasks:

{tabs.split(chr(10)*2)[2]}

### J3. Pre-registered expectations vs this run
* **E1** (failures common, worse without feedback): **not supported** here: 0/16 failures (contaminated, blind).
* **E2** (>=10x spread across attempts): **untested** (one attempt per cell). Measurement floor: the same construction at different widths differs up to ~2.3x in absolute sigma50 (extra zero entries inject noise, cf. F1) or in relative sigma50 (draw variation), so differences below ~2-3x are not meaningful.
* **E3** (sigma50 near the Tracr range 1e-4..1e-2, far below digital >= 0.1): **supported**: absolute sigma50 of the 12 core attempts spans {min(sig):.1e}..{max(sig):.1e}; the three shared tasks lie within a factor ~17 of Tracr (length 17x more tolerant, hist 0.6x, reverse 1.0-1.7x), the largest value (majority8, constrained) is 2.9e-2.
* **E4** (constrained vs unconstrained differ via size, not margin structure): **partly supported**: certified and attack radii are identical for the two conditions of a task (same construction), while absolute sigma50 differs by up to ~2.3x through the number of zero entries.
* **E5** (category 4 feasible only for the smallest tasks; sparse, integer-valued): **partly**: feasible for `length` and `parity` (146-6706 parameters, 0.5-19% non-zeros), but the values are not integer (thresholds 1/k, coefficients up to 98).
* **E6** (certified radius orders of magnitude below sigma50; attack within ~10x of it): **partly**: sigma50/cert is 1e2-1e3 and above (cert {min(cert):.1e}..{max(cert):.1e}); attack/cert (category 3b, range per task): {ra}; i.e. within 10x for dyck, hist, length, majority, not for parity and reverse.

### J4. Observations and competing explanations (descriptive; n = 1 per cell)
* All constructions use hard attention (maximum |w| {min(mx):.0f}-{max(mx):.0f}) and BOS-based counting plus piecewise-linear ReLU lookups; `hist` and `reverse` therefore use the same mechanism as the Tracr compilation, which may explain why their tolerance is close to Tracr's. `length` is 17x more tolerant than Tracr: possible reasons are a smaller number of components in the critical path or Tracr's wider one-hot encodings; not tested.
* Noise on attention query/key parameters is often harmless (sigma50 up to > 0.9 or `inf` in several attempts) while the embeddings, the value/output path and zero-entry noise limit the networks; the group ordering resembles A1. Large weights do not imply fragility by themselves: robustness is set by the least-margin step (candidate 15 in `THEORY_CANDIDATES.md`).
* The sample-certified radius is 1e-6..1e-3, far below sigma50; the attack finds failures within 1.5-24x of it.

### J5. Limitations
One LLM, one attempt per cell, contaminated context, tasks are small, hidden-test seeds fixed, certificate sample-based and loose, attack weak (30 steps, 1 restart), category 3b/4 boundary relies on transcript review, Tracr certificates/attacks not applicable, no trained same-family baselines yet. The fresh-session run (`llm_synthesis/README_FRESH_SESSION.md`) is still to be done by the project owner and will be added as a separate, never-pooled run.

"""
s = open(H + "/../../RESULTS.md").read()
if "## J. LLM-synthesized" in s: s = s[:s.index("## J. LLM-synthesized")] + sec + s[s.index("## Other result files"):]
else: s = s.replace("## Other result files", sec + "## Other result files", 1)
row = "| Perturbation targets (slow / fast / joint / fresh / activation) are different failure mechanisms | supported (I4): joint < either alone; slow-weight noise present at write time is absorbed, drift after writing is not |"
add = "\n| Correct LLM-synthesized transformers (blind, contaminated, 1 attempt/cell, 6 tasks) are as fragile as Tracr-compiled ones (abs sigma50 6e-4..3e-2) | preliminary (J); fresh-session run pending |"
if row in s and add.strip() not in s: s = s.replace(row, row + add)
open(H + "/../../RESULTS.md", "w").write(s); print(len(s))
