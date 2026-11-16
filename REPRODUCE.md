# Reproducing the experiments

All scripts were written and run in a single Linux environment (Python 3.12, JAX on CPU). After the repository was cleaned (absolute paths replaced by paths relative to the repository root) not every script was re-run; the LLM-synthesis harness, the theory scripts and the report builder were re-tested from a fresh copy. Long scripts are resumable and write JSON to `results/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Scripts in `code/` are run from inside `code/` (`cd code && python <script>.py`). Scripts in `llm_synthesis/harness/` are run from inside that folder, scripts in `theory/` from inside `theory/`, and `report/build_report.py`, `report/make_figures.py` from inside `report/`.

**External dependency.** `code/lsrl_counter.py` (the linear-state counter of Part D1) needs the external `lsrl` package, which is not included; put it at `external/LSRL` or set `LSRL_PATH`.

## LLM-synthesis harness (Parts J and K)

```bash
cd llm_synthesis/harness
python selftest.py                                   # interval-bound soundness check
python run_eval.py --task parity --condition unconstrained \
  --submission ../submissions/contam__parity__unconstrained__3b__none__a1.py --out /tmp/out.json
python feedback.py --task parity --condition unconstrained --submission <file>   # what the feedback condition shows
```

The frozen package (protocol, prompts, harness, `FREEZE.json`) can be verified with `python llm_synthesis/freeze.py --check` on a clean copy (see `llm_synthesis/HARNESS_CHANGELOG.md` for the v2 patch).

## Theory scripts (Part L)

```bash
cd theory
python tangent_sim.py dev && python tangent_sim.py heldout      # predictor v3
python v4.py fresh && python v4.py tracr && python v4.py bilinear  # predictor v4
python cross_construction.py && python cross_construction.py bilinear
```

## Report

```bash
cd report && python make_figures.py && python build_report.py     # regenerates figures and REPORT.md from RESULTS.md and results/
```

## Code and result files by experiment (from RESULTS.md)

| Section | Code | Results |
|---|---|---|
| A1 | `code/run_sweeps2.py`, `code/fragility.py` | `results/results2.json` |
| A2 | `code/trained_baseline.py`, `code/compare_noise.py`, `code/noln.py` | `results/compare.json`, `results/noln.json` |
| A3 | `code/audit_counting.py` | `results/audit_counting.json` |
| A4 | `code/control.py` | `results/control.json` |
| B1 | `code/scaling.py` | `results/scaling.json` |
| B2 | `code/scaling2.py` | `results/scaling2.json` |
| B3 | `code/scaling3.py` | `results/scaling3.json` |
| B4 | `code/diagnose.py` | `results/diagnose.json` |
| B5 | `code/chunk_count.py`, `code/chunk_big.py`, `code/diagnose_chunk.py` | `results/chunk_count.json`, `results/chunk_big.json`, `results/diagnose_chunk.json` |
| C1 | `code/hardening.py`, `code/run_hardening2.py` | `results/hardening2.json` |
| C2 | `code/run_noise_train.py` | `results/noise_train.json` |
| C3 | `code/balance.py`, `code/balance_scaling.py` | `results/balance_scaling.json`, `results/balance_32_plain.json` |
| C4 | `code/balance2.py` | `results/balance2.json` |
| C5 | `code/ibp_train.py` | `results/ibp_train.json` |
| D1 | `code/digital_counter.py`, `code/lsrl_counter.py` | `results/digital_counter.json`, `results/lsrl_counter.json` |
| D2 | `code/digital_prefix_hist.py` | `results/digital_prefix_hist.json` |
| D3 | `code/second_program.py` | `results/second_program.json` |
| D4 | `code/width_matched.py` | `results/width_matched.json` |
| D5 | `code/digital_rel.py` | `results/digital_rel.json` |
| D6 | `code/digital_tune.py` | `results/digital_tune.json` |
| D7 | `code/clip_activation.py` | `results/clip_activation.json` |
| D8 | `code/ideas_round.py`, `code/bilinear_lib.py` | `results/ideas_round.json` |
| D9 | `code/tied_bilinear.py` | `results/tied_bilinear.json` |
| D10 | `code/matched_gru.py` | `results/matched_gru.json` |
| D11 | `code/trained_gru.py`, `code/gru_long.py` | `results/trained_gru_128.json`, `results/trained_gru_long_256.json` |
| E1 | `code/digital_compiler.py`, `code/run_compiler_tests.py`, `code/exhaustive.py` | `results/compiler_tests.json` |
| E2 | `code/cert_attack.py`, `code/cert_attack2.py`, `code/cert_attack3.py` | `results/cert_attack.json`, `results/cert_attack2.json`, `results/cert_attack3.json` |
| E3 | `code/cert_scaling.py` | `results/cert_scaling.json` |
| E4 | `code/zeros_cert.py` | `results/zeros_cert.json` |
| E5 | `code/cert_bilinear.py`, `code/bilinear_lib.py` | `results/cert_bilinear.json` |
| E6 | `code/cert_bilinear2.py`, `code/cert_bilinear.py`, `code/bilinear_lib.py` | `results/cert_bilinear2.json` |
| E7 | `code/func_attack_round2.py`, `code/cert_bilinear2.py` | `results/func_attack_round2.json` |
| F1 | `code/zeros_empirical.py` | `results/zeros_empirical.json` |
| F2 | `code/composition_exp.py` | `results/composition_exp.json` |
| F3 | `code/comp_perhead.py`, `code/comp_cert.py` | `results/comp_perhead.json`, `results/comp_cert.json` |
| F4 | `code/gru_merge.py` | `results/gru_merge.json` |
| G1 | `code/noise_aware_hist.py` | `results/noise_aware_hist.json` |
| G2 | `code/hybrid_transformer.py`, `code/hybrid2.py`, `code/retrain_baselines.py` | `results/hybrid_transformer.json`, `results/hybrid2.json`, `results/retrain_baselines.json` |
| H1 | `code/wild_round.py` | `results/wild_round.json` |
| H2 | `code/willshaw_fsm.py` | `results/willshaw_fsm.json` |
