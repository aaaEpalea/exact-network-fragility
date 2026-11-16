# Theory exploration 4: predictor v4 (tangent-cone + saturation margins). Last predictor experiment, 2026-11-13
Pre-registration: `PREREG_v4_saturation.md` (before any v4 value). Code: `v4.py`; data `v4_fresh.json`, `v4_tracr.json`, `v4_bilinear.json`.

## Results (absolute noise; median |log10 error|, fraction within 3x)
| construction | n | v3 | v4 |
|---|---|---|---|
| LLM-built family, fresh run (held-out run) | 35 | 0.116, 77%, Spearman 0.817 | **0.103, 89%, Spearman 0.932** |
| Tracr-compiled (tasks with defined values) | 5 | 0.66, 40% | 0.38, 60% |
| bilinear restoring-state (12 programs) | 12 | 0.69, 33% | **0.23, 100%** (all predicted/measured in 0.50-0.83) |
| Tracr task | measured | v3 / measured | v4 / measured |
|---|---|---|---|
| reverse | 6.10e-04 | 51.54 | 2.39 |
| sort_unique | 1.14e-03 | 1.62 | 1.21 |
| hist | 1.38e-03 | 4.53 | 4.53 |
| length | 2.69e-04 | 157.11 | 157.11 |
| frac_prevs | 1.30e-03 | 0.92 | 0.92 |
| sort_freq | below grid | both below grid | both below grid |

## Verdict by the pre-registered rule
* (a) LLM-built family: met and improved (only one attempt more than 5x off: `parity` unconstrained a3, 7.1x; `reverse` is now predicted within 1.5x, was 12-21x).
* (c) bilinear: met (median error 0.23, 12 of 12 within 3x, slightly pessimistic as pre-announced).
* (b) Tracr: **not met under the conservative reading** (5 tasks with defined values: median error 0.38 > 0.3, 60% within 3x < 2/3). If `sort_freq` (both measurement and prediction below the grid) is counted as an agreement, the numbers are 4 of 6 within 3x and median 0.23; I use the conservative reading because exploration 3 did too. `reverse` improved from 52x to 2.4x, but `length` (157x) and `hist` (4.5x) are unchanged: the saturation margins do not bind there, so another mechanism (not identified) makes them fail far earlier than first-order sensitivity says.
* Overall: v4 is NOT declared successful (rule requires (b) and (c)); it is a substantial but partial improvement. The predictor line of work stops here, as pre-registered.

## What survives
* An explanatory result: for hard-attention / saturating constructions the failure level is well described (within ~2x for 83% of LLM-built attempts, 12/12 bilinear programs, Tracr `reverse`/`sort_unique`/`frac_prevs`) by the first-order response of the output margins together with the first-order response of the saturation (switching) margins of attention and clip units. This is first-order, unit-level margin analysis; it is close in spirit to interval certificates and to local-sensitivity bounds, so novelty is expected to be limited (prior-art check pending).
* Residual unexplained: Tracr `length` and `hist` (mechanism unidentified).
* Not tested: trained networks.
