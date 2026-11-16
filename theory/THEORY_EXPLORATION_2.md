# Theory exploration 2: a parameter-free tangent-cone (directional-derivative) predictor of noise failure levels (2026-11-13)
Pre-registrations: `PREREG_linear_predictor.md`, `PREREG_cliff_experiment.md` (both written before the corresponding numbers were computed). Code: `lin_predictor.py`, `lin_sim.py`, `cliff.py`, `kink.py`, `tangent_sim.py`. Data: `lin_*.json`, `cliff_dev.json`, `kink_dev.json`, `tangent_dev.json`, `tangent_heldout.json`.

## Sequence of results
1. Gradient-based predictors fail on the development set (contaminated run, 12-16 networks): v1 (min over samples of margin/||g||_2) Spearman 0.50, slope 0.24; v2 (Monte-Carlo on the linearized model) over-predicts robustness by 1.5-83x.
2. Pre-registered cliff experiment (development set): H1 (flat-then-collapse margin profiles for non-linear networks) REJECTED, H3 (block amplification only in cliff networks) REJECTED, H2 (attribution for `reverse`) PARTLY supported (failure needs mlp_in AND attention value/output together; q/k not involved). Block gains of 12-97x occur in every network (the interpolation MLPs have slopes up to ~98).
3. Unplanned finding (kink test): autodiff gives relu'(0) = 0, but these constructions put ReLU breakpoints exactly at the values the clean network produces (lookup knots at attainable inputs, thresholds at decision values). With the sensitive one-sided derivative the first-order L-inf radius matches the attack-found radius within 0.86-1.4x for 10 of 12 networks (previously 94-172x off for `majority8`/`dyck8`). Only `reverse` (softmax-saturated hard attention) stays 52x off.
4. Predictor v3 (pre-registered before running, no fitted constants): m_i(theta + delta) ~ m_i + f'_i(theta; delta) with f' the one-sided directional derivative (computed by finite differences at s = 1e-5), Gaussian delta, 150 draws, project sigma50 definition.

## Held-out test of v3 (pre-registered criterion: median |log10 error| <= 0.3 and Spearman >= 0.8, absolute noise)
Held-out = fresh-session run, 35 correct 3b/no-feedback attempts (not used for development).
| | n | Spearman | median abs log10 err | within 2x | within 3x |
|---|---|---|---|---|---|
| absolute, all attempts | 35 | 0.817 | 0.116 | 71% | 77% |
| absolute, excluding `reverse` | 29 | 0.898 | 0.083 | 79% | 86% |
| relative, excluding `reverse` | 21 | 0.982 | 0.084 | 86% | 100% |
| non-zero-only, excluding `reverse` | 24 | 0.935 | 0.146 | 75% | 83% |
| zero-only, excluding `reverse` | 29 | 0.860 | 0.049 | 79% | 83% |
**The pre-registered criterion is met** (Spearman only marginally: 0.817). Failures: `reverse` (predicted 1.6-21x too robust; softmax saturation) and one `parity` attempt (7.1x). Per-task median pred/measured: length 1.08, hist 1.07, parity 0.98, majority8 1.73, dyck8 0.66, reverse 15.4. Development set (12 networks): median error 0.09, Spearman 0.46 (0.83 excluding `reverse`).

## Caveats (important)
* The held-out set shares the architecture family, tasks, and (kind of) author with the development set. It is a held-out RUN, not a held-out construction method. The Stage 5 test across construction methods (Tracr, bilinear/digital, trained) has NOT been done.
* 79% of the variance of measured log10 sigma50 is between tasks. Within tasks (task means removed) the predictor has R^2 0.48 (0.65 excluding `reverse`) and Spearman 0.45: it reproduces the 38x spread of `parity` constrained (measured 6.5e-4 / 2.5e-2 / 2.7e-3, predicted 5.0e-4 / 5.6e-2 / 2.1e-3) but not the ranking of `majority8` / `dyck8` constrained attempts.
* The predictor needs ~150 forward passes per noise mode, more than the sweep it predicts: its value is explanatory (first-order tangent-cone behaviour explains the failure level), not cheaper.
* Mathematically this is first-order analysis of piecewise-linear networks (directional derivatives); the contribution would be the empirical finding that it suffices for these constructions and the identification of its failure mode (softmax-saturated hard attention), not a new theorem.
* Exploration used the same data for several rounds of hypothesis rejection; only v3 was frozen before the held-out evaluation.

## Next test that would make this a real result
Apply v3 unchanged to Tracr-compiled networks (length, hist, reverse and the other project tasks; measured sigma50 in RESULTS.md A1/A2), to the bilinear/digital designs, and to trained transformers (held-out construction methods).
