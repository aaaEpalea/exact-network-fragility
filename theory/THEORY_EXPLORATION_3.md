# Theory exploration 3: cross-construction test of the tangent-cone predictor v3 (2026-11-13)
Pre-registration: `PREREG_cross_construction.md` (before any prediction was computed). Code: `cross_construction.py`; data `cross_tracr.json`, `cross_bilinear.json`. Predictor and settings unchanged from exploration 2.

## Result: the predictor does NOT transfer across construction methods
### Tracr-compiled networks (project tasks; absolute and relative noise)
| Tracr task | measured abs sigma50 | predicted | predicted / measured | relative-noise measured | relative-noise predicted |
|---|---|---|---|---|---|
| reverse | 6.10e-04 | 3.15e-02 | 51.54 | 1.65e-03 | 7.12e-01 |
| sort_unique | 1.14e-03 | 1.85e-03 | 1.62 | 3.04e-03 | 5.93e-01 |
| frac_prevs | 1.30e-03 | 1.19e-03 | 0.92 | 3.14e-03 | 3.04e-03 |
| sort_freq | nan | nan | both below grid (fragile at 1e-5) | 4.01e-05 | 9.19e-02 |
| hist | 1.38e-03 | 6.25e-03 | 4.53 | 2.78e-03 | 6.51e-03 |
| length | 2.69e-04 | 4.23e-02 | 157.11 | 9.90e-04 | 2.10e-02 |

Pre-registered criterion (median |log10 error| <= 0.3 and at least 4 of 6 tasks within 3x): **not met**: median |log10 error| = 0.66 over the 5 tasks with defined values, 2 of 5 within 3x (sort_freq: both measurement and prediction are below the grid at sigma = 1e-5, which counts as agreement but not as a success). The predictor over-predicts robustness for `length` (157x), `reverse` (52x), `hist` (4.5x); it is accurate for `frac_prevs` (0.92x) and `sort_unique` (1.6x). Relative noise is mostly wildly over-predicted (`reverse` 430x, `sort_unique` 200x), because relative noise on large Tracr weights is a large absolute perturbation that leaves the linear regime.

### Bilinear restoring-state networks (12 programs, independent networks, absolute noise)
Predicted / measured sigma50: median 4.9x, range 2.1-7.4x (all over-predict; measured 0.13-0.17, predicted 0.3-1.1). The pre-registered mechanism was slightly wrong: I predicted "no failure" (flat clip), but the predictions are finite; the conclusion (stress test failed) stands.

## What the combined evidence says
| construction (held-out for the predictor) | predicted / measured sigma50 |
|---|---|
| LLM-built `length`, `hist`, `parity`, `frac`-type linear failures (held-out run) | 0.9-1.1x |
| Tracr `frac_prevs`, `sort_unique` | 0.9-1.6x |
| LLM-built `majority8`, `dyck8` | 0.7-1.7x (medians) |
| Tracr `hist` | 4.5x |
| bilinear restoring-state networks | 2-7x |
| LLM-built `reverse` | 12-21x |
| Tracr `reverse`, `length` | 52x, 157x |
The ratio is close to 1 when the failure is a plain linear margin crossing and grows when saturating units (softmax with large scores, clip) absorb small perturbations until a threshold is crossed. So the tangent-cone model is a valid description only in the non-saturated regime; the **ratio predicted/measured is a descriptive "saturation index"** of a network, not something the first-order data predict by itself.

## Status
* The within-family result of exploration 2 (held-out RUN, Spearman 0.82) stands as a statement about LLM-built networks of this family; it does not generalise to Tracr or bilinear networks. It must not be presented as a general predictor.
* A candidate next step (not run): predictor v4 = minimum over saturating units of (distance of the pre-activation to its switching boundary) / (first-order sensitivity of that pre-activation), i.e. apply the first-order idea to the decision variable of the saturated unit; this is close to the interval-certificate logic and would need a pre-registered test across construction methods. Trained networks (the third held-out construction method) were not tested.
* Theory-paper status: no theorem; one negative result (first-order models do not predict failure across constructions), one descriptive diagnostic (saturation index). Benchmark/empirical paper remains the realistic output.
