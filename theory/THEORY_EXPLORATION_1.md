> **CORRECTION (same day, see THEORY_EXPLORATION_2.md):** the "cliff failures" reported below for `majority8` and `dyck8` were an artifact of autodiff giving relu'(0) = 0 at ReLU kinks that these constructions place exactly at the clean operating point. With the one-sided (sensitive) derivative the first-order L-inf radius matches the attack radius within 0.86-1.4x for 10 of 12 networks; only `reverse` remains a genuine cliff (52x). Observation 1 and the "6 of 16 non-linear" statement below should be read with this correction; observation 2 (padding non-invariance) is also a kink/degenerate-point effect and has not been re-tested.

# Theory exploration 1: can a first-order (gradient) analysis predict parameter-noise failure? (data-first check, 2026-11-13)
Pre-registration: `PREREG_linear_predictor.md` (written before any value was computed). Development set = contaminated run (16 networks); held-out set = fresh run (not touched).

## Predictor v1 (min over samples of margin / gradient norm), development set
| mode | n | Spearman | log-log slope (b=1 expected) | median abs log10 error | within 2x |
|---|---|---|---|---|---|
| absolute noise | 16 | 0.50 | 0.24 | 0.39 | 50% |
| relative | 16 | 0.61 | 0.34 | 0.37 | 38% |
| non-zero only | 16 | 0.44 | 0.13 | 0.45 | 31% |
Pre-registered held-out criterion (median log10 error <= 0.3 and Spearman >= 0.8) is not reachable from these development numbers; v1 is rejected on the development set (held-out run not needed to reject it, to be run for completeness).
Group level: unembed (rho 0.76) and mlp_in (0.80) are predicted reasonably, attention value/output is not (rho -0.17), embeddings are off by ~10x.

## Predictor v2 (Monte-Carlo noise on the first-order model m_i + g_i.delta; no constants), development set
Linearized prediction / measured sigma50 (absolute): hist 1.5x, parity 2.2-2.5x, length 4.5-6.4x, dyck(unconstrained) 5.6x, majority(constrained) 7.1x, dyck(constrained) 17.6x, majority(unconstrained) 27x, reverse 50-83x. All 16 networks fail BEFORE the linear model says they should (ratio 1.5-83x). v2 is not a usable predictor either, but the ratio is not random: it is smallest for networks whose failure is a plain output-margin crossing and largest for hard-attention / saturated constructions.

## L-infinity: first-order radius vs attack-found failure (development set)
min_i m_i/||g_i||_1 equals the attack first-failure to within 1.5x for 10 of 16 networks (length x4: 1.00x; hist x2: 1.03x; parity x4: 1.03x-1.5x); the other 6 fail below the first-order radius by 2x (dyck constrained), 52x (reverse), 94x (majority), 172x (dyck unconstrained). So for ~60% of networks the worst-case failure is locally linear in the parameters (and the sign-gradient attack finds it exactly), for the rest a non-linear mechanism fails much earlier.

## Observations that are not explained by anything in THEORY_CANDIDATES.md
1. Non-linear "cliff" failures: for saturated-attention constructions the output margin has a flat gradient at the clean point (softmax saturation) yet fails at tiny finite perturbations; the output-margin gradient is the wrong variable. Hypothesis to test: robustness is controlled by the margin of the last analog variable that feeds a hard decision (e.g. the length estimate M in `reverse`, whose error must stay below 0.5), and its sensitivity, not by output margins.
2. Predicted sensitivity is not invariant to zero-padding: the same construction at two widths differs 84x in v1 prediction (dyck) but only 2.3x in measurement, because unused parameters sitting at degenerate points (q = k = 0, uniform attention) have large first-order sensitivity yet little measured effect. Hypothesis: sensitivity must be weighted by the downstream decision margin, not by the raw gradient.
3. The linearization gap (v2 ratio) orders constructions by type; it may be a usable descriptive quantity.

## Honest status
No theorem or predictor survives. What exists: a falsified simple predictor (reported with its numbers), one clean regularity (first-order radius = attack radius when failure is linear), and two testable hypotheses (decision-variable margin; padding non-invariance). Next experiment if pursued: margin-vs-perturbation profiles along the attack direction and per-failure attribution to the first analog variable that crosses its threshold.
