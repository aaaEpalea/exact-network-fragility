
## Part L. Theory exploration: a first-order predictor of failure levels

After the benchmark data existed we asked whether anything *explains* where failures occur. This part is a record of a sequence of pre-registered predictors, including the ones that failed and an artifact we had to retract. It is the confirmatory part of the report; the earlier parts are mostly exploratory.

### L1. Pre-registration and the first predictors (gradient-based, v1 and v2)

#### Pre-registration: first-order (linear) predictor of parameter-noise failure levels
Written before any predictor value was computed. Date of writing: 2026-11-13.

**Data split.** Development set = contaminated run (J, 16 networks). Held-out set = fresh run (K, correct attempts of 3b/no-feedback and category 4). The fresh run is not looked at until the predictor and its constants are frozen.

**Predictor** (for a network with clean margins m_i > 0 on the evaluation samples i = (sequence, position), margin = logit gap for categorical outputs, atol - |error| for numeric outputs; g_i = gradient of m_i w.r.t. all parameters; w = parameter vector):
* absolute Gaussian noise: sigma_hat = min_i m_i / ||g_i||_2
* relative noise (delta_j = sigma w_j z_j): sigma_hat_rel = min_i m_i / ||g_i * w||_2
* non-zero-only / zero-only noise: restrict g_i to non-zero / zero entries
* parameter group G: restrict g_i to the group
* L-inf box: rho_hat = min_i m_i / ||g_i||_1   (to be compared with the certified radius and the attack-found radius)
The measured quantities are sigma50 (project definition), certified radius and attack first-failure from the harness.

**Fit (development only).** log10 sigma50 = a + b log10 sigma_hat, ordinary least squares over networks, separately per noise mode; also report b = 1 with a free offset only. The constants (a, b) are then frozen.

**Held-out test.** Apply the frozen constants to the fresh run. Report: Spearman rho, median absolute error in log10, fraction within factor 2 and 3, and every network that is off by more than a factor 5. Success criterion decided now: median absolute log10 error <= 0.3 (factor 2) and Spearman >= 0.8 on the absolute-noise mode. If the criterion is not met the predictor is reported as failed.

**Counterexample search (after the test).** Look for pairs of networks with similar sigma_hat but sigma50 differing by more than 5x.


> **CORRECTION (same day, see THEORY_EXPLORATION_2.md):** the "cliff failures" reported below for `majority8` and `dyck8` were an artifact of autodiff giving relu'(0) = 0 at ReLU kinks that these constructions place exactly at the clean operating point. With the one-sided (sensitive) derivative the first-order L-inf radius matches the attack radius within 0.86-1.4x for 10 of 12 networks; only `reverse` remains a genuine cliff (52x). Observation 1 and the "6 of 16 non-linear" statement below should be read with this correction; observation 2 (padding non-invariance) is also a kink/degenerate-point effect and has not been re-tested.

#### Theory exploration 1: can a first-order (gradient) analysis predict parameter-noise failure? (data-first check, 2026-11-13)
Pre-registration: `PREREG_linear_predictor.md` (written before any value was computed). Development set = contaminated run (16 networks); held-out set = fresh run (not touched).

##### Predictor v1 (min over samples of margin / gradient norm), development set
| mode | n | Spearman | log-log slope (b=1 expected) | median abs log10 error | within 2x |
|---|---|---|---|---|---|
| absolute noise | 16 | 0.50 | 0.24 | 0.39 | 50% |
| relative | 16 | 0.61 | 0.34 | 0.37 | 38% |
| non-zero only | 16 | 0.44 | 0.13 | 0.45 | 31% |
Pre-registered held-out criterion (median log10 error <= 0.3 and Spearman >= 0.8) is not reachable from these development numbers; v1 is rejected on the development set (held-out run not needed to reject it, to be run for completeness).
Group level: unembed (rho 0.76) and mlp_in (0.80) are predicted reasonably, attention value/output is not (rho -0.17), embeddings are off by ~10x.

##### Predictor v2 (Monte-Carlo noise on the first-order model m_i + g_i.delta; no constants), development set
Linearized prediction / measured sigma50 (absolute): hist 1.5x, parity 2.2-2.5x, length 4.5-6.4x, dyck(unconstrained) 5.6x, majority(constrained) 7.1x, dyck(constrained) 17.6x, majority(unconstrained) 27x, reverse 50-83x. All 16 networks fail BEFORE the linear model says they should (ratio 1.5-83x). v2 is not a usable predictor either, but the ratio is not random: it is smallest for networks whose failure is a plain output-margin crossing and largest for hard-attention / saturated constructions.

##### L-infinity: first-order radius vs attack-found failure (development set)
min_i m_i/||g_i||_1 equals the attack first-failure to within 1.5x for 10 of 16 networks (length x4: 1.00x; hist x2: 1.03x; parity x4: 1.03x-1.5x); the other 6 fail below the first-order radius by 2x (dyck constrained), 52x (reverse), 94x (majority), 172x (dyck unconstrained). So for ~60% of networks the worst-case failure is locally linear in the parameters (and the sign-gradient attack finds it exactly), for the rest a non-linear mechanism fails much earlier.

##### Observations that are not explained by anything in THEORY_CANDIDATES.md
1. Non-linear "cliff" failures: for saturated-attention constructions the output margin has a flat gradient at the clean point (softmax saturation) yet fails at tiny finite perturbations; the output-margin gradient is the wrong variable. Hypothesis to test: robustness is controlled by the margin of the last analog variable that feeds a hard decision (e.g. the length estimate M in `reverse`, whose error must stay below 0.5), and its sensitivity, not by output margins.
2. Predicted sensitivity is not invariant to zero-padding: the same construction at two widths differs 84x in v1 prediction (dyck) but only 2.3x in measurement, because unused parameters sitting at degenerate points (q = k = 0, uniform attention) have large first-order sensitivity yet little measured effect. Hypothesis: sensitivity must be weighted by the downstream decision margin, not by the raw gradient.
3. The linearization gap (v2 ratio) orders constructions by type; it may be a usable descriptive quantity.

##### Honest status
No theorem or predictor survives. What exists: a falsified simple predictor (reported with its numbers), one clean regularity (first-order radius = attack radius when failure is linear), and two testable hypotheses (decision-variable margin; padding non-invariance). Next experiment if pursued: margin-vs-perturbation profiles along the attack direction and per-failure attribution to the first analog variable that crosses its threshold.


### L2. Pre-registered cliff experiment and the kink artifact

#### Pre-registration: cliff-failure experiment (written before running; development set = contaminated run, 12 networks of category 3b)
Quantities per network: d* = the attack perturbation at 1.3 x the harness first-failure radius (increased x1.5 until it fails); m_i(t) = margin of sample (sequence, position) i at parameters theta + t d*; t_x = smallest t on a grid with min_i m_i(t) <= 0; linear model m_i0 + t g_i.d*.
* **H1 (cliff vs linear).** Networks whose first-order L-inf radius is >= 10x the attack radius (majority8, reverse, dyck8-unconstrained, in the previous exploration) show a flat-then-collapse profile: min_i m_i(0.5 t_x) >= 0.8 min_i m_i(0). Networks with first-order radius within 1.5x of the attack radius (length, hist, parity) show a gradual decline: min_i m_i(0.5 t_x) <= 0.6 min_i m_i(0). H1 is rejected if either group violates its bound for more than one network.
* **H2 (attribution, `reverse`).** For `reverse` the failure at d* is caused by the groups that compute the length variable M (embeddings, attention value/output, MLP): zeroing the q/k component of d* leaves the failure, zeroing the MLP components removes it. Rejected if q/k zeroing removes the failure or MLP zeroing does not.
* **H3 (block amplification).** In cliff networks one block amplifies the perturbation-induced deviation of the residual stream by >= 10x relative to the preceding block, in linear networks no block does (all gains < 10x). Rejected if more than one network contradicts.
Decision rule for the overall decision-variable hypothesis: supported only if H1 and H2 are not rejected.

##### Results of the three pre-registered tests (development set; before the next step)
* H1 REJECTED: along the attack direction the margin declines linearly (m(0.5 t_x)/m0 = 0.47-0.50) for all networks except `reverse` (0.99, flat-then-collapse); `majority8` and `dyck8`, which have first-order radius 94-172x above the attack radius, decline linearly too.
* H2 PARTLY SUPPORTED: for `reverse` removing the q/k component leaves the failure, removing mlp_in or attention value/output components removes it, removing mlp_out or embeddings does not; no single group alone causes the failure (the failure needs mlp_in AND attention value/output together).
* H3 REJECTED: attention and MLP blocks amplify the deviation by 12-97x in every network, linear ones included.
* Unplanned finding (kink test, `kink.py`): with relu'(0) = 1 instead of autodiff's 0, the first-order L-inf radius matches the attack radius within 0.86-1.4x for 10 of 12 networks (including majority8 and dyck8); only `reverse` stays 52x off. The earlier "cliff" label for majority8/dyck8 was an artifact of ReLU kinks placed exactly at the operating point.

#### Pre-registration v3 (written before running): directional-derivative (tangent-cone) predictor, parameter-free
Model: m_i(theta + delta) ~ m_i(theta) + f'_i(theta; delta), where f'_i is the one-sided directional derivative (positively homogeneous in delta), obtained exactly as (m_i(theta + s*delta) - m_i(theta))/s with s = 1e-5, delta = z (absolute), |theta| z (relative), z restricted to non-zero / zero entries; z ~ N(0, I), 150 draws. Predicted accuracy(sigma) = mean over draws and sequences of 1[min over positions of (m_i + sigma f'_i(z)) > 0]; sigma50 by the project interpolation on the project grid. No fitted constants.
* Development set (contaminated run, 3b, 12 networks): report median |log10(pred/measured)| and Spearman for each mode. Expectation: good (<= factor 2) for all networks except `reverse` (softmax-saturation cliff).
* Held-out test (fresh run, correct 3b/no-feedback attempts): success = median |log10 error| <= 0.3 and Spearman >= 0.8 on absolute noise, computed over all attempts and also excluding `reverse`; every attempt off by > factor 5 is listed. Failing the criterion is reported as failure.


![Figure L2.](figures/L2_first_order_vs_attack.png)

**Figure L2.** First-order L-infinity radius divided by the attack-found radius for the 12 development networks, with autodiff gradients (relu'(0)=0) and with the one-sided derivative (relu'(0)=1). A ratio of 1 means the first-order model predicts the attack exactly.

### L3. Predictor v3 (tangent-cone) and its held-out test

#### Theory exploration 2: a parameter-free tangent-cone (directional-derivative) predictor of noise failure levels (2026-11-13)
Pre-registrations: `PREREG_linear_predictor.md`, `PREREG_cliff_experiment.md` (both written before the corresponding numbers were computed). Code: `lin_predictor.py`, `lin_sim.py`, `cliff.py`, `kink.py`, `tangent_sim.py`. Data: `lin_*.json`, `cliff_dev.json`, `kink_dev.json`, `tangent_dev.json`, `tangent_heldout.json`.

##### Sequence of results
1. Gradient-based predictors fail on the development set (contaminated run, 12-16 networks): v1 (min over samples of margin/||g||_2) Spearman 0.50, slope 0.24; v2 (Monte-Carlo on the linearized model) over-predicts robustness by 1.5-83x.
2. Pre-registered cliff experiment (development set): H1 (flat-then-collapse margin profiles for non-linear networks) REJECTED, H3 (block amplification only in cliff networks) REJECTED, H2 (attribution for `reverse`) PARTLY supported (failure needs mlp_in AND attention value/output together; q/k not involved). Block gains of 12-97x occur in every network (the interpolation MLPs have slopes up to ~98).
3. Unplanned finding (kink test): autodiff gives relu'(0) = 0, but these constructions put ReLU breakpoints exactly at the values the clean network produces (lookup knots at attainable inputs, thresholds at decision values). With the sensitive one-sided derivative the first-order L-inf radius matches the attack-found radius within 0.86-1.4x for 10 of 12 networks (previously 94-172x off for `majority8`/`dyck8`). Only `reverse` (softmax-saturated hard attention) stays 52x off.
4. Predictor v3 (pre-registered before running, no fitted constants): m_i(theta + delta) ~ m_i + f'_i(theta; delta) with f' the one-sided directional derivative (computed by finite differences at s = 1e-5), Gaussian delta, 150 draws, project sigma50 definition.

##### Held-out test of v3 (pre-registered criterion: median |log10 error| <= 0.3 and Spearman >= 0.8, absolute noise)
Held-out = fresh-session run, 35 correct 3b/no-feedback attempts (not used for development).
| | n | Spearman | median abs log10 err | within 2x | within 3x |
|---|---|---|---|---|---|
| absolute, all attempts | 35 | 0.817 | 0.116 | 71% | 77% |
| absolute, excluding `reverse` | 29 | 0.898 | 0.083 | 79% | 86% |
| relative, excluding `reverse` | 21 | 0.982 | 0.084 | 86% | 100% |
| non-zero-only, excluding `reverse` | 24 | 0.935 | 0.146 | 75% | 83% |
| zero-only, excluding `reverse` | 29 | 0.860 | 0.049 | 79% | 83% |
**The pre-registered criterion is met** (Spearman only marginally: 0.817). Failures: `reverse` (predicted 1.6-21x too robust; softmax saturation) and one `parity` attempt (7.1x). Per-task median pred/measured: length 1.08, hist 1.07, parity 0.98, majority8 1.73, dyck8 0.66, reverse 15.4. Development set (12 networks): median error 0.09, Spearman 0.46 (0.83 excluding `reverse`).

##### Caveats (important)
* The held-out set shares the architecture family, tasks, and (kind of) author with the development set. It is a held-out RUN, not a held-out construction method. The Stage 5 test across construction methods (Tracr, bilinear/digital, trained) has NOT been done.
* 79% of the variance of measured log10 sigma50 is between tasks. Within tasks (task means removed) the predictor has R^2 0.48 (0.65 excluding `reverse`) and Spearman 0.45: it reproduces the 38x spread of `parity` constrained (measured 6.5e-4 / 2.5e-2 / 2.7e-3, predicted 5.0e-4 / 5.6e-2 / 2.1e-3) but not the ranking of `majority8` / `dyck8` constrained attempts.
* The predictor needs ~150 forward passes per noise mode, more than the sweep it predicts: its value is explanatory (first-order tangent-cone behaviour explains the failure level), not cheaper.
* Mathematically this is first-order analysis of piecewise-linear networks (directional derivatives); the contribution would be the empirical finding that it suffices for these constructions and the identification of its failure mode (softmax-saturated hard attention), not a new theorem.
* Exploration used the same data for several rounds of hypothesis rejection; only v3 was frozen before the held-out evaluation.

##### Next test that would make this a real result
Apply v3 unchanged to Tracr-compiled networks (length, hist, reverse and the other project tasks; measured sigma50 in RESULTS.md A1/A2), to the bilinear/digital designs, and to trained transformers (held-out construction methods).


### L4. Cross-construction test of v3 (Tracr and bilinear networks)

#### Pre-registration: cross-construction test of the tangent-cone predictor v3 (written before any prediction on these networks was computed)
Predictor: exactly `tangent_sim.py` (s = 1e-5, 150 draws, Gaussian, project sigma grid), applied unchanged; margins per (sequence, position): categorical = logit gap, numeric = atol - |error|. Measured sigma50: the project harness noise protocol (static Gaussian, 5 seeds, exact-sequence accuracy).
* **Tracr-compiled networks** (project tasks reverse, sort_unique, hist, length, frac_prevs, sort_freq; abs/rel/nonzero-only/zero-only): success criterion as before on absolute noise: median |log10 error| <= 0.3 AND at least 4 of 6 tasks within a factor of 3; Spearman reported (n = 6, indicative only). A predicted sigma50 of infinity (no failure on the grid) counts as a failure of the predictor.
* **Bilinear restoring-state networks** (independent bilinear networks of 12 programs of the mixed-family pool, absolute noise, grid 10^-3..10^0.3): EXPECTED TO FAIL. Reason stated in advance: the clip unit is flat at the operating point (states sit at 0/1 deep in saturation), so the first-order response is ~0 and the model predicts no failure (inf) while the measured sigma50 is ~0.1; failure there is a nonlinear threshold crossing (like softmax saturation in `reverse`). This part is a stress test of the predictor's domain of validity, not a success test.
* **Trained networks**: not run in this step (no frozen trained models available); planned.


#### Theory exploration 3: cross-construction test of the tangent-cone predictor v3 (2026-11-13)
Pre-registration: `PREREG_cross_construction.md` (before any prediction was computed). Code: `cross_construction.py`; data `cross_tracr.json`, `cross_bilinear.json`. Predictor and settings unchanged from exploration 2.

##### Result: the predictor does NOT transfer across construction methods
###### Tracr-compiled networks (project tasks; absolute and relative noise)
| Tracr task | measured abs sigma50 | predicted | predicted / measured | relative-noise measured | relative-noise predicted |
|---|---|---|---|---|---|
| reverse | 6.10e-04 | 3.15e-02 | 51.54 | 1.65e-03 | 7.12e-01 |
| sort_unique | 1.14e-03 | 1.85e-03 | 1.62 | 3.04e-03 | 5.93e-01 |
| frac_prevs | 1.30e-03 | 1.19e-03 | 0.92 | 3.14e-03 | 3.04e-03 |
| sort_freq | nan | nan | both below grid (fragile at 1e-5) | 4.01e-05 | 9.19e-02 |
| hist | 1.38e-03 | 6.25e-03 | 4.53 | 2.78e-03 | 6.51e-03 |
| length | 2.69e-04 | 4.23e-02 | 157.11 | 9.90e-04 | 2.10e-02 |

Pre-registered criterion (median |log10 error| <= 0.3 and at least 4 of 6 tasks within 3x): **not met**: median |log10 error| = 0.66 over the 5 tasks with defined values, 2 of 5 within 3x (sort_freq: both measurement and prediction are below the grid at sigma = 1e-5, which counts as agreement but not as a success). The predictor over-predicts robustness for `length` (157x), `reverse` (52x), `hist` (4.5x); it is accurate for `frac_prevs` (0.92x) and `sort_unique` (1.6x). Relative noise is mostly wildly over-predicted (`reverse` 430x, `sort_unique` 200x), because relative noise on large Tracr weights is a large absolute perturbation that leaves the linear regime.

###### Bilinear restoring-state networks (12 programs, independent networks, absolute noise)
Predicted / measured sigma50: median 4.9x, range 2.1-7.4x (all over-predict; measured 0.13-0.17, predicted 0.3-1.1). The pre-registered mechanism was slightly wrong: I predicted "no failure" (flat clip), but the predictions are finite; the conclusion (stress test failed) stands.

##### What the combined evidence says
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

##### Status
* The within-family result of exploration 2 (held-out RUN, Spearman 0.82) stands as a statement about LLM-built networks of this family; it does not generalise to Tracr or bilinear networks. It must not be presented as a general predictor.
* A candidate next step (not run): predictor v4 = minimum over saturating units of (distance of the pre-activation to its switching boundary) / (first-order sensitivity of that pre-activation), i.e. apply the first-order idea to the decision variable of the saturated unit; this is close to the interval-certificate logic and would need a pre-registered test across construction methods. Trained networks (the third held-out construction method) were not tested.
* Theory-paper status: no theorem; one negative result (first-order models do not predict failure across constructions), one descriptive diagnostic (saturation index). Benchmark/empirical paper remains the realistic output.


### L5. Predictor v4 (tangent-cone plus saturation margins)

#### Pre-registration: predictor v4 = v3 with saturation margins added to the failure set (written before any v4 value was computed; this is the LAST predictor experiment)
v3 (tangent-cone Monte-Carlo, s = 1e-5, 150 draws, absolute Gaussian noise, project sigma grid) is unchanged except that a sequence now counts as failed if EITHER an output margin OR a saturation margin becomes non-positive:
* softmax attention (family and Tracr): for every (layer, head, query position >= 1) the clean top set S = {keys with score >= max - 0.01}; margin = min_{j in S} s_j - max_{j not in S} s_j (S kept fixed); if S is the whole allowed set the margin is +inf.
* clip units (bilinear): for every (step, state unit) with clean pre-activation z: on-units margin z - 0.75, off-units margin 0.25 - z.
No constants. Absolute noise only.
Success criterion (decided now), evaluated separately on: (a) LLM-built family, fresh run (35 attempts); (b) Tracr (6 tasks); (c) bilinear (12 programs): median |log10(pred/measured)| <= 0.3 and >= 2/3 of cases within 3x. v4 is declared successful only if (b) and (c) both meet it and (a) does not get worse than v3 (median error <= 0.3). Otherwise it is reported as failed and the predictor line of work stops.
Known risk stated in advance: set flips can be harmless, so v4 may under-predict robustness (too pessimistic).


#### Theory exploration 4: predictor v4 (tangent-cone + saturation margins). Last predictor experiment, 2026-11-13
Pre-registration: `PREREG_v4_saturation.md` (before any v4 value). Code: `v4.py`; data `v4_fresh.json`, `v4_tracr.json`, `v4_bilinear.json`.

##### Results (absolute noise; median |log10 error|, fraction within 3x)
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

##### Verdict by the pre-registered rule
* (a) LLM-built family: met and improved (only one attempt more than 5x off: `parity` unconstrained a3, 7.1x; `reverse` is now predicted within 1.5x, was 12-21x).
* (c) bilinear: met (median error 0.23, 12 of 12 within 3x, slightly pessimistic as pre-announced).
* (b) Tracr: **not met under the conservative reading** (5 tasks with defined values: median error 0.38 > 0.3, 60% within 3x < 2/3). If `sort_freq` (both measurement and prediction below the grid) is counted as an agreement, the numbers are 4 of 6 within 3x and median 0.23; I use the conservative reading because exploration 3 did too. `reverse` improved from 52x to 2.4x, but `length` (157x) and `hist` (4.5x) are unchanged: the saturation margins do not bind there, so another mechanism (not identified) makes them fail far earlier than first-order sensitivity says.
* Overall: v4 is NOT declared successful (rule requires (b) and (c)); it is a substantial but partial improvement. The predictor line of work stops here, as pre-registered.

##### What survives
* An explanatory result: for hard-attention / saturating constructions the failure level is well described (within ~2x for 83% of LLM-built attempts, 12/12 bilinear programs, Tracr `reverse`/`sort_unique`/`frac_prevs`) by the first-order response of the output margins together with the first-order response of the saturation (switching) margins of attention and clip units. This is first-order, unit-level margin analysis; it is close in spirit to interval certificates and to local-sensitivity bounds, so novelty is expected to be limited (prior-art check pending).
* Residual unexplained: Tracr `length` and `hist` (mechanism unidentified).
* Not tested: trained networks.


![Figure L5.](figures/L1_predictors.png)

**Figure L5.** Predicted against measured sigma50 for predictor v3 (left) and v4 (right) on three construction methods. v4 brings the LLM-built networks and the bilinear networks close to the diagonal; the compiled `length` and `hist` programs stay far above it (the predictor is too optimistic).

### L6. Theory extraction: consolidated status

#### Theory extraction (Stage 5), consolidated status, 2026-11-14
Inputs: `results/implementation_dataset.csv` (115 networks/designs: LLM-built transformers (contaminated + fresh runs), Tracr-compiled transformers, bilinear restoring-state networks, state-code designs), `theory/THEORY_EXPLORATION_1..4.md`, `theory/PREREG_*.md`, `PRIOR_ART_AUDIT.md`. No theorem was found; this document states what the data explain, what they do not, and what is formally established.

##### 1. Definitions (Layer A)
For a network F_theta and a finite or sampled input domain X with exactness criterion "all outputs correct", the project uses four quantities that must not be mixed:
* rho_cert(theta): largest L-infinity box (entrywise, absolute) around theta on which an interval-bound argument proves exactness (a lower bound on rho; sample-certified for transformers, exhaustive for the restoring designs).
* rho_att(theta): smallest box on which a function-level attack finds a wrong output (an upper bound on rho). rho_cert <= rho <= rho_att, where rho = inf{||delta||_inf : exists x in X, F_(theta+delta)(x) != f(x)}.
* sigma50(theta): the Gaussian noise level (iid entries, fixed through the sequence; absolute, relative, zero-only, non-zero-only or per parameter group) at which exact-sequence accuracy crosses 50%. This is a median-failure level, not an infimum.
* fresh-per-step and activation noise are different mathematical problems (project results I2, I4).
Elementary facts (known, not new): for a margin m_i(theta + delta) = m_i - g_i.delta, the L-inf failure radius is min_i m_i/||g_i||_1, the per-sample Gaussian failure probability is Phi(-m_i/(sigma ||g_i||_2)), and a union bound over N samples gives accuracy >= 1 - N Phi(...). Check on the data (12 development networks): the ratio sigma50/rho_att has median 9.8, the worst-sample ratio ||g||_1/||g||_2 has median 4.1; correlation of the logs is weak (slope 4.1, R^2 0.31), so the 'sqrt(dk) gap' candidate is only qualitatively supported.

##### 2. Empirical regularities (Step 2)
1. Construction METHOD explains little by itself: on the three shared tasks LLM-built transformers lie within ~17x of Tracr in absolute sigma50 (length 17x more tolerant, hist 0.6x, reverse 1.0-1.7x), while attempts of the SAME LLM on the SAME task differ by up to 38x (`parity` constrained) and fresh `majority8`/`dyck8` attempts are 4-45x more tolerant than the contaminated ones. Design choices (counting trick, signed averaging, saturation) matter more than who built the weights.
2. Restoring-state designs (bilinear, one-hot/replication codes) tolerate absolute noise of ~0.1-0.17; non-restoring constructions (Tracr, LLM-built hist/length/reverse/parity) fail at 6e-4-2.5e-2; LLM-built counters (`majority8`, `dyck8`) reach 0.03-0.12.
3. Replication buys activation tolerance and costs parameter tolerance; sparse codes (Willshaw) are exact but less robust than one-hot (I2, matches the patent's distributed-code analysis).
4. Perturbation targets fail differently; slow-weight noise present at write time is absorbed, drift afterwards is not (I4, interpreter only).
5. Parameter sharing creates a shared failure mode: one perturbation breaks all k heads (I1).
6. Worst-case radii shrink with the number of stored programs (~P^-0.6) while Gaussian tolerance per program drops only mildly (I4, I5).
7. In the LLM-built networks the first-order L-inf radius equals the attack radius within 1.4x for 10 of 12 networks once the one-sided derivative at ReLU kinks is used (the constructions place ReLU breakpoints exactly at the operating point).

##### 3. Competing hypotheses and status (Step 3)
| Hypothesis | Evidence | Status |
|---|---|---|
| H1 output-margin / gradient first-order model explains failure levels | gradient-based predictor v1 Spearman 0.50 (dev) ; v2 over-optimistic 1.5-83x | **rejected as a general account** |
| H2 tangent-cone (one-sided first-order) model v3 | held-out run Spearman 0.82, median error 1.3x; fails Tracr (median log10 error 0.66) and bilinear (2-7x too optimistic) | **supported within the LLM-built family only** |
| H3 H2 plus saturation margins (attention top-set gaps, clip exits) = v4 | held-out run Spearman 0.93 (89% within 3x); bilinear 12/12 within 2x; Tracr reverse 52x -> 2.4x; Tracr length 157x and hist 4.5x unchanged | **partly supported; not declared successful by the pre-registered rule** |
| H4 state separation / internal margins predict robustness | slack/row Spearman 0.96 with fresh-activation sigma50 (24 clustered cells); replication leaves the certified tube unchanged but raises Gaussian tolerance | **mixed; clustered evidence, no causal intervention beyond the code study** |
| H5 error correction (restoring nonlinearity) explains tolerance | restoring designs tolerate ~0.1-0.17; certificates hold for any length | **supported for activation noise and absolute parameter noise of restoring designs; not isolated by a matched-readout intervention** |
| H6 parameter sharing reduces/changes failure directions | shared failure mode, budget independent of k (I1) | **supported** |
| H7 redundancy | replication trades axes; interpreter shows no shared-capacity gain | **supported (known in the 1990s literature)** |
| H8 composition | pipelines cost almost nothing under noise (I6) | **weakly supported (one interpreter)** |
| H9 readout structure (analog averaging vs digital restoring) is the cause of fragility | Tracr/LLM-built use BOS-averaging + ReLU lookup; no intervention with matched outputs | **untested causally** |
| H10 the construction method (compiler / analytic / LLM) determines robustness | see regularity 1 | **not supported** |

##### 4. Candidate relationships and their domain (Steps 4 and 6)
* R1 (v4 predictor). sigma50 ~ the Gaussian median-failure level of the first-order response of {output margins, attention top-set gaps, clip-exit margins}, Monte-Carlo over noise draws, no free constants. Domain: noise small enough that no unit outside the listed margins switches; validated on LLM-built transformers (held-out run), bilinear restoring networks, Tracr tasks where saturation binds; fails where another mechanism acts (Tracr `length`, `hist`). Not a bound.
* R2 (certificates). Interval-bound tube invariance gives exactness for any sequence length (restoring designs) or on sampled inputs (transformers). Tightness measured: certified-to-attack gap 1.5-3.4x (tied), 4-10x (interpreter), 1.5-33x (LLM-built, sample-certified).
* Both are established empirically; neither is a theorem, and R2's tools are known (IBP, Omlin-Giles stability analysis).

##### 5. Held-out and counterexample status (Steps 7 and 8)
Held-out: fresh LLM run (done), Tracr (done, failed for length/hist/reverse before v4), bilinear (done), trained networks (NOT done). Counterexamples found: Tracr `length` and `hist` (v4 predicts 157x and 4.5x too optimistic), LLM `parity` unconstrained a3 (7.1x), `majority8` unconstrained (v3 9x). A mechanism for the Tracr cases is unidentified.

##### 6. Stage 5.4 required outputs
| Output | Status |
|---|---|
| 1 consolidated dataset | done (`results/implementation_dataset.csv`, 115 rows; trained networks missing) |
| 2 candidate structural properties | done (`THEORY_CANDIDATES.md`, 16 items) |
| 3 competing hypotheses | done (section 3) |
| 4 mathematical definitions | done (section 1) |
| 5 preliminary derivations or bounds | elementary relations only; no new bound |
| 6 experiments distinguishing explanations | done for first-order vs saturation (pre-registered v3, v4); readout-structure intervention NOT done |
| 7 at least one held-out prediction exercise | done (fresh run; Tracr; bilinear) |
| 8 account of what is explained / unexplained | this section and sections 2-5 |
| 9 final experimental protocol informed by the theory | below |

##### 7. Protocol for the full-scale run informed by the theory
Run the shared harness on: more LLMs and more attempts (spread between attempts is the largest unexplained source of variance), trained transformers of the same family (held-out construction method for v3/v4), the Tracr tasks at several lengths; record for every network the v3/v4 predictions and the saturation index (predicted/measured sigma50); add an intervention that changes the readout while preserving outputs (for H9) and an investigation of Tracr `length`/`hist`. Report pre-registered predictor results per construction method, never pooled.

##### 8. Verdict
There is no mathematical result here that survives the prior-art audit. What survives is empirical: a cross-construction benchmark, the finding that within-method variation rivals between-method variation, and a first-order + saturation-margin explanation with a clearly measured domain of validity.


### L7. Candidate relationships considered

#### Theory candidates (Stage 5 inputs): merged list, status as of the LLM-synthesis package
Hypotheses, not results. "Prior-art risk" is a guess to be settled by the prior-art audit. Sources: **M** = candidates from the empirical rounds (RESULTS.md I), **S5** = Stage 5 document (layers A-C).

| # | Candidate | Source | Evidence so far | Competing explanation to separate | Prior-art risk |
|---|---|---|---|---|---|
| 1 | Three distinct radii must be kept apart: certified (lower bound, box norm), attack-found (upper bound), Gaussian sigma50 (median failure); they differ by ~sqrt(dk) (norm1(k) vs norm2(k)) | M + S5 A (norms, absolute vs relative) | interpreter radii ~50x below sigma50; tied designs within 1.5-3.4x; LLM-built transformers: sigma50/cert ~1e2-1e3 (J) | gap from margin geometry, not norm ratio | high |
| 2 | Tube invariance gives exactness for any length: the clip is flat in the saturated region (J_s ~ 0), errors are erased each step | M + S5 B (recurrent error route) | all digital/bilinear/interpreter certificates | tolerance from margins alone, not contraction | medium |
| 3 | Shared failure mode under parameter tying: one perturbation breaks all k heads at ~2x; budget independent of k | M + S5 B #9 | tied-design attacks (I1) | symmetric margins vs structural sharing | unknown |
| 4 | Write-time absorption: perturbations present when a program is written are absorbed; drift afterwards is not | M + S5 B #6 | interpreter study (I4) | recalibration vs margins computed under actual weights | medium |
| 5 | Replication buys activation tolerance, costs parameter tolerance | M + S5 B #6 | codes study (I2) | per-entry shrinkage (1/r) vs redundancy | high |
| 6 | Capacity of a margin-constrained sparse-target memory ~0.5 x stored transitions, independent of n | M + S5 C (cost) | interpreter scaling (I3) | optimiser limit vs true capacity | high |
| 7 | Certified radius ~ P^-0.6 with stored programs (key interference) | M + S5 B #8 | interpreter certificates (I5) | interference vs fast-memory norm growth | medium |
| 8 | Perturbation-model taxonomy: fixed vs fresh, absolute vs relative, parameter group | S5 A | fresh >= static for slow weights; abs vs rel differ by target | static draws are correlated across steps, fresh are not | medium |
| 9 | Jacobian products / Lyapunov growth predict tolerance across designs and lengths | S5 B | not measured | margins alone suffice | medium |
| 10 | State separation and internal margins predict robustness | S5 A, B #1, #5 | slack/row predicts fresh-activation sigma50 (Spearman 0.96, clustered cells); codes = encoding intervention | direct effect vs correlation with sensitivity | medium |
| 11 | Parameter-group sensitivity scales with the group's fan-in (gain/threshold far less sensitive) | S5 A | interpreter (0.39-0.56 vs ~0.1); LLM-built attention q/k groups often insensitive (J) | fan-in vs saturation of the downstream nonlinearity | low-medium |
| 12 | Readout structure (analog averaging vs digital restoring) drives fragility | S5 B #7 | earlier audit, correlational | needs an intervention changing readout with outputs fixed | medium |
| 13 | Composition of restoring stages does not compound errors (min over stages, not sum) | S5 B #8 | pipelines 0.065 vs 0.068 alone (I6) | analog stages compound | medium |
| 14 | Exactness, robustness and cost are not in a universal trade-off | S5 C | bilinear exact and tolerant; replication trades axes; interpreter no gain | none yet | low |
| 15 | Large hard-attention scores (abs(w) ~ 50-300) coexist with sigma50 ~ 1e-3: robustness is set by the least-margin component, not by weight size | M (new, J) | LLM-built transformers (J) | weight scale vs margin | unknown |

Held-out discipline: the LLM-synthesized networks (J, and the fresh-session run) are reserved as held-out data for the Stage 5 predictive test; they must not be used to develop candidates 1-14.

| 16 | Failure level of piecewise-linear constructed networks is set by the one-sided (tangent-cone) first-order response; constructions place ReLU kinks exactly at the operating point, so gradients miss the sensitivity; saturated hard attention (`reverse`) is the failure mode of the first-order model | M (new, theory exploration 2) | held-out run: Spearman 0.82, median error 0.116 log10 (same family, same tasks); 10/12 development networks have first-order L-inf radius = attack radius within 1.4x | gradient/Hessian-based sharpness measures; nonlinear saturation | unknown (piecewise-linear analysis is standard; the application is untested) |

Correction (2026-11-13): candidates 1 and 9 were tested only through autodiff gradients, which are wrong at ReLU kinks; see `theory/THEORY_EXPLORATION_2.md`.

Update (2026-11-13, exploration 3): candidate 16 holds only within the LLM-built family; across construction methods the tangent-cone predictor fails (Tracr: median log10 error 0.66, 2 of 5 within 3x; bilinear: over-predicts robustness 2-7x). Reframed as a descriptive saturation index (predicted/measured sigma50). See `theory/THEORY_EXPLORATION_3.md`.

Update (exploration 4): adding saturation (attention top-set / clip-exit) margins to the tangent-cone model (v4) gives Spearman 0.93 and median error 0.10 (log10) on the held-out LLM-built run, 12/12 bilinear programs within 2x, but still fails Tracr `length` (157x) and `hist` (4.5x). Partial result; predictor line closed. See `theory/THEORY_EXPLORATION_4.md`.

