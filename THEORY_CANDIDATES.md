# Theory candidates (Stage 5 inputs): merged list, status as of the LLM-synthesis package
Hypotheses, not results. "Prior-art risk" is a guess to be settled by the prior-art audit. Sources: **M** = candidates from the empirical rounds (RESULTS.md I), **S5** = Stage 5 document (layers A-C).

| # | Candidate | Source | Evidence so far | Competing explanation to separate | Prior-art risk |
|---|---|---|---|---|---|
| 1 | Three distinct radii must be kept apart: certified (lower bound, box norm), attack-found (upper bound), Gaussian sigma50 (median failure); they differ by ~sqrt(dk) (||k||_1 vs ||k||_2) | M + S5 A (norms, absolute vs relative) | interpreter radii ~50x below sigma50; tied designs within 1.5-3.4x; LLM-built transformers: sigma50/cert ~1e2-1e3 (J) | gap from margin geometry, not norm ratio | high |
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
| 15 | Large hard-attention scores (|w| ~ 50-300) coexist with sigma50 ~ 1e-3: robustness is set by the least-margin component, not by weight size | M (new, J) | LLM-built transformers (J) | weight scale vs margin | unknown |
| 16 | Failure level of piecewise-linear constructed networks is set by the one-sided (tangent-cone) first-order response; constructions place ReLU kinks exactly at the operating point, so gradients miss the sensitivity; saturated hard attention (`reverse`) is the failure mode of the first-order model | M (new, theory exploration 2) | held-out run: Spearman 0.82, median error 0.116 log10 (same family, same tasks); 10/12 development networks have first-order L-inf radius = attack radius within 1.4x | gradient/Hessian-based sharpness measures; nonlinear saturation | unknown (piecewise-linear analysis is standard; the application is untested) |

Held-out discipline: the LLM-synthesized networks (J, and the fresh-session run) are reserved as held-out data for the Stage 5 predictive test; they must not be used to develop candidates 1-14.

Correction (2026-11-13): candidates 1 and 9 were tested only through autodiff gradients, which are wrong at ReLU kinks; see `theory/THEORY_EXPLORATION_2.md`.

Update (2026-11-13, exploration 3): candidate 16 holds only within the LLM-built family; across construction methods the tangent-cone predictor fails (Tracr: median log10 error 0.66, 2 of 5 within 3x; bilinear: over-predicts robustness 2-7x). Reframed as a descriptive saturation index (predicted/measured sigma50). See `theory/THEORY_EXPLORATION_3.md`.

Update (exploration 4): adding saturation (attention top-set / clip-exit) margins to the tangent-cone model (v4) gives Spearman 0.93 and median error 0.10 (log10) on the held-out LLM-built run, 12/12 bilinear programs within 2x, but still fails Tracr `length` (157x) and `hist` (4.5x). Partial result; predictor line closed. See `theory/THEORY_EXPLORATION_4.md`.
