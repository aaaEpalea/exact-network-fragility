# Theory extraction (Stage 5), consolidated status, 2026-11-14
Inputs: `results/implementation_dataset.csv` (115 networks/designs: LLM-built transformers (contaminated + fresh runs), Tracr-compiled transformers, bilinear restoring-state networks, state-code designs), `theory/THEORY_EXPLORATION_1..4.md`, `theory/PREREG_*.md`, `PRIOR_ART_AUDIT.md`. No theorem was found; this document states what the data explain, what they do not, and what is formally established.

## 1. Definitions (Layer A)
For a network F_theta and a finite or sampled input domain X with exactness criterion "all outputs correct", the project uses four quantities that must not be mixed:
* rho_cert(theta): largest L-infinity box (entrywise, absolute) around theta on which an interval-bound argument proves exactness (a lower bound on rho; sample-certified for transformers, exhaustive for the restoring designs).
* rho_att(theta): smallest box on which a function-level attack finds a wrong output (an upper bound on rho). rho_cert <= rho <= rho_att, where rho = inf{||delta||_inf : exists x in X, F_(theta+delta)(x) != f(x)}.
* sigma50(theta): the Gaussian noise level (iid entries, fixed through the sequence; absolute, relative, zero-only, non-zero-only or per parameter group) at which exact-sequence accuracy crosses 50%. This is a median-failure level, not an infimum.
* fresh-per-step and activation noise are different mathematical problems (project results I2, I4).
Elementary facts (known, not new): for a margin m_i(theta + delta) = m_i - g_i.delta, the L-inf failure radius is min_i m_i/||g_i||_1, the per-sample Gaussian failure probability is Phi(-m_i/(sigma ||g_i||_2)), and a union bound over N samples gives accuracy >= 1 - N Phi(...). Check on the data (12 development networks): the ratio sigma50/rho_att has median 9.8, the worst-sample ratio ||g||_1/||g||_2 has median 4.1; correlation of the logs is weak (slope 4.1, R^2 0.31), so the 'sqrt(dk) gap' candidate is only qualitatively supported.

## 2. Empirical regularities (Step 2)
1. Construction METHOD explains little by itself: on the three shared tasks LLM-built transformers lie within ~17x of Tracr in absolute sigma50 (length 17x more tolerant, hist 0.6x, reverse 1.0-1.7x), while attempts of the SAME LLM on the SAME task differ by up to 38x (`parity` constrained) and fresh `majority8`/`dyck8` attempts are 4-45x more tolerant than the contaminated ones. Design choices (counting trick, signed averaging, saturation) matter more than who built the weights.
2. Restoring-state designs (bilinear, one-hot/replication codes) tolerate absolute noise of ~0.1-0.17; non-restoring constructions (Tracr, LLM-built hist/length/reverse/parity) fail at 6e-4-2.5e-2; LLM-built counters (`majority8`, `dyck8`) reach 0.03-0.12.
3. Replication buys activation tolerance and costs parameter tolerance; sparse codes (Willshaw) are exact but less robust than one-hot (I2, matches the patent's distributed-code analysis).
4. Perturbation targets fail differently; slow-weight noise present at write time is absorbed, drift afterwards is not (I4, interpreter only).
5. Parameter sharing creates a shared failure mode: one perturbation breaks all k heads (I1).
6. Worst-case radii shrink with the number of stored programs (~P^-0.6) while Gaussian tolerance per program drops only mildly (I4, I5).
7. In the LLM-built networks the first-order L-inf radius equals the attack radius within 1.4x for 10 of 12 networks once the one-sided derivative at ReLU kinks is used (the constructions place ReLU breakpoints exactly at the operating point).

## 3. Competing hypotheses and status (Step 3)
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

## 4. Candidate relationships and their domain (Steps 4 and 6)
* R1 (v4 predictor). sigma50 ~ the Gaussian median-failure level of the first-order response of {output margins, attention top-set gaps, clip-exit margins}, Monte-Carlo over noise draws, no free constants. Domain: noise small enough that no unit outside the listed margins switches; validated on LLM-built transformers (held-out run), bilinear restoring networks, Tracr tasks where saturation binds; fails where another mechanism acts (Tracr `length`, `hist`). Not a bound.
* R2 (certificates). Interval-bound tube invariance gives exactness for any sequence length (restoring designs) or on sampled inputs (transformers). Tightness measured: certified-to-attack gap 1.5-3.4x (tied), 4-10x (interpreter), 1.5-33x (LLM-built, sample-certified).
* Both are established empirically; neither is a theorem, and R2's tools are known (IBP, Omlin-Giles stability analysis).

## 5. Held-out and counterexample status (Steps 7 and 8)
Held-out: fresh LLM run (done), Tracr (done, failed for length/hist/reverse before v4), bilinear (done), trained networks (NOT done). Counterexamples found: Tracr `length` and `hist` (v4 predicts 157x and 4.5x too optimistic), LLM `parity` unconstrained a3 (7.1x), `majority8` unconstrained (v3 9x). A mechanism for the Tracr cases is unidentified.

## 6. Stage 5.4 required outputs
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

## 7. Protocol for the full-scale run informed by the theory
Run the shared harness on: more LLMs and more attempts (spread between attempts is the largest unexplained source of variance), trained transformers of the same family (held-out construction method for v3/v4), the Tracr tasks at several lengths; record for every network the v3/v4 predictions and the saturation index (predicted/measured sigma50); add an intervention that changes the readout while preserving outputs (for H9) and an investigation of Tracr `length`/`hist`. Report pre-registered predictor results per construction method, never pooled.

## 8. Verdict
There is no mathematical result here that survives the prior-art audit. What survives is empirical: a cross-construction benchmark, the finding that within-method variation rivals between-method variation, and a first-order + saturation-margin explanation with a clearly measured domain of validity.
