## 16. Synthesis: what we know, what we do not, and what is open

### What the evidence supports

1. **Compiled transformers are fragile, and the fragility is a property of the realization.** Exactly-correct compiled networks tolerate noise 15 to 170 times smaller than trained ones in the relative metric, and the cause is concentrated in precise analog readouts from attention averages (A, B). This was not repaired by exact rescaling, attention-scale changes, balancing or noise-aware fine-tuning beyond a few-fold factor, and the decay with sequence length survived all of them (C).
2. **Restoring-state designs trade parameters for tolerance and for guarantees.** One-hot, snapped, bilinear designs reach sigma50 of about 0.1, are exact for any length, carry certificates that are within about 2 to 3 times of a real counterexample, and cost parameters that grow with the state space (D, E, I). They are a re-derivation of the 1990s second-order encoding literature.
3. **Shared failure modes are the price of sharing.** Weight-tied heads fail together at the same budget, independent of how many heads are tied (I1); a shared fast-weight memory fails together too, and its worst-case radius shrinks with load (I5).
4. **A shared interpreter gives shared execution, not shared capacity.** The memory, the compute and the worst-case robustness of the fast-weight interpreter all scale at least linearly with the number of stored programs (I3 to I5), consistent with the known capacity limit of associative memories.
5. **Construction method explains less than design choice.** LLM-written transformers can be exactly correct at the first attempt, their robustness is comparable to compiled ones on shared tasks, and attempts by the same LLM at the same task differ by up to 38 times (J, K). The within-method spread is as large as the between-method gap.
6. **First-order sensitivity of margins, including saturation margins, explains much of the failure level in the LLM-built family** (rank correlation 0.93 on a held-out run, bilinear networks within a factor of 2) but not for two compiled programs (L).

### What we do not know

* Whether trained transformers obey the same first-order picture (not tested).
* The mechanism by which the compiled `length` and `hist` programs fail 157 and 4.5 times earlier than first-order sensitivity predicts.
* Whether the LLM-synthesis results hold for other models, more attempts or harder tasks, and what the feedback condition would show when run correctly.
* Whether readout structure, rather than something correlated with it, causes the fragility of compiled programs (we have no intervention that changes the readout and keeps the outputs).
* Whether any of the quantities here lead to a theorem: none did.

### Open questions

* Can a construction principle be derived that predicts, before measurement, which of several equivalent realizations of a program is robust, and builds a more robust one without losing exactness?
* How does the picture change for trained networks, larger state spaces, and real hardware noise models?
* Is there a shared-capacity composition of programs that beats block-diagonal stacking in robustness?

## 17. Limitations

* **Scale.** All tasks are small (sequence lengths up to 7 to 128, state spaces up to about 100, parameter counts from hundreds to a few hundred thousand). Conclusions about larger models are extrapolation.
* **Statistics.** Most cells use one constructed or trained model and 3 to 5 noise seeds. Differences below about a factor of 2 are not meaningful; sigma50 depends on a coarse grid (16 points over five decades).
* **Exploratory versus confirmatory.** Only the LLM-synthesis protocol and the predictor experiments (Parts J to L) were pre-registered. Parts A to I are exploratory, with hypotheses formed after earlier results; several rounds of the theory exploration reused the same development data (only the final predictors were frozen before their held-out tests).
* **LLM experiments.** One LLM family in a chat interface with unknown sampling; one to three attempts per cell; a contaminated first run; a compromised feedback condition; unresolved model identity for five chats; missing transcripts for 24 chats.
* **Certificates and attacks.** Transformer certificates are sample-based and loose; attacks are weak (few steps, one restart) so attack radii are upper bounds on the true radius, and the interpreter's gap of 4 to 10 times may overstate looseness.
* **Prior art.** The audit used about sixteen search queries, read two primary sources, and could not use a citation-graph search; "no hit" means "not found in a limited search".
* **No theory.** No theorem emerged; the predictor is empirical, validated in one family, and fails elsewhere.

## 18. Reproducibility and repository guide

**Repository.** https://github.com/aaaEpalea/exact-network-fragility (archived release, Zenodo Concept DOI https://doi.org/10.5281/zenodo.24275680; latest version v1.0.1: https://doi.org/10.5281/zenodo.24291013; this report's source is in `report/`; installation and a command for every experiment are in `REPRODUCE.md`).

**Environment.** Python 3.12, JAX (CPU), NumPy, SciPy, Matplotlib, optax; Tracr and dm-haiku installed from the Tracr repository for the compiled baselines. The longest experiments run in minutes to about an hour on one CPU core; the Part I scaling study took about 1.5 hours in total. Every long script is resumable.

**Layout.**

| Path | Content |
|---|---|
| `RESULTS.md` | the consolidated lab notebook this report is built from |
| `code/` | all experiment code (compiler, designs, certificates, attacks, interpreter, harness drivers) |
| `results/` | JSON results and logs for every experiment; `implementation_dataset.csv` (115 networks) |
| `figures/` | all figures |
| `llm_synthesis/` | frozen protocol (`PROTOCOL.md`), harness (`harness/`), prompts, contaminated run, fresh run (`fresh_run/`), changelog |
| `theory/` | pre-registrations, predictor code, explorations 1 to 4, cross-construction data |
| `THEORY_CANDIDATES.md`, `THEORY_EXTRACTION.md`, `PRIOR_ART_AUDIT.md`, `PRIOR_ART_NOTES.md` | theory and literature documents |

**Frozen artifacts.** The LLM-synthesis package (protocol, prompts, harness) has SHA-256 `a528b01d96fb9158dbd42af1ca260c1336866e7f191d0f5be1ebc42866fdc8e1` (version v1); a speed-only patch of the attack (v2, numerically identical) was used for the fresh run (`llm_synthesis/HARNESS_CHANGELOG.md`).

**Data.** The consolidated dataset has one row per network with architecture, size, weight statistics, measured sigma50 under four protocols, certificate and attack radii where available, and predictor outputs. Per-attempt LLM submissions are stored exactly as received.

**Licences.** Code: MIT. Text, figures and data: CC BY 4.0.

## Appendix A. Correction and deviation log

| Where | What was claimed or done | What we found | Status |
|---|---|---|---|
| E2 | An early attack with a mismatched tube width suggested the ReLU-snap certificate was loose | With the matching tube, all programs hold at 1 times and break at 1.1 times | superseded |
| E5 vs E6 | Initial bilinear/tied certificates and one attack that could not break the bilinear majority certificate | Re-done with fine tube-width search and a function-level attack (E6, E7) | superseded |
| H2 | "No exact Willshaw code for majority at N of 32 and 64" | Artifact of an OR-style readout; with a least-squares readout exact codes exist (I2) | corrected |
| G2 | Trained transformers lose tolerance strongly with length | Weaker once trained properly (about N to the minus 0.6 to minus 0.8) | weakened |
| I6 | Online program addition: no earlier program survives | True for plain delta rule; the Adam-based write is a worse case from step normalization | clarified |
| J | LLM-built networks use a "cliff" failure mechanism for majority and dyck | Artifact of autodiff reporting zero slope at ReLU kinks placed at the operating point; with the one-sided derivative the first-order radius matches the attack radius (L) | retracted |
| K0 | Feedback condition results | Most likely run with the wrong condition flag for unconstrained cells; first answers all correct; later revisions in two cells changed architecture | compromised, reported separately |
| K0 | Model identity of last five literal-weight chats | Reported as GPT by the project owner; ids not confirmed | unresolved |
| K | Harness v1 vs v2 | v2 caches the attack compilation (about 7 times faster), first-failure value identical to the last digit on a test model | documented |
| L | Gradient-based predictors v1 and v2 | Rejected on the development set | negative |
| L | Predictor v3 on Tracr and bilinear networks | Failed (Tracr median log10 error 0.66; bilinear 2 to 7 times too optimistic) | negative |
| L | Predictor v4 | Works for the LLM-built family and bilinear networks, fails for Tracr `length` and `hist`; not declared successful by the pre-registered rule | partial |
| E7 | k=15 attack run died once | Resumed with the same script | documented |
| General | Environment | Background jobs were killed between tool calls in the sandbox, so long runs were executed in resumable chunks | documented |

## Appendix B. Glossary

* **Exact-sequence accuracy**: fraction of sequences for which every output position is correct.
* **sigma50**: absolute per-parameter noise standard deviation at which exact-sequence accuracy falls below 50 percent.
* **alpha50**: the same with noise proportional to each tensor's RMS value (scale-free).
* **Relative (`rel`) noise**: noise proportional to each weight; zeros stay zero.
* **Zero-only / non-zero-only noise**: noise only on entries that are zero / non-zero in the clean network.
* **Crosstalk**: effect of noise on unused (zero) weights on the computation.
* **Snapping / restoring state**: a nonlinearity applied each step that maps a noisy state back to a valid one (one-hot).
* **Digital design**: one-hot state with snapping. **Bilinear design**: state update z = sum over tokens of W_token times state, with snapping.
* **Weight-tied design**: several programs share role matrices; each program only chooses which role each token uses.
* **Certified radius**: the largest box of parameter perturbations for which an interval-bound proof of exactness holds. **Attack radius**: smallest box where a search finds a wrong output.
* **Tube**: set of states within a small distance of a valid one-hot state; the invariant of the certificate.
* **Slow weights / fast weights**: parameters shared by all programs / parameters written per program from a program tape.
* **Saturation margin**: how far a saturating unit (attention top-set gap, clip pre-activation) is from switching.
* **Tangent-cone (directional-derivative) model**: first-order model of a piecewise-linear function using one-sided derivatives.
* **Contaminated / fresh run**: LLM run by the instance that wrote the harness / by sessions that had seen nothing of the project.
* **Pre-registered**: hypotheses and success criteria written before the data were produced.
