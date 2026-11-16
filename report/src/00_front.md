# How fragile are exactly-correct neural networks? A technical report on constructed, compiled, trained and LLM-written networks under parameter noise

*Technical report and lab notebook. Code, harness, data and every intermediate result are in the accompanying repository: GITHUB_REPO_URL*

## Summary

We asked a narrow question: **if a neural network computes a program exactly, how much random noise on its parameters can it take before it starts to make mistakes, and does that depend on how the network was built?** We studied it with a stack of experiments on small, fully specified computations (counting, histograms, sorting, reversing, parity, majority, bracket matching), on networks built five different ways: compiled from a program (Tracr), constructed by hand ("digital" restoring-state and bilinear designs), trained by gradient descent, composed from several programs, and written by a large language model (an LLM).

The main findings, with the strength of evidence stated up front:

1. **Compiled transformers are very fragile.** Transformers produced by the Tracr compiler stop being exact at noise levels of about 1e-4 to 1e-3 per parameter, 15 to 170 times lower (relative scale) than trained transformers of the same depth. The fragility sits in precise analog readouts from attention averages; counting is the extreme case and gets worse with sequence length (Part A, B). *Exploratory, many seeds, consistent across six programs.*
2. **Hand-built "restoring-state" designs are far more tolerant**, because each step snaps the state back to a valid value, and they come with provable guarantees: exactness for sequences of any length and a certified noise radius, which we also attacked to show it is within about 2 to 3 times of a real counterexample (Parts D, E, I). **This construction is not new**: second-order recurrent networks that encode automata with stability proofs were published by Omlin and Giles in 1996, and a related patent covers fault-tolerant variants (see Part M). We reproduce and extend it; we do not claim it.
3. **Several clever ideas did not work**: hardening compiled models by rescaling gives only a few times improvement and does not change the decay with length; certificate-driven and noise-injection training broke exactness; error-correcting sparse codes and a shared "universal interpreter" with programs stored in fast weights execute different programs exactly but save no memory or robustness over separate networks (Parts C, H, I). *These are negative results, reported in full.*
4. **LLM-written transformers are exactly correct surprisingly often and are about as fragile as compiled ones, with large variation between attempts.** In a fresh-session experiment 57 of 60 first answers were exactly correct on all 335,922 inputs of length up to 7; robustness spanned 6e-4 to 0.12, and attempts by the same LLM at the same task differed by up to 38 times, as much as the difference between LLM-built and compiled networks (Parts J, K). *Preliminary: one model family, one to three attempts per cell; one of the four experimental conditions was compromised by an operator error (Part K0).*
5. **A simple first-order model explains much of where failures happen.** A parameter-free predictor, written down before the held-out test, predicts noise failure levels of the LLM-built networks with a rank correlation of 0.93 and of bilinear networks to within a factor of 2, but fails for two compiled programs (Part L). *Pre-registered and held out within one family; not general.*
6. **Prior art**: much of the digital/bilinear and fast-weight material overlaps published work; the controlled cross-construction comparison appears to be new, but our literature search was limited (Part M).

## How to read this report

Every result carries one of three labels.

* **Confirmatory**: the protocol, hypotheses and success criteria were written down and frozen *before* the data were produced (LLM-synthesis protocol; predictor v3, v4; the cross-construction test; the cliff experiment). The pre-registration files are in the repository.
* **Exploratory**: the experiment was run to see what happens; hypotheses were formed after looking at earlier results. This is most of Parts A to I.
* **Negative / corrected**: an idea that failed, or an earlier statement we had to retract. These are collected in the correction log (Appendix A) and marked in the text.

Parts are in the order in which the work was done, because later parts depend on earlier ones. Each result block lists the code and result files that produced it, then the table or figure, then what it shows, then a short plain-language takeaway. Large raw tables are folded. Terms such as sigma50 and alpha50 are defined in the glossary and in Section 2.

## Contents

1. Background and related work
2. Methods common to all experiments
3. Part A: Audit, compiled programs are fragile
4. Part B: Mechanism and scaling with sequence length
5. Part C: Hardening Tracr-compiled models
6. Part D: Digital (restoring-state) designs
7. Part E: Compiler and certificates
8. Part F: Zeros, crosstalk and composition
9. Part G: Trained-transformer comparisons and hybrids
10. Part H: Wild ideas
11. Part I: Weight-tied attacks, error-correcting codes, fast-weight universal interpreter
12. Part J: LLM-synthesized transformers, protocol and contaminated run
13. Part K: LLM-synthesized transformers, fresh-session run
14. Part L: Theory exploration, a first-order predictor of failure levels
15. Part M: Prior-art audit
16. Synthesis: what we know, what we do not, and what is open
17. Limitations
18. Reproducibility and repository guide
19. Appendix A: correction and deviation log
20. Appendix B: glossary
21. References
