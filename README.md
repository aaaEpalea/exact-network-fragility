# How fragile are exactly-correct neural networks?

Code, harness, data and report for a study of how much random noise on the parameters an *exactly correct* network tolerates, and how that depends on how the network was built: compiled from a program (Tracr), hand-constructed (restoring-state and bilinear designs), trained, composed from several programs, or written by a large language model.

* **Report (read this first):** WEBSITE_REPORT_URL
* **Report source:** [`report/REPORT.md`](report/REPORT.md) with figures in [`report/figures/`](report/figures/)
* **Lab notebook with every result table:** [`RESULTS.md`](RESULTS.md)
* **Consolidated dataset (115 networks):** [`results/implementation_dataset.csv`](results/implementation_dataset.csv)

## Main findings (details and caveats in the report)

1. Tracr-compiled transformers lose exactness at noise levels of about 1e-4 to 1e-3 per parameter, 15 to 170 times below trained transformers (relative scale); the cause is concentrated in precise analog readouts from attention averages.
2. Hand-built restoring-state designs tolerate about 0.1, are exact for any length and carry certificates within about 2 to 3 times of a real counterexample. The construction is a re-derivation of Omlin and Giles (1996) and a related patent, not a new method.
3. Hardening compiled models, error-correcting codes and a shared fast-weight interpreter did not give the hoped-for gains (negative results are reported in full).
4. LLM-written transformers were exactly correct in 57 of 60 first answers and are about as fragile as compiled ones, with up to 38 times variation between attempts of the same model.
5. A parameter-free first-order model with saturation margins predicts failure levels well for LLM-built and bilinear networks but fails for two compiled programs.

## Layout

| Path | Content |
|---|---|
| `report/` | the technical report (Markdown, parts, sources, figures, scripts that rebuild it) |
| `RESULTS.md` | consolidated lab notebook |
| `code/` | experiment code for Parts A to I |
| `results/` | JSON results and logs; `implementation_dataset.csv` |
| `figures/` | figures produced during the experiments |
| `llm_synthesis/` | frozen protocol, harness, prompts, contaminated run, fresh-session run (`fresh_run/`), changelog |
| `theory/` | pre-registrations, predictors, cross-construction tests |
| `THEORY_CANDIDATES.md`, `THEORY_EXTRACTION.md`, `PRIOR_ART_AUDIT.md`, `PRIOR_ART_NOTES.md` | theory and literature documents |
| `tools/` | helper scripts |

## Getting started

See [`REPRODUCE.md`](REPRODUCE.md) for installation and the command for each experiment.

## Licences and citation

Code: MIT (`LICENSE`). Text, figures and data: CC BY 4.0 (`LICENSE-DATA.md`). Citation metadata: `CITATION.cff`.

## Status and honesty notes

This is a lab-notebook-style technical report, not a peer-reviewed paper. Parts A to I are exploratory; the LLM-synthesis protocol and the predictor tests (Parts J to L) were pre-registered. Known deviations (a compromised feedback condition, an unresolved model identity for five chats, retracted claims) are listed in Appendix A of the report.
