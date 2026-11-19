# Technical report: package contents (for conversion to the website)

* `REPORT.md` - the complete report as a single Markdown file (GitHub-flavoured; images as `![alt](figures/file.png)`; large raw tables folded in `<details>` blocks; no math markup; blockquotes mark the plain-language takeaways).
* `parts/part_A.md ... part_M.md` - the same text, one file per part (the combined file is built from them plus `src/`).
* `src/` - hand-written front matter, background and methods, synthesis, limitations, reproducibility, appendices, references.
* `figures/` - 19 figures (PNG, 140 dpi for the new ones). `FIGURES.md` lists every caption.
* `build_report.py`, `make_figures.py`, `overlay_AF.py`, `overlay_GM.py` - the scripts that regenerate the report and the new figures from `RESULTS.md`, the JSON result files and the theory documents, so numbers in the text and tables always match the data.

Conversion notes: `<details>` blocks are valid in MDX; the "Part" headings are level 2 and result blocks level 3 (A1, A2, ...) so a table of contents can be generated from the headings.
