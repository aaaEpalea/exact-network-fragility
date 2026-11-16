import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))   # repository root
import json, os, numpy as np
P = REPO; rows = json.load(open("../results/fresh_summary.json")); con = json.load(open(P + "/llm_synthesis/results/contaminated_summary.json"))
f3 = lambda v: "-" if v is None else f"{v:.2e}"; TASKS = ["length", "hist", "reverse", "parity", "majority8", "dyck8"]; CONDS = ["constrained", "unconstrained"]
cell = lambda t, c, cat="3b", fb="none": sorted([d for d in rows if d["task"] == t and d["cond"] == c and d["cat"] == cat and d["fb"] == fb and d["rev"] is None], key=lambda d: d["att"])
ident = {("length", "constrained"): "3 of 3 identical to each other and to the contaminated attempt", ("length", "unconstrained"): "a1 = a2 identical; a3 differs", ("majority8", "unconstrained"): "a2 = a3 identical"}
T1 = ["| task | condition | a1 | a2 | a3 | spread (max/min) | contaminated | params (a1/a2/a3) | sample-cert radius (a1/a2/a3) | attack/cert (a1/a2/a3) | identical weights |", "|---|---|---|---|---|---|---|---|---|---|---|"]; spreads = {}; med = {}
for t in TASKS:
    for c in CONDS:
        ds = cell(t, c); v = [d["s50"]["abs"] for d in ds if d.get("s50")]; spreads[(t, c)] = max(v) / min(v); med[(t, c)] = float(np.median(v)); cc = next(x for x in con if x["task"] == t and x["condition"] == c and x["category"] == "3b")
        T1.append(f"| {t} | {c} | " + " | ".join(f3(d["s50"]["abs"]) if d.get("s50") else d["status"] for d in ds) + f" | {spreads[(t, c)]:.1f}x | {f3(cc['s50']['abs'])} | " + "/".join(str(d.get("params", "-")) for d in ds) + " | " + "/".join(f3(d.get("cert")) for d in ds) + " | " + "/".join(f"{d['atk']/d['cert']:.1f}" if d.get("cert") else "-" for d in ds) + f" | {ident.get((t, c), 'all differ')} |")
T2 = ["| attempt | status | params | nonzero | sigma50 abs | rel | nonzero-only | cert | attack |", "|---|---|---|---|---|---|---|---|---|"]
for d in sorted([d for d in rows if d["cat"] == "4"], key=lambda d: d["name"]): T2.append(f"| {d['name']} | {d['status']} | {d.get('params', '-')} | {('%.3f' % d['nz']) if d.get('nz') else '-'} | {f3(d['s50']['abs']) if d.get('s50') else '-'} | {f3(d['s50']['rel']) if d.get('s50') else '-'} | {f3(d['s50']['nonzero_only']) if d.get('s50') else '-'} | {f3(d.get('cert'))} | {f3(d.get('atk'))} |")
T3 = ["| attempt file (revision) | status | params | sigma50 abs |", "|---|---|---|---|"]
for d in sorted([d for d in rows if d["fb"] == "clean_check"], key=lambda d: d["name"]): T3.append(f"| {d['name']} | {d['status']} | {d.get('params', '-')} | {f3(d['s50']['abs']) if d.get('s50') else '-'} |")
a3 = [d for d in rows if d["cat"] == "3b" and d["fb"] == "none" and d.get("s50")]; ratio = [d["s50"]["abs"] / d["cert"] for d in a3]; ac = [d["atk"] / d["cert"] for d in a3]; hi = [d for d in a3 if d["s50"]["abs"] >= 0.1]
e2 = [f"{t} {c} {s:.0f}x" for (t, c), s in spreads.items() if s >= 10]; e4 = "; ".join(f"{t} {med[(t,'unconstrained')]/med[(t,'constrained')]:.1f}" for t in TASKS)
sec = f"""## K. LLM-synthesized transformers: fresh-session run (uncontaminated), first pass

**Status: preliminary pointer data; one LLM family in chat UI, up to 3 attempts per cell. Never pooled with the contaminated run (J). Protocol deviations are listed in K0.** Frozen package unchanged (`freeze.py --check`: package unchanged, sha256 `a528b01d...`). Files: `llm_synthesis/fresh_run/` (submissions exactly as received, records, per-attempt results `*.json`, `fresh_summary.json`), `harness/batch_fresh.py`, `harness/aggregate_fresh.py`, `harness/splice_K.py`, `HARNESS_CHANGELOG.md`.

### K0. Deviations and open items (read first)
* **Priority 3 (feedback) is compromised.** For all six unconstrained feedback cells the first answer is valid and correct, yet `feedback.py` run with `--condition constrained` returns a validation error on it (with the matching condition it prints 200/200); the project owner reports the chats ended with the model calling the fix impossible after a validation error. The most likely cause is that the wrong `--condition` was passed; not confirmed. In the constrained cells `hist` and `parity` the later revisions changed the architecture and are invalid under the constrained condition, although the first answers were correct. These 12 cells are reported but excluded from any feedback-vs-no-feedback comparison.
* **Model identity.** The project owner reports that the last five priority-4 chats used GPT instead of Claude. Exact attempt ids were not provided and no records/`meta.json` were included; priority-4 results below are therefore NOT attributed to a model. They must not be pooled with the Claude results until the ids are confirmed.
* **Missing material.** No transcripts for the 12 feedback chats and 12 category-4 chats, no per-attempt records, no sampling settings (chat defaults).
* **Harness v2.** `evaluate.py` was patched for speed (attack functions compiled once per model); the attack first-failure value is bit-identical to v1 on a test model. Group-wise tools and the frozen files are otherwise unchanged (`HARNESS_CHANGELOG.md`).

### K1. Clean-correctness gate (exhaustive over all 335,922 sequences + hidden test set)
First answers: **57 of 60 correct**. Priority 1-2 (3b, no feedback): 35 of 36 correct (failure: `hist` constrained a3, 0% accuracy). Priority 3 (feedback cells): 12 of 12 first answers correct. Priority 4 (literal weights): 10 of 12 correct (failures: `parity` constrained a3 and `parity` unconstrained a3, both 59% accuracy / 106,431 wrong sequences).

### K2. Category 3b, no feedback: robustness per attempt (absolute sigma50)

{chr(10).join(T1)}

Full per-attempt metrics (relative, nonzero-only, zero-only, parameter groups, certificate, attack) are in `fresh_summary.json` and the per-attempt result files.

### K3. Category 4 (literal weights; model attribution pending, see K0)

{chr(10).join(T2)}

### K4. Feedback cells (compromised, see K0)

{chr(10).join(T3)}

### K5. Pre-registered expectations vs the fresh run
* **E1** (synthesis failures common): **not supported** (57/60 first answers correct; 3 failures).
* **E2** (>= 10x spread between attempts of one cell): **supported in {len(e2)} of 12 cells** ({', '.join(e2)}); the other cells vary by 4.3x or less.
* **E3** (sigma50 nearer the Tracr range than the digital range >= 0.1): **not supported for the counter tasks**: {len(hi)} of {len(a3)} correct 3b attempts have absolute sigma50 >= 0.1 ({', '.join(d['name'].split('__3b')[0] + ' ' + d['att'] for d in hi)}), and `majority8`/`dyck8` span 0.026-0.12. `length`, `hist`, `reverse`, `parity` stay at 6.5e-4-2.5e-2.
* **E4** (constrained vs unconstrained differ via size, not margin structure): **mixed**; median unconstrained/constrained sigma50 per task: {e4}. 
* **E5** (category 4 only on the smallest tasks, sparse, integer-valued): feasible on `length` and `parity` (10/12 correct, non-zero fraction 1-28%), values not integer.
* **E6** (certificate orders of magnitude below sigma50; attack within ~10x of it): sigma50/cert ranges {min(ratio):.0f}-{max(ratio):.0f}x (median {np.median(ratio):.0f}x), i.e. as low as {min(ratio):.0f}x; attack/cert is <= 10x in {100*np.mean(np.array(ac) <= 10):.0f}% of attempts (max {max(ac):.0f}x).

### K6. Contaminated vs fresh, and convergence
* Fresh medians vs the contaminated attempt: within 1.1x for `length` (both conditions), `reverse` constrained and `parity` constrained; 2-4.4x for `hist` (both) and `parity` unconstrained; 8.7x for `reverse` unconstrained. `length` constrained: **the three fresh attempts and the contaminated attempt have bit-identical weights**. Other cells: only `length` unconstrained (a1 = a2) and `majority8` unconstrained (a2 = a3) repeat; all other attempts have different weights even when sigma50 is similar. So convergence to one design is the exception except for the simplest task, not the rule.
* `majority8` and `dyck8`: fresh attempts are mostly more tolerant than the contaminated ones (fresh medians/contaminated: `majority8` constrained 2.2x, unconstrained 11x; `dyck8` constrained 22x, unconstrained 8x; one fresh `majority8` constrained attempt equals it). A plausible explanation (untested) is a different counting design (signed averaging instead of the BOS-weight trick); the contaminated design was the less robust one here.
* Candidate explanation to test in Stage 5 (not a conclusion): robustness is set by the least-margin step of the chosen construction; the same LLM produces constructions whose sigma50 differs by up to 38x within a cell.

### K7. Limitations
Chat UI with unknown sampling; one model family (plus unattributed GPT chats in priority 4); at most 3 attempts per cell; contaminated and fresh runs differ in what the model had seen; tasks small; certificate sample-based; attack weak; priority 3 compromised; transcripts and records incomplete.

"""
s = open(P + "/RESULTS.md").read()
if "## K. LLM-synthesized" in s: s = s[:s.index("## K. LLM-synthesized")] + sec + s[s.index("## Other result files"):]
else: s = s.replace("## Other result files", sec + "## Other result files", 1)
row = "| Correct LLM-synthesized transformers (blind, contaminated, 1 attempt/cell, 6 tasks) are as fragile as Tracr-compiled ones (abs sigma50 6e-4..3e-2) | preliminary (J); fresh-session run pending |"
new = "| LLM-synthesized transformers (fresh session, up to 3 attempts/cell, 6 tasks) are exactly correct in 57/60 first answers; robustness varies by task and attempt (absolute sigma50 6e-4..0.12, up to 38x between attempts of one cell) | preliminary (K); feedback condition compromised, model attribution of priority 4 pending |"
if row in s: s = s.replace(row, new)
elif new not in s: s = s.replace("## Other result files", "", 0)
open(P + "/RESULTS.md", "w").write(s); print(len(s)); print("E2", e2, "| hi", [(d['name'], round(d['s50']['abs'], 3)) for d in hi], "| ratio", round(min(ratio)), round(max(ratio)), "| atk<=10", np.mean(np.array(ac) <= 10))
