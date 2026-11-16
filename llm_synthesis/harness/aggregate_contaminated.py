"""Builds results/contaminated_summary.json, per-attempt records, and the markdown tables for RESULTS.md section J from results/contam__*.json and the Tracr baseline."""
import json, glob, os, re
R = os.path.join(os.path.dirname(__file__), "..", "results"); rec = os.path.join(os.path.dirname(__file__), "..", "records"); tr = json.load(open(os.path.join(R, "tracr_baseline_same_harness.json")))
rows = []
for f in sorted(glob.glob(R + "/contam__*.json")):
    if f.endswith(".clean.json"): continue
    r = json.load(open(f)); name = os.path.basename(f)[8:-5]; m = re.match(r"(\w+?)__(constrained|unconstrained)__(3b|4)__(none|clean_check)__(a\d+)", name); n = r["noise"]
    d = dict(attempt=name, task=m.group(1), condition=m.group(2), category=m.group(3), feedback=m.group(4), status=r["status"], params=r["params"], config=r["config"], cert=r["cert_radius_sample"], attack=r["attack_first_break"],
             s50={k: v["s50"] for k, v in n.items()}, exhaustive=r["exhaustive"])
    rows.append(d)
    t = json.load(open(os.path.join(rec, "record_template.json"))); t.update(attempt_id=name + "__contaminated", run="contaminated", construction_category=m.group(3), synthesis_condition=m.group(2), task=m.group(1), llm_identity="Claude (Sonnet 5.5, project session)", llm_version="as in session system prompt",
        interface="claude.ai chat with code execution", sampling_settings="chat default, unknown", prompt_file=f"prompts/rendered/{m.group(1)}__{m.group(2)}__{m.group(3)}__{m.group(4)}.md", independent_attempt_number=1, new_chat_per_attempt=False, execution_feedback_available="none", revisions_allowed=0, revisions_used=0,
        code_execution_available_to_llm="yes (not used on the construction before it was finalised)", architecture=r["config"], n_params=r["params"]["n_params"], weight_generation_procedure=("code computing analytic weights (contaminated_authoring.py)" if m.group(3) == "3b" else "literal weight values written by hand in JSON"),
        raw_transcript_file="records/transcripts/contaminated_run_note.md", submission_file="submissions/contam__" + name + (".py" if m.group(3) == "3b" else ".json"), notes="clean correct on hidden test set and exhaustively; " + ("same weights as the 3b attempt" if m.group(3) == "4" else ""))
    json.dump(t, open(os.path.join(rec, "contam__" + name + ".json"), "w"), indent=1)
json.dump(rows, open(os.path.join(R, "contaminated_summary.json"), "w"), indent=1)
open(os.path.join(rec, "transcripts", "contaminated_run_note.md"), "w").write("The contaminated run was performed inside the project session; its transcript is the project conversation and is not separable. The authoring record is `submissions/contaminated_authoring.py` (3b) and the hand-written JSON files (category 4). The 16 constructions were written before the checker was run on any of them; the harness had been run only on a random model and on Tracr baselines.\n")
f3 = lambda v: "-" if v is None else ("inf" if v == float("inf") else "nan" if v != v else f"{v:.2e}")
L = ["| task | condition | cat | params | nonzero | max abs w | sigma50 abs | rel | nonzero-only | cert radius | attack first break | attack/cert | abs/cert |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for d in rows: L.append(f"| {d['task']} | {d['condition']} | {d['category']} | {d['params']['n_params']} | {d['params']['nonzero_frac']:.3f} | {d['params']['max_abs']:.0f} | {f3(d['s50']['abs'])} | {f3(d['s50']['rel'])} | {f3(d['s50']['nonzero_only'])} | {f3(d['cert'])} | {f3(d['attack'])} | {d['attack']/d['cert']:.1f} | {d['s50']['abs']/d['cert']:.0f} |")
G = ["| task | condition | embeddings | attn q/k | attn v/o | mlp in | mlp out | unembed | zero-only |", "|---|---|---|---|---|---|---|---|---|"]
for d in rows:
    if d["category"] == "3b": s = d["s50"]; G.append(f"| {d['task']} | {d['condition']} | " + " | ".join(f3(s[k]) for k in ["group/embeddings", "group/attn_query/key", "group/attn_value/out", "group/mlp_in", "group/mlp_out", "group/unembed", "zero_only"]) + " |")
C = ["| task | Tracr (compiler, category 1) abs | LLM constrained abs | LLM unconstrained abs | Tracr rel | LLM constrained rel | LLM unconstrained rel | Tracr params | LLM constrained / unconstrained params |", "|---|---|---|---|---|---|---|---|---|"]
for t in ["length", "hist", "reverse"]:
    c = next(d for d in rows if d["task"] == t and d["condition"] == "constrained" and d["category"] == "3b"); u = next(d for d in rows if d["task"] == t and d["condition"] == "unconstrained" and d["category"] == "3b")
    C.append(f"| {t} | {f3(tr[t]['abs']['s50'])} | {f3(c['s50']['abs'])} | {f3(u['s50']['abs'])} | {f3(tr[t]['rel']['s50'])} | {f3(c['s50']['rel'])} | {f3(u['s50']['rel'])} | {tr[t]['params']['n_params']} | {c['params']['n_params']} / {u['params']['n_params']} |")
open(os.path.join(R, "contaminated_tables.md"), "w").write("\n".join(L) + "\n\n" + "\n".join(G) + "\n\n" + "\n".join(C) + "\n"); print("\n".join(C)); print("\n".join(G[:4]))
