"""usage: python run_eval.py --task parity --submission ../submissions/x.py --condition constrained|unconstrained [--out file.json] [--quick] [--meta meta.json]
Validates the submission, applies the clean-correctness gate (exhaustive over all 335,922 sequences + hidden test set), and ONLY IF it passes runs the full robustness evaluation."""
import argparse, json, time, sys, numpy as np
import tasks as T, family as F, evaluate as E
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--submission", required=True); ap.add_argument("--condition", required=True, choices=["constrained", "unconstrained"])
ap.add_argument("--out"); ap.add_argument("--quick", action="store_true"); ap.add_argument("--clean-only", action="store_true"); ap.add_argument("--meta"); a = ap.parse_args()
t0 = time.time(); rec = dict(task=a.task, condition=a.condition, submission=a.submission, meta=json.load(open(a.meta)) if a.meta else {})
cfg, params = F.load_submission(a.submission); errs = F.validate(a.task, cfg, params, a.condition); rec["config"] = cfg; rec["validation_errors"] = errs
if errs: rec["status"] = "invalid_submission"
else:
    ev = E.FamilyEval(a.task, cfg, params); rec["params"] = E.param_stats(params); rec["test_set_accuracy"] = ev.acc(ev.params); rec["exhaustive"] = ev.exhaustive_clean()
    rec["clean_correct"] = bool(rec["exhaustive"]["all_correct"] and rec["test_set_accuracy"] == 1.0); rec["status"] = "correct" if rec["clean_correct"] else "synthesis_failure"
    if rec["clean_correct"] and not a.clean_only:
        rec["noise"] = E.noise_suite(ev, seeds=3 if a.quick else E.SEEDS, groups=not a.quick)
        rec["cert_radius_sample"] = E.cert_radius(ev); rec["attack_first_break"] = E.attack_break(ev, max(1e-6, rec["cert_radius_sample"]), steps=20 if a.quick else 30)
rec["seconds"] = time.time() - t0; print(json.dumps({k: v for k, v in rec.items() if k not in ("noise",)}, indent=1, default=str)[:3000])
if rec.get("noise"): print("sigma50:", {k: f"{v['s50']:.3g}" for k, v in rec["noise"].items()})
if a.out: json.dump(rec, open(a.out, "w"), indent=1, default=str)
