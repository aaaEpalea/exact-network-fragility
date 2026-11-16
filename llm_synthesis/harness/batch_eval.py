"""batch: python batch_eval.py clean|full [name-filter]  -> results/<attempt>.json for every contaminated submission (skips existing)"""
import sys, os, glob, subprocess, json, re
mode = sys.argv[1]; flt = sys.argv[2] if len(sys.argv) > 2 else ""
sub = os.path.join(os.path.dirname(__file__), "..", "submissions"); res = os.path.join(os.path.dirname(__file__), "..", "results")
for p in sorted(glob.glob(sub + "/contam__*.py") + glob.glob(sub + "/contam__*.json")):
    if p.endswith("authoring.py"): continue
    name = os.path.basename(p).rsplit(".", 1)[0]; m = re.match(r"contam__(\w+?)__(constrained|unconstrained)__(3b|4)__(none|clean_check)__(a\d+)", name)
    if flt and flt not in name: continue
    out = os.path.join(res, name + (".clean.json" if mode == "clean" else ".json"))
    if os.path.exists(out): continue
    args = ["python", "run_eval.py", "--task", m.group(1), "--submission", p, "--condition", m.group(2), "--out", out, "--meta", "../records/meta_contaminated.json"] + (["--clean-only"] if mode == "clean" else [])
    r = subprocess.run(args, capture_output=True, text=True, cwd=os.path.dirname(__file__)); j = json.load(open(out)) if os.path.exists(out) else {"status": "crashed", "err": r.stderr[-300:]}
    print(f"{name:55s} {j.get('status'):20s} test_acc={j.get('test_set_accuracy')} exh_wrong={j.get('exhaustive',{}).get('n_wrong')} {j.get('validation_errors') or ''} {j.get('err','')}", flush=True)
