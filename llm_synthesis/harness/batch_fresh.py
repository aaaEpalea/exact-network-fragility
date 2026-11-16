"""python batch_fresh.py clean|full : evaluates every file in submissions/ (fresh run). clean_check cells: every revision gets a clean-only evaluation; the FINAL revision (highest) is the attempt's submission."""
import sys, os, glob, subprocess, json, re
H = os.path.dirname(os.path.abspath(__file__)); S = H + "/../submissions"; R = H + "/../results"; mode = sys.argv[1]; flt = sys.argv[2] if len(sys.argv) > 2 else ""; os.makedirs(R, exist_ok=True)
pat = re.compile(r"(\w+?)__(constrained|unconstrained)__(3b|4)__(none|clean_check)__(a\d+)(?:_rev(\d+))?$")
for p in sorted(glob.glob(S + "/*.py") + glob.glob(S + "/*.json")):
    name = os.path.basename(p).rsplit(".", 1)[0]; m = pat.match(name)
    if not m or (flt and flt not in name): continue
    base = name.split("_rev")[0]; revs = sorted(int(re.search(r"_rev(\d+)", q).group(1)) for q in glob.glob(S + f"/{base}_rev*"))
    final = (m.group(4) == "clean_check" and m.group(6) is not None and int(m.group(6)) == max(revs)) or m.group(4) == "none"
    if mode == "full" and not final: continue
    out = f"{R}/{name}" + (".clean.json" if mode == "clean" else ".json")
    if os.path.exists(out): continue
    args = ["python", "run_eval.py", "--task", m.group(1), "--submission", p, "--condition", m.group(2), "--out", out] + (["--clean-only"] if mode == "clean" else [])
    r = subprocess.run(args, capture_output=True, text=True, cwd=H); j = json.load(open(out)) if os.path.exists(out) else {"status": "crashed", "err": r.stderr[-300:]}
    print(f"{name:58s} {j.get('status'):20s} test={j.get('test_set_accuracy')} wrong={j.get('exhaustive',{}).get('n_wrong')} {(j.get('validation_errors') or [''])[0][:110]} {j.get('err','')[:150]}", flush=True)
