import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))   # repository root
import json, glob, os, re, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R = H + "/../results"; rows = []
pat = re.compile(r"(\w+?)__(constrained|unconstrained)__(3b|4)__(none|clean_check)__(a\d+)(?:_rev(\d+))?$")
for f in sorted(glob.glob(R + "/*.json")):
    if f.endswith(".clean.json") or "summary" in f: continue
    name = os.path.basename(f)[:-5]; m = pat.match(name)
    if not m: continue
    r = json.load(open(f)); d = dict(name=name, task=m.group(1), cond=m.group(2), cat=m.group(3), fb=m.group(4), att=m.group(5), rev=m.group(6), status=r["status"])
    if r["status"] == "correct" and "noise" in r:
        n = r["noise"]; d.update(params=r["params"]["n_params"], nz=r["params"]["nonzero_frac"], maxw=r["params"]["max_abs"], cfg=r["config"], cert=r["cert_radius_sample"], atk=r["attack_first_break"], s50={k: v["s50"] for k, v in n.items()})
    rows.append(d)
json.dump(rows, open(R + "/fresh_summary.json", "w"), indent=1)
con = {(x["task"], x["condition"], x["category"]): x for x in json.load(open(REPO + "/llm_synthesis/results/contaminated_summary.json"))}
f3 = lambda v: "-" if v is None else f"{v:.2e}"
print("CELL (3b, no feedback): fresh abs sigma50 per attempt | spread max/min | contaminated | params per attempt")
for t in ["length", "hist", "reverse", "parity", "majority8", "dyck8"]:
    for c in ["constrained", "unconstrained"]:
        ds = [d for d in rows if d["task"] == t and d["cond"] == c and d["cat"] == "3b" and d["fb"] == "none"]; v = [d["s50"]["abs"] for d in ds if d.get("s50")]
        print(f"{t:10s}{c[:5]:6s} {[f3(d['s50']['abs']) if d.get('s50') else d['status'] for d in ds]} spread {max(v)/min(v):.1f}x  contam {f3(con[(t,c,'3b')]['s50']['abs'])}  params {[d.get('params') for d in ds]} cert {[f3(d.get('cert')) for d in ds]} atk/cert {[round(d['atk']/d['cert'],1) if d.get('cert') else None for d in ds]}")
print("\nCATEGORY 4:")
for d in rows:
    if d["cat"] == "4": print(d["name"], d["status"], f3(d.get("s50", {}).get("abs")) if d.get("s50") else "", d.get("params"), f"nz={d['nz']:.3f}" if d.get("nz") else "")
print("\nFEEDBACK cells:")
for d in rows:
    if d["fb"] == "clean_check": print(d["name"], d["status"], f3(d.get("s50", {}).get("abs")) if d.get("s50") else "", d.get("params"))
allabs = [d["s50"]["abs"] for d in rows if d.get("s50") and d["fb"] == "none" and d["cat"] == "3b"]; print("\n3b none: n correct evaluated", len(allabs), "abs s50 range", f3(min(allabs)), f3(max(allabs)))
