import json, warnings
import numpy as np
warnings.filterwarnings("ignore")
exec(open("cert_attack2.py").read().split("out = {}")[0])
res = json.load(open("cert_attack2.json")); out = {}
for name, spec in specs.items():
    eps, d = res[name]["eps"], res[name]["tube"]; row = {}
    for mult in [1.0, 1.1, 1.25, 1.5, 2.0]:
        w = max(attack(spec, mult * eps, d, steps=200, restarts=6, seed=s) for s in range(2))
        row[mult] = w
    out[name] = row
    print(f"{name:24s} (eps*={eps:.2e}, tube={d}) worst violation by PGD attack (negative = guarantee holds): " + "  ".join(f"{m}x:{v:+.4f}" for m, v in row.items()), flush=True)
json.dump({k: {str(m): v for m, v in r.items()} for k, r in out.items()}, open("cert_attack3.json", "w")); print("DONE")
