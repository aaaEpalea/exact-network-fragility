import json, warnings
warnings.filterwarnings("ignore")
from hybrid2 import run
res = {}
for N, seed in [(16, 0), (16, 1), (32, 1), (64, 0)]:
    res[f"N{N}|seed{seed}"] = run(N, False, 8000, seed, "hist"); json.dump(res, open("retrain_baselines.json", "w"))
print("DONE", flush=True)
