"""Refined certified radius (log bisection, several tube widths) vs number of state dims D."""
import json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import digital_compiler as dc
def refined(spec, deltas=(0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15)):
    p, meta = dc.compile_spec(spec)
    ok = lambda e: next((d for d in deltas if e <= d and dc.tube_ok(spec, p, meta, e, d)), None)
    lo, hi = None, None
    for e in np.logspace(-5, -1, 9):
        if ok(e) is not None: lo = e
        else: hi = e; break
    if lo is None: return 0.0, None, meta["D"]
    if hi is None: return lo, ok(lo), meta["D"]
    for _ in range(7):
        mid = float(np.sqrt(lo * hi))
        if ok(mid) is not None: lo = mid
        else: hi = mid
    return lo, ok(lo), meta["D"]
res = {}; t0 = time.time()
for fam, mk, Ns in [("majority", dc.majority, [2, 4, 8, 16, 32, 64]), ("dyck", dc.dyck, [2, 4, 8, 16, 32, 64])]:
    for N in Ns:
        eps, d, D = refined(mk(N)); res[f"{fam}|{N}"] = {"D": D, "eps": eps, "tube": d}
        print(f"{fam:9s} N={N:3d} D={D:4d}  refined certified eps* = {eps:.3e} (tube {d})  eps*xD = {eps*D:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open("cert_scaling.json", "w"))
print("DONE", flush=True)
