import json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import digital_compiler as dc
from zeros_empirical import majority_ab, dyck_ab
N = 16
pool = [majority_ab(N, 0, 1), dyck_ab(N, 0, 1), majority_ab(N, 2, 3), dyck_ab(N, 2, 3)]
DELTAS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1)
def refined(spec, protect):
    p, meta = dc.compile_spec(spec)
    ok = lambda e: next((d for d in DELTAS if e <= d and dc.tube_ok(spec, p, meta, e, d, protect)), None)
    lo = hi = None
    for e in np.logspace(-5, -1.5, 8):
        if ok(e) is not None: lo = e
        else: hi = e; break
    for _ in range(6):
        mid = float(np.sqrt(lo * hi))
        if ok(mid) is not None: lo = mid
        else: hi = mid
    return lo, meta["D"]
res = {}; t0 = time.time()
for k in [1, 2, 4]:
    spec = pool[0] if k == 1 else dc.compose_parallel(pool[:k]); e0, D = refined(spec, False); e1, _ = refined(spec, True)
    res[f"k{k}"] = {"D": D, "dense": e0, "protected": e1}; print(f"k={k} D={D:3d}: certified eps*  dense={e0:.2e}  zeros-protected={e1:.2e}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open("comp_cert.json", "w"))
print("DONE", flush=True)
