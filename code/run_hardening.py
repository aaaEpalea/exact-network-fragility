import json, time, warnings
warnings.filterwarnings("ignore")
from fragility import *
from hardening import *
SEEDS = 5
ABS = [float(x) for x in np.logspace(-5, -1, 13)]
CS = [4.0, 2.0, 0.5, 0.25, 0.1]
VARIANTS = {"base": dict(), "snap_all": dict(snap=True), "snap_layer": dict(snap=True, snap_sites="layer")}
for c in CS:
    VARIANTS[f"c={c}"] = dict(c=c)
    VARIANTS[f"c={c}+snap"] = dict(c=c, snap=True)
res, t0 = {}, time.time()
for name in ["hist", "sort_unique", "reverse", "length", "frac_prevs"]:
    prog, kind, atol, samp = build_programs()[name]
    T = Task(name, prog, kind, atol, samp); net = CompiledNet(T)
    fwd = jax.jit(net.predict)
    res[name] = {}
    for vn, kw in VARIANTS.items():
        hp = net.harden(T.params, **kw)
        clean = accuracy_hp(net, hp, fwd)
        r = {"clean": clean}
        if clean >= 0.999:
            r["abs"] = {str(s): [accuracy_hp(net, perturb(hp, s, jax.random.PRNGKey(k)), fwd) for k in range(SEEDS)] for s in ABS}
        res[name][vn] = r
        print(f"{name:11s} {vn:14s} clean={clean:.3f}", flush=True)
    json.dump(res, open("hardening.json", "w"))
    print(name, "done", round(time.time() - t0), "s", flush=True)
print("ALL DONE", flush=True)
