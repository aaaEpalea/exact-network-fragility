import json, time, warnings
warnings.filterwarnings("ignore")
from fragility import *
from hardening import *
SEEDS = 5
ABS = [float(x) for x in np.logspace(-5, -1, 13)]
VARIANTS = {"base": dict(), "balance": dict(balance=True), "c=0.25": dict(c=0.25),
            "balance+c=0.5": dict(balance=True, c=0.5), "balance+c=0.25": dict(balance=True, c=0.25),
            "balance+c=0.1": dict(balance=True, c=0.1), "snap_all": dict(snap=True),
            "balance+snap": dict(balance=True, snap=True), "ideal_clip": dict(ideal=True),
            "balance+ideal_clip": dict(balance=True, ideal=True)}
res, t0 = {}, time.time()
for name in ["hist", "sort_unique", "reverse", "length", "frac_prevs"]:
    prog, kind, atol, samp = build_programs()[name]
    T = Task(name, prog, kind, atol, samp); net = CompiledNet(T)
    res[name] = {}
    for vn, kw in VARIANTS.items():
        ideal = kw.pop("ideal", False)
        fwd = jax.jit(lambda hp, t, _i=ideal: net.predict(hp, t, _i))
        hp = net.harden(T.params, **kw)
        clean = accuracy_hp(net, hp, fwd)
        r = {"clean": clean}
        if clean >= 0.999:
            for mode in ["abs", "nz"]:
                r[mode] = {str(s): [accuracy_hp(net, perturb(hp, s, jax.random.PRNGKey(k), mode), fwd) for k in range(SEEDS)] for s in ABS}
        res[name][vn] = r
        print(f"{name:11s} {vn:20s} clean={clean:.3f}", flush=True)
    json.dump(res, open("hardening2.json", "w"))
    print(name, "done", round(time.time() - t0), "s", flush=True)
print("ALL DONE", flush=True)
