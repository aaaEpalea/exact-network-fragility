"""C4: certified radii (interval bounds, any length) and function-level attacks (certificate -> first failure) for the FWI; targets: fast weights only, slow weights only, joint ray."""
import json, time, os, numpy as np, jax
from fw_lib import *; from fw_cert import *; from exp_C2_robust import build_ok, dk_for
OUT = "../results/C4_cert.json"; t0 = time.time()
def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for P in [1, 2, 4, 8, 16]:
        key = f"cert|P{P}"
        if key in res: continue
        progs = pool("mixed", P, 21); th, phi, d, nmax, Kmax = build_ok(progs, dk_for("mixed", P)); rad = certified_radii(progs, th, phi, nmax, Kmax); res[key] = dict(dk=d, **rad); json.dump(res, open(OUT, "w"), indent=1)
        print(f"CERT P={P:2d} dk={d}: eps_fast={rad['eps_F']:.2e} eps_slow={rad['eps_theta']:.2e} joint t={rad['joint_t']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    for P in [1, 4]:
        key = f"attack|P{P}"
        if key in res: continue
        progs = pool("mixed", P, 21); th, phi, d, nmax, Kmax = build_ok(progs, dk_for("mixed", P)); rad = res[f"cert|P{P}"]; L = 32
        X = attack_set(P, L, B=40); refs = np.stack([ref_run(p, X) for p in progs]); out = res.get(key + "|partial", {})
        for tgt, (eF, eT, tc) in {"fast": (rad["eps_F"], 0.0, 1.0), "slow": (0.0, rad["eps_theta"], 1.0), "joint": (rad["eps_F"], rad["eps_theta"], rad["joint_t"])}.items():
            if tgt in out: continue
            tb, m0, lo = first_break(progs, th, phi, refs, X, eF, eT, tc if tgt == "joint" else 1.0, nmax, steps=40, restarts=1, grow=2.0, tmax=64, bis=3)
            out[tgt] = dict(eps_fast=eF, eps_slow=eT, t_cert=tc, first_break_t=tb, margin_at_cert=m0, largest_unbroken_t=lo); res[key + "|partial"] = out; json.dump(res, open(OUT, "w"), indent=1)
            print(f"ATTACK P={P} {tgt:5s}: cert t={tc:.2f}, margin at cert {m0:.3f}, first break at t={tb} (unbroken up to {lo:.2f})  ({time.time()-t0:.0f}s)", flush=True)
        res[key] = out; json.dump(res, open(OUT, "w"), indent=1)
    print("DONE", flush=True)
if __name__ == "__main__": main()
