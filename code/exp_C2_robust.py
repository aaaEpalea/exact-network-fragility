"""C2: robustness of the fast-weight interpreter (FWI) vs number of programs, and separated perturbation targets (Q6-Q9).
 (a) adding programs: program 0 is identical for every P; its sigma50 under fast-weight-only / slow-weight-only / joint static noise, plus joint (all programs exact), with dk = 1.3 x minimal, and with dk held fixed.
 (b) targets at P in {1,4,16}: slow-only (after writing), slow-only BEFORE writing (fast weights written under the perturbed slow weights = self-calibration), fast-only, joint, fresh-per-step versions, activation noise,
     snap gain/threshold only; absolute sigma50 and relative alpha50.  (c) independent-network and block-diagonal baselines on the same programs."""
import json, time, sys, os, numpy as np, jax, jax.numpy as jnp
from fw_lib import *; from baselines import *
OUT = "../results/C2_robust.json"; t0 = time.time(); C1 = json.load(open("../results/C1_scaling.json"))
def pstats(accf, th, phi, mode, rel=False, noise_mode="static", sigmas=SIG, NS=3, fix=None):
    pp0, ppm, jj = {}, {}, {}
    for s in sigmas:
        a0, am, jv = [], [], []
        for i in range(NS):
            key = jax.random.PRNGKey(2000 + i)
            if fix == "ab":   # perturb only snap gain / threshold
                th2 = dict(th); k1, k2 = jax.random.split(key); th2["a"] = th["a"] + s * jax.random.normal(k1); th2["b"] = th["b"] + s * jax.random.normal(k2); a, j = accf(th2, phi, key)
            elif noise_mode == "static":
                th2, phi2 = perturb(th, phi, key, s if mode in ("theta", "both") else 0.0, s if mode in ("phi", "both") else 0.0, rel); a, j = accf(th2, phi2, key)
            elif noise_mode == "fresh": a, j = accf(th, phi, key, dict(F=s if mode in ("phi", "both") else 0.0, theta=s if mode in ("theta", "both") else 0.0))
            elif noise_mode == "act": a, j = accf(th, phi, key, dict(act=s))
            a = np.asarray(a); a0.append(float(a[0])); am.append(float(a.mean())); jv.append(float(j))
        pp0[str(s)] = a0; ppm[str(s)] = am; jj[str(s)] = jv
    return dict(prog0=s50(pp0), mean=s50(ppm), joint=s50(jj))
def dk_for(kind, P, f=1.3): return int(round(C1[f"{kind}|P{P}"]["dk"] * f / 16)) * 16
def build_ok(progs, dk):
    nmax = max(p["n"] for p in progs); Kmax = max(p["K"] for p in progs)
    for seed in range(3):
        for f in (1.0, 1.15, 1.3):
            d = int(round(dk * f / 16)) * 16; th, phi, info = build_interpreter(progs, d, seed, nmax=nmax, Kmax=Kmax)
            if info["written"] and verify_exact(progs, th, phi, nmax, Kmax)["exact"]: return th, phi, d, nmax, Kmax
    raise RuntimeError("no exact interpreter")
def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    # ---------------- (a) adding programs
    fixed_dk = dk_for("mixed", 16)
    for P in [1, 2, 4, 8, 16]:
        key = f"adding|P{P}"
        if key in res: continue
        progs = pool("mixed", P, 21); r = {}
        for tag, dk in [("prop", dk_for("mixed", P)), ("fixeddk", fixed_dk)]:
            if tag == "fixeddk" and dk == dk_for("mixed", P): r[tag] = r["prop"]; continue
            th, phi, d, nmax, Kmax = build_ok(progs, dk); X, refs, acc = make_eval(progs, nmax, Kmax, 40); thj, phj = to_j(th), to_j(phi)
            r[tag] = dict(dk=d, **{m: pstats(acc, thj, phj, m) for m in ["phi", "theta", "both"]}, phi_rel=pstats(acc, thj, phj, "phi", rel=True), theta_rel=pstats(acc, thj, phj, "theta", rel=True))
            print(f"adding P={P:2d} {tag:7s} dk={d:5d}: prog0 sigma50 fast-only {r[tag]['phi']['prog0']:.3f} slow-only {r[tag]['theta']['prog0']:.3f} joint {r[tag]['both']['prog0']:.3f} | all-programs-exact: fast {r[tag]['phi']['joint']:.3f} slow {r[tag]['theta']['joint']:.3f} both {r[tag]['both']['joint']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        ev = Evaluator(progs, 40); r["indep"] = {k: baseline_s50(ev, "indep", rel=(k == "rel")) for k in ["abs", "rel"]}; r["blockdiag"] = {k: baseline_s50(ev, "blockdiag", rel=(k == "rel")) for k in ["abs", "rel"]}
        print(f"   baselines P={P}: indep abs per-prog {r['indep']['abs']['per_program']:.3f} joint {r['indep']['abs']['joint']:.3f} | blockdiag per-prog {r['blockdiag']['abs']['per_program']:.3f} joint {r['blockdiag']['abs']['joint']:.3f}", flush=True)
        res[key] = r; json.dump(res, open(OUT, "w"), indent=1)
    # ---------------- (b) targets
    for P in [1, 4, 16]:
        key = f"targets|P{P}"
        if key in res: continue
        progs = pool("mixed", P, 21); th, phi, d, nmax, Kmax = build_ok(progs, dk_for("mixed", P)); X, refs, acc = make_eval(progs, nmax, Kmax, 40); thj, phj = to_j(th), to_j(phi); r = dict(dk=d)
        ps = pstats if P < 16 else (lambda *a, **k: pstats(*a, sigmas=SIG[::2], NS=2, **k))      # P=16: half the sigma grid, 2 seeds (compute)
        r["static_slow_after_write"] = ps(acc, thj, phj, "theta"); r["static_fast"] = ps(acc, thj, phj, "phi"); r["static_joint"] = ps(acc, thj, phj, "both")
        r["static_slow_rel"] = ps(acc, thj, phj, "theta", rel=True); r["static_fast_rel"] = ps(acc, thj, phj, "phi", rel=True); r["static_joint_rel"] = ps(acc, thj, phj, "both", rel=True)
        r["fresh_slow"] = ps(acc, thj, phj, "theta", noise_mode="fresh"); r["fresh_fast"] = ps(acc, thj, phj, "phi", noise_mode="fresh"); r["fresh_joint"] = ps(acc, thj, phj, "both", noise_mode="fresh")
        r["act_fresh"] = ps(acc, thj, phj, None, noise_mode="act"); r["gain_threshold_only"] = ps(acc, thj, phj, None, fix="ab")
        if P <= 4:      # slow weights perturbed BEFORE the program is written: the fast weights are written through the defective interpreter
            sg = [s for s in SIG if s <= 0.3][::2]; pp, jj = {}, {}
            for s in sg:
                av, jv = [], []
                for i in range(2):
                    th2, _ = perturb(thj, phj, jax.random.PRNGKey(3000 + i), s, 0.0); th2n = {k: np.asarray(v) for k, v in th2.items()}
                    _, ph2, inf = build_interpreter(progs, d, 0, nmax=nmax, Kmax=Kmax, th=th2n); a, j = acc(th2, to_j(ph2), jax.random.PRNGKey(i)); av.append(float(np.mean(a))); jv.append(float(j))
                pp[str(s)] = av; jj[str(s)] = jv
            r["slow_before_write"] = dict(mean=s50(pp), joint=s50(jj), curve={k: float(np.mean(v)) for k, v in pp.items()})
        res[key] = r; json.dump(res, open(OUT, "w"), indent=1)
        f = lambda k: f"{r[k]['mean']:.3f}/{r[k]['joint']:.3f}"
        print(f"targets P={P:2d} dk={d}: sigma50 mean/joint | static slow {f('static_slow_after_write')} fast {f('static_fast')} joint {f('static_joint')} | rel(alpha50) slow {f('static_slow_rel')} fast {f('static_fast_rel')} joint {f('static_joint_rel')} | fresh slow {f('fresh_slow')} fast {f('fresh_fast')} joint {f('fresh_joint')} | act {f('act_fresh')} | gain/thr {f('gain_threshold_only')}" + (f" | slow-before-write {r['slow_before_write']['mean']:.3f}/{r['slow_before_write']['joint']:.3f}" if "slow_before_write" in r else "") + f"  ({time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)
if __name__ == "__main__": main()
