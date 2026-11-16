"""C1: exactness, parameters and execution cost of the shared fast-weight interpreter vs number P and diversity of programs (Q1,Q3,Q4,Q5).
For each (diversity, P): minimal key width dk (on a grid, bisection) for which ONE shared fast-weight memory executes all P programs exactly (exhaustive one-step certificate with
margin band + long random-sequence check); private-memory variant (a separate fast memory per program, same slow weights); and the baselines."""
import json, time, sys, os, numpy as np, jax, jax.numpy as jnp
from fw_lib import *; from baselines import *
NMAX = 21; OUT = "../results/C1_scaling.json"; t0 = time.time()
def try_dk(progs, dk, seed=0, steps=1000):
    nmax = max(pr["n"] for pr in progs); Kmax = max(pr["K"] for pr in progs); th, phi, info = build_interpreter(progs, dk, seed, steps=steps, nmax=nmax, Kmax=Kmax)
    return th, phi, info, info["written"] and verify_exact(progs, th, phi, nmax, Kmax)["exact"]
def min_dk(progs, seed=0, steps=1000, lo_f=0.25, hi_f=2.2, npts=14):
    M = len(triples(progs, 0)); grid = sorted({int(round(M * f / 16)) * 16 for f in np.geomspace(lo_f, hi_f, npts)}); grid = [g for g in grid if g >= 16]
    lo, hi = 0, len(grid) - 1; best = None; cache = {}
    # exponential-then-bisect over the sorted grid (monotone assumption, checked by verifying the final choice)
    while lo <= hi:
        mid = (lo + hi) // 2; th, phi, info, ok = try_dk(progs, grid[mid], seed, steps); cache[grid[mid]] = ok
        if ok: best = (grid[mid], th, phi, info); hi = mid - 1
        else: lo = mid + 1
    return best, M
def long_run_ok(progs, th, phi, L=60):
    nmax = max(pr["n"] for pr in progs); Kmax = max(pr["K"] for pr in progs); X, refs, acc = make_eval(progs, nmax, Kmax, L=L, B=32, seed=5); a, j = acc(to_j(th), to_j(phi), jax.random.PRNGKey(0)); return float(j)
def run():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for kind, Ps in [("template", [1, 2, 4, 8, 16]), ("family", [1, 2, 4, 8, 16]), ("mixed", [1, 2, 4, 8, 16]), ("random", [1, 2, 4, 8, 12])]:
        for P in Ps:
            key = f"{kind}|P{P}"
            if key in res: continue
            progs = pool(kind, P, NMAX); nmax = max(pr["n"] for pr in progs); Kmax = max(pr["K"] for pr in progs); st = diversity_stats(progs)
            best, M = min_dk(progs)
            r = dict(kind=kind, P=P, M_tape=M, nmax=nmax, Kmax=Kmax, diversity=st, names=[pr["name"] for pr in progs])
            if best is None: r.update(shared_found=False)
            else:
                dk, th, phi, info = best; ver = verify_exact(progs, th, phi, nmax, Kmax); lr = long_run_ok(progs, th, phi)
                fast = phi["F"].size + phi["R"].size + phi["S0"].size; slow_hashed = dk * (V + nmax) + 2; slow = slow_hashed + dk * P
                r.update(shared_found=True, dk=dk, exact_certificate=ver["exact"], long_run_joint_acc=lr, margins={k: ver[k] for k in ver if "min_pos" in k or "max_neg" in k}, write_steps=info["write_steps"],
                         params_fast=int(fast), params_slow_explicit_codes=int(slow), params_slow_hashed_codes=int(slow_hashed), params_total=int(fast + slow), params_total_hashed=int(fast + slow_hashed), macs_step=macs_per_step("fwi", progs, dk),
                         fast_rms=float(np.sqrt(np.mean(phi["F"] ** 2))), fast_frac_gt_001=float((np.abs(phi["F"]) > 0.01).mean()))
            # private fast memory per program (same kind of interpreter, no sharing between programs)
            priv = 0; pdks = []; ok_all = True
            if P <= 8:
                for pr in progs:
                    b, _ = min_dk([pr])
                    if b is None: ok_all = False; break
                    dkp, thp, php, ip = b; pdks.append(dkp); priv += php["F"].size + php["R"].size + php["S0"].size
                r.update(private_found=ok_all, private_dks=pdks, private_fast_params=int(priv))
            r.update(params_indep_dense=indep_params(progs), params_indep_nonzero=indep_nonzero(progs), params_blockdiag=blockdiag_params(progs), macs_indep=macs_per_step("indep", progs), macs_blockdiag=macs_per_step("blockdiag", progs),
                     params_onehot_shared_fast=int(nmax * V * nmax * P))
            tp, U = tied_dictionary_params(progs); r.update(params_tied_dictionary=int(tp), tied_U=int(U))
            res[key] = r; json.dump(res, open(OUT, "w"), indent=1)
            print(f"{key:14s} M={M:5d} U={st['distinct_token_maps']:3d}/{st['total_token_maps']:3d} | " + (f"dk_min={r['dk']:5d} exact={r['exact_certificate']} long={r['long_run_joint_acc']:.2f} FWI params {r['params_total']:7d} (fast {r['params_fast']}, slow {r['params_slow_explicit_codes']}) macs {r['macs_step']}" if r["shared_found"] else "shared: not found up to 2.2M")
                  + f" | indep {r['params_indep_dense']} blockdiag {r['params_blockdiag']} tiedDict {r['params_tied_dictionary']} | priv {r.get('private_fast_params')}  ({time.time()-t0:.0f}s)", flush=True)
def nscale(res):
    """does the fast-memory width needed per stored program grow with the number of states n?  (random automata, P = 1 and 4)"""
    for P in [1, 4]:
        for n in [8, 12, 16, 24, 32, 48, 64, 96]:
            key = f"nscale|P{P}|n{n}"
            if key in res: continue
            progs = [random_fsm(n, 4, 7000 + 11 * i + n) for i in range(P)]; M = len(triples(progs, 0)); best, _ = min_dk(progs, lo_f=0.08, hi_f=2.2, npts=18); r = dict(P=P, n=n, M_tape=M)
            if best is None: r.update(shared_found=False)
            else:
                dk, th, phi, info = best; ver = verify_exact(progs, th, phi, n, 4); fast = phi["F"].size + phi["R"].size + phi["S0"].size
                r.update(shared_found=True, dk=dk, dk_over_M=dk / M, exact_certificate=ver["exact"], params_fast=int(fast), params_F=int(phi["F"].size), params_indep_dense=indep_params(progs), params_indep_nonzero=indep_nonzero(progs), widths=info["widths"],
                         params_slow_explicit=int(dk * (V + n + P) + 2))
            res[key] = r; json.dump(res, open(OUT, "w"), indent=1)
            print(f"{key:20s} M={M:5d} " + (f"dk_min={r['dk']:5d} ({r['dk_over_M']:.2f} M)  F params {r['params_F']:7d}  fast {r['params_fast']:7d} vs indep dense {r['params_indep_dense']:7d} nonzero {r['params_indep_nonzero']:5d}  slow(explicit) {r['params_slow_explicit']}" if r["shared_found"] else "not found") + f"  ({time.time()-t0:.0f}s)", flush=True)
if __name__ == "__main__":
    run(); res = json.load(open(OUT)); nscale(res); print("DONE", flush=True)
