"""C5: composition / sequential execution on ONE shared interpreter (Q10).
 (a) pipeline A->B: the readout symbol of program A (time-multiplexed micro-step 1) is the input token of program B (micro-step 2); both programs live in the same fast-weight memory F, state registers are separate.
     exactness vs the reference composition, robustness (fast-only / slow-only / joint static noise) vs the same two programs run alone, and vs pipelined independent networks.
 (b) online program addition: programs are written into the SAME memory one after another (delta rule on the new tape only, no rehearsal); do earlier programs survive?  with rehearsal?"""
import json, time, os, numpy as np, jax, jax.numpy as jnp
from fw_lib import *; from baselines import *; from exp_C2_robust import build_ok, dk_for
OUT = "../results/C5_comp.json"; t0 = time.time()
def run_pipe(th, phi, X, pA, pB, nmax):
    dk = th["A"].shape[0]; sq = jnp.sqrt(dk * 1.0); xs = jnp.swapaxes(jax.nn.one_hot(X, V), 0, 1); B = X.shape[0]; dR, dS = phi["R"].shape[1], phi["S0"].shape[1]
    snap = lambda z: jnp.clip(th["a"] * z - th["b"], 0.0, 1.0); gA, gB = th["C"][:, pA], th["C"][:, pB]
    init = lambda g: jnp.broadcast_to(snap(phi["S0"] @ (g[:dS] / sq)), (B, nmax))
    def stepf(s, x, g):
        kv = (x @ th["A"].T) * (s @ th["B"].T) * g / sq; s2 = snap(kv @ phi["F"].T); rk = ((s2 @ th["B"].T) * g / sq)[:, :dR]; return s2, rk @ phi["R"].T
    def f(c, x):
        sA, sB = c; sA2, yA = stepf(sA, x, gA); xB = jax.nn.one_hot(jnp.argmax(yA, -1), V); sB2, yB = stepf(sB, xB, gB); return (sA2, sB2), jnp.argmax(yB, -1)
    _, ys = jax.lax.scan(f, (init(gA), init(gB)), xs); return jnp.swapaxes(ys, 0, 1)
def run_pipe_indep(pa, pb, X):                      # independent networks, discrete hand-off
    ya = jnp.argmax(run_bil(pa, X), -1); return jnp.argmax(run_bil(pb, ya), -1)
def pipe_ref(A, B, X): return ref_run(B, ref_run(A, X))
def pipeline():
    pr = [majority_ab(10, 0, 1), dyck_ab(19, 0, 1), mod_counter(5, 0), reset_counter(21, 0, 1, 5), random_fsm(21, 4, 77)]; pairs = [(0, 2), (0, 3), (1, 2), (1, 3), (0, 4), (2, 3), (3, 4)]
    nmax, Kmax = 21, 5; th, phi, d, _, _ = build_ok(pr, 288); thj, phj = to_j(th), to_j(phi); L = 40; rng = np.random.default_rng(0); X = rng.choice(V, size=(32, L)); Xj = jnp.array(X)
    refs = {p: pipe_ref(pr[p[0]], pr[p[1]], X) for p in pairs}; jf = {p: jax.jit(lambda t, h, p=p: run_pipe(t, h, Xj, p[0], p[1], nmax)) for p in pairs}
    ok = lambda t, h: np.array([float((np.asarray(jf[p](t, h)) == refs[p]).all(1).mean()) for p in pairs])
    clean = ok(thj, phj); print("pipelines", pairs, "clean exact:", clean, flush=True); r = dict(pairs=pairs, dk=d, clean=clean.tolist())
    ind = build_indep(pr); indj = [{k: jnp.asarray(v) for k, v in p.items()} for p in ind]; jfi = {p: jax.jit(lambda a, b, p=p: run_pipe_indep(a, b, Xj)) for p in pairs}
    ok_i = lambda lst: np.array([float((np.asarray(jfi[p](lst[p[0]], lst[p[1]])) == refs[p]).all(1).mean()) for p in pairs]); r["indep_clean"] = ok_i(indj).tolist()
    for mode in ["phi", "theta", "both"]:
        cur = {}
        for s in SIG:
            v = []
            for i in range(3):
                th2, ph2 = perturb(thj, phj, jax.random.PRNGKey(10 + i), s if mode != "phi" else 0.0, s if mode != "theta" else 0.0); v.append(ok(th2, ph2))
            cur[str(s)] = np.mean(v, 0).tolist()
        r[f"pipe_{mode}"] = dict(per_pipeline=[s50({k: [c[j]] for k, c in cur.items()}) for j in range(len(pairs))], mean_curve=s50({k: [float(np.mean(c))] for k, c in cur.items()}))
        # the same programs run alone
        Pn = len(pr); X0, refs0, acc = make_eval(pr, nmax, Kmax, L, seed=0); r[f"alone_{mode}"] = sigma50_curve(acc, thj, phj, mode)["per_program"]
    cur = {}
    for s in SIG:
        v = []
        for i in range(3):
            lst = [noisy(p, s, jax.random.PRNGKey(100 * q + i)) for q, p in enumerate(indj)]; v.append(ok_i(lst))
        cur[str(s)] = [float(np.mean(np.mean(v, 0)))]
    r["indep_pipeline_sigma50"] = s50(cur)
    print(f"pipeline: FWI sigma50 fast-only {r['pipe_phi']['mean_curve']:.3f} slow-only {r['pipe_theta']['mean_curve']:.3f} joint {r['pipe_both']['mean_curve']:.3f} | same programs alone {r['alone_phi']:.3f}/{r['alone_theta']:.3f}/{r['alone_both']:.3f} | independent pipelined nets {r['indep_pipeline_sigma50']:.3f}", flush=True)
    return r
OPT = "sgd"
def online():
    Ptot = 12; progs = pool("mixed", Ptot, 21); nmax, Kmax = 21, max(p["K"] for p in progs); out = {}
    for tag, dk in [("dk_1.3x_min16", int(round(dk_for("mixed", 16) / 16)) * 16), ("dk_0.8x_min16", int(round(0.8 * dk_for("mixed", 16) / 1.3 / 16)) * 16)]:
        th = make_theta(dk, nmax, Ptot, 0); res = []
        # NOTE: tape([pr]) uses program index 0 for the code column; rebuild with correct program index
        full = tape(progs, th, nmax, Kmax)["F"]; T = triples(progs, nmax); owner = T[:, 0]
        Fm, t = write_memory(full[0][owner < 4], full[1][owner < 4], steps=1500); okp = lambda F, q: margins_ok(full[0][owner == q] @ F.T, full[1][owner == q]); F_inc = Fm.copy(); F_reh = Fm.copy()
        for q in range(4, Ptot):
            sel = owner == q; F_inc, t1 = write_memory(full[0][sel], full[1][sel], steps=1500, F0=F_inc, opt=OPT)
            msel = owner <= q; F_reh, t2 = write_memory(full[0][msel], full[1][msel], steps=1500, F0=F_reh, opt=OPT)
            old_inc = int(sum(okp(F_inc, j) for j in range(q))); old_reh = int(sum(okp(F_reh, j) for j in range(q))); new_inc = bool(okp(F_inc, q)); new_reh = bool(okp(F_reh, q))
            res.append(dict(n_programs=q + 1, incremental=dict(new_ok=new_inc, old_still_exact=old_inc, old_total=q, steps=t1), rehearsal=dict(new_ok=new_reh, old_still_exact=old_reh, steps=t2)))
            print(f"online[{OPT}] {tag} add program #{q+1}: no-rehearsal new ok={new_inc} old exact {old_inc}/{q} | with rehearsal new ok={new_reh} old exact {old_reh}/{q} (steps {t2})  ({time.time()-t0:.0f}s)", flush=True)
        out[tag + "|" + OPT] = dict(dk=dk, steps=res)
    return out
def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if "pipeline" not in res: res["pipeline"] = pipeline(); json.dump(res, open(OUT, "w"), indent=1)
    if "online" not in res: res["online"] = online(); json.dump(res, open(OUT, "w"), indent=1)
    print("DONE", flush=True)
if __name__ == "__main__": main()
