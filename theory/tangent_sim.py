import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))   # repository root
import sys, glob, json, re, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, REPO + "/llm_synthesis/harness")
import tasks as T, family as F, evaluate as E
def predict(sub, task, draws=150, s0=1e-5, seed=0):
    cfg, params = F.load_submission(sub); ev = E.FamilyEval(task, cfg, params); P = ev.params
    def margins(p):
        out = []
        for L in ev.lens:
            o = F.forward(p, cfg, ev.tok[L])[:, 1:]; e = ev.exp_arr[L]
            if ev.t["kind"] == "cat": corr = jnp.take_along_axis(o, e[..., None], -1)[..., 0]; oth = jnp.where(jax.nn.one_hot(e, o.shape[-1]) > 0, -1e9, o).max(-1); m = corr - oth
            else: m = ev.t["atol"] - jnp.abs(o[..., 0] - e)
            out.append(m.ravel())
        return jnp.concatenate(out)
    mj = jax.jit(margins); m0 = np.asarray(mj(P)); seqstart = []; pos = 0
    for L in ev.lens:
        for b in range(len(ev.tok[L])): seqstart.append(pos); pos += L
    seqstart = np.array(seqstart); keys = sorted(P); res = {}
    for mode in ("abs", "rel", "nonzero_only", "zero_only"):
        rng = np.random.default_rng(seed); fp = []
        for d in range(draws):
            q = {}
            for k in keys:
                z = rng.standard_normal(P[k].shape).astype(np.float32); w = np.asarray(P[k])
                z = z * (np.abs(w) if mode == "rel" else (w != 0) if mode == "nonzero_only" else (w == 0) if mode == "zero_only" else 1.0)
                q[k] = P[k] + s0 * jnp.asarray(z)
            fp.append((np.asarray(mj(q)) - m0) / s0)
        fp = np.stack(fp, 1); curve = {}
        for s in E.SIG: mm = np.minimum.reduceat(m0[:, None] + s * fp, seqstart, axis=0); curve[str(s)] = [float((mm > 0).mean())]
        res[mode] = E.s50(curve)
    return res
def run(subdir, resdir, pattern, prefix, out, only3b=True):
    import os; res = json.load(open(out)) if os.path.exists(out) else {}
    for f in sorted(glob.glob(subdir + "/" + pattern)):
        base = f.split("/")[-1].rsplit(".", 1)[0]; rj = f"{resdir}/{base}.json"
        if base in res: continue
        try: r = json.load(open(rj))
        except Exception: continue
        if r.get("status") != "correct" or "noise" not in r or "__3b__" not in base: continue
        t = re.match(prefix + r"(\w+?)__(constrained|unconstrained)__", base).group(1); pr = predict(f, t)
        res[base] = dict(task=t, pred=pr, meas={k: r["noise"][k]["s50"] for k in pr}); print(base[:46].ljust(46), {k: f"{res[base]['meas'][k]:.2e}/{pr[k]:.2e}" for k in ("abs", "rel")}, flush=True); json.dump(res, open(out, "w"), indent=1)
if __name__ == "__main__":
    if sys.argv[1] == "dev": run(REPO + "/llm_synthesis/submissions", REPO + "/llm_synthesis/results", "contam__*", "contam__", "tangent_dev.json")
    else: run(REPO + "/llm_synthesis/fresh_run/submissions", REPO + "/llm_synthesis/fresh_run/results", "*__none__a*", "", "tangent_heldout.json")
