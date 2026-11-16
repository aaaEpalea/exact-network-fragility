"""v2 predictor: simulate the Gaussian-noise experiment on the FIRST-ORDER model m_i(theta+delta) ~ m_i + g_i . delta (no constants, no fitting)."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))   # repository root
import sys, os, glob, json, re, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, REPO + "/llm_synthesis/harness")
import tasks as T, family as F, evaluate as E
SIGMAS = E.SIG
def build(sub, task):
    cfg, params = F.load_submission(sub); ev = E.FamilyEval(task, cfg, params); names = sorted(ev.params); shapes = {k: ev.params[k].shape for k in names}; sizes = [int(np.prod(shapes[k])) for k in names]
    flat0 = jnp.concatenate([ev.params[k].ravel() for k in names]); off = np.concatenate([[0], np.cumsum(sizes)]); unflat = lambda v: {k: v[off[i]:off[i + 1]].reshape(shapes[k]) for i, k in enumerate(names)}
    def margins(v):
        p = unflat(v); out = []
        for L in ev.lens:
            o = F.forward(p, cfg, ev.tok[L])[:, 1:]; e = ev.exp_arr[L]
            if ev.t["kind"] == "cat": corr = jnp.take_along_axis(o, e[..., None], -1)[..., 0]; oth = jnp.where(jax.nn.one_hot(e, o.shape[-1]) > 0, -1e9, o).max(-1); m = corr - oth
            else: m = ev.t["atol"] - jnp.abs(o[..., 0] - e)
            out.append(m.ravel())
        return jnp.concatenate(out)
    starts = np.concatenate([[0] + list(np.cumsum([len(ev.tok[L]) * L for L in ev.lens])[:-1])]); seqstart = []; pos = 0
    for L in ev.lens:
        for b in range(len(ev.tok[L])): seqstart.append(pos); pos += L
    return np.asarray(margins(flat0)), np.asarray(jax.jacrev(margins)(flat0)), np.asarray(flat0), np.array(seqstart)
def lin_s50(m, G, w, seqstart, mode, draws=60, seed=0):
    rng = np.random.default_rng(seed); scale = {"abs": np.ones_like(w), "rel": np.abs(w), "nonzero_only": (w != 0).astype(float), "zero_only": (w == 0).astype(float)}[mode]; Gs = G * scale[None]; curve = {}
    for s in SIGMAS:
        Z = Gs @ (s * rng.standard_normal((len(w), draws))); mm = np.minimum.reduceat(m[:, None] + Z, seqstart, axis=0); curve[str(s)] = [float((mm > 0).all(0).mean() if False else (mm > 0).mean())]
    return E.s50(curve)
def run(subdir, resdir, pattern, prefix, out):
    res = {}
    for f in sorted(glob.glob(subdir + "/" + pattern)):
        base = os.path.basename(f).rsplit(".", 1)[0]; rj = f"{resdir}/{base}.json"
        if not os.path.exists(rj): continue
        r = json.load(open(rj))
        if r.get("status") != "correct" or "noise" not in r: continue
        t = re.match(prefix + r"(\w+?)__(constrained|unconstrained)__", base).group(1); m, G, w, ss = build(f, t)
        res[base] = dict(task=t, s50={k: r["noise"][k]["s50"] for k in ("abs", "rel", "nonzero_only", "zero_only")}, lin={k: lin_s50(m, G, w, ss, k) for k in ("abs", "rel", "nonzero_only", "zero_only")}, cert=r["cert_radius_sample"], atk=r["attack_first_break"], rho_lin=float((m / (np.abs(G).sum(1) + 1e-30)).min()))
        print(base[:50], {k: f"{res[base]['s50'][k]:.2e}/{res[base]['lin'][k]:.2e}" for k in ("abs", "rel")}, flush=True); json.dump(res, open(out, "w"), indent=1)
if __name__ == "__main__":
    if sys.argv[1] == "dev": run(REPO + "/llm_synthesis/submissions", REPO + "/llm_synthesis/results", "contam__*", "contam__", "linsim_dev.json")
    else: run(REPO + "/llm_synthesis/fresh_run/submissions", REPO + "/llm_synthesis/fresh_run/results", "*__none__a*", "", "linsim_heldout.json")
