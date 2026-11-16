import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))   # repository root
import sys, os, glob, json, re, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, REPO + "/llm_synthesis/harness")
import tasks as T, family as F, evaluate as E
def analyze(sub, task, cond):
    cfg, params = F.load_submission(sub); ev = E.FamilyEval(task, cfg, params); names = sorted(ev.params); shapes = {k: ev.params[k].shape for k in names}; sizes = [int(np.prod(shapes[k])) for k in names]
    flat0 = jnp.concatenate([ev.params[k].ravel() for k in names]); off = np.concatenate([[0], np.cumsum(sizes)])
    def unflat(v): return {k: v[off[i]:off[i + 1]].reshape(shapes[k]) for i, k in enumerate(names)}
    def margins(v):
        p = unflat(v); out = []
        for L in ev.lens:
            o = F.forward(p, cfg, ev.tok[L])[:, 1:]; e = ev.exp_arr[L]
            if ev.t["kind"] == "cat":
                corr = jnp.take_along_axis(o, e[..., None], -1)[..., 0]; oth = jnp.where(jax.nn.one_hot(e, o.shape[-1]) > 0, -1e9, o).max(-1); m = corr - oth
            else: m = ev.t["atol"] - jnp.abs(o[..., 0] - e)
            out.append(m.ravel())
        return jnp.concatenate(out)
    m = np.asarray(margins(flat0)); G = np.asarray(jax.jacrev(margins)(flat0)); w = np.asarray(flat0); assert (m > 0).all()
    grp = {g: np.concatenate([np.full(sizes[i], F.GROUPS[g](k)) for i, k in enumerate(names)]).astype(bool) for g in F.GROUPS}; nz = w != 0
    n2 = lambda mask, wt=1.0: np.sqrt(((G * mask * wt) ** 2).sum(1)) + 1e-30
    pred = {"abs": (m / n2(np.ones_like(w))).min(), "rel": (m / n2(np.ones_like(w), np.abs(w))).min(), "nonzero_only": (m / n2(nz)).min(), "zero_only": (m / n2(~nz)).min(), "rho_linf": (m / (np.abs(G).sum(1) + 1e-30)).min()}
    for g, mask in grp.items(): pred["group/" + g] = (m / n2(mask)).min()
    pred["min_margin"] = float(m.min()); pred["n_params"] = int(w.size); return {k: float(v) for k, v in pred.items()}
def run(subdir, resdir, pattern, prefix):
    out = {}
    for f in sorted(glob.glob(subdir + "/" + pattern)):
        base = os.path.basename(f).rsplit(".", 1)[0]; rj = f"{resdir}/{base}.json"
        if not os.path.exists(rj): continue
        r = json.load(open(rj))
        if r.get("status") != "correct" or "noise" not in r: continue
        m = re.match(prefix + r"(\w+?)__(constrained|unconstrained)__", base); t = m.group(1)
        out[base] = dict(pred=analyze(f, t, m.group(2)), s50={k: v["s50"] for k, v in r["noise"].items()}, cert=r["cert_radius_sample"], atk=r["attack_first_break"], task=t, cond=m.group(2)); print(base, round(out[base]["pred"]["abs"], 6), out[base]["s50"]["abs"], flush=True)
    return out
if __name__ == "__main__":
    which = sys.argv[1]
    if which == "dev": res = run(REPO + "/llm_synthesis/submissions", REPO + "/llm_synthesis/results", "contam__*", "contam__")
    else: res = run(REPO + "/llm_synthesis/fresh_run/submissions", REPO + "/llm_synthesis/fresh_run/results", "*__none__a*", "")
    json.dump(res, open(f"lin_{which}.json", "w"), indent=1)
