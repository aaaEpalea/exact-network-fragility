import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))   # repository root
import sys, glob, json, re, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, REPO + "/llm_synthesis/harness"); sys.path.insert(0, REPO + "/code")
import numpy as np, jax, jax.numpy as jnp
import tasks as T, family as F, evaluate as E
def scores_forward(params, cfg, tokens):
    T_ = tokens.shape[1]; x = params["embed"][tokens] + params["pos_embed"][:T_][None]; dh = cfg["d_head"]; allowed = jnp.tril(jnp.ones((T_, T_), bool)) if cfg["causal"] else jnp.ones((T_, T_), bool); S = []
    for i in range(cfg["n_layers"]):
        p = lambda n: params[f"l{i}.{n}"]
        q = jnp.einsum("btd,hde->bhte", x, p("wq")) + p("bq")[None, :, None]; k = jnp.einsum("btd,hde->bhte", x, p("wk")) + p("bk")[None, :, None]; v = jnp.einsum("btd,hde->bhte", x, p("wv")) + p("bv")[None, :, None]
        s = jnp.einsum("bhte,bhse->bhts", q, k) / jnp.sqrt(dh * 1.0); S.append(s); s = jnp.where(allowed[None, None], s, -1e9); a = jax.nn.softmax(s, -1); o = jnp.einsum("bhts,bhse->bhte", a, v)
        x = x + jnp.einsum("bhte,hed->btd", o, p("wo")) + p("bo"); x = x + jax.nn.relu(x @ p("w1") + p("b1")) @ p("w2") + p("b2")
    return x @ params["unembed.w"] + params["unembed.b"], S, allowed
def gaps(S, allowed, tops):
    out = []
    for s, top in zip(S, tops):
        rest = allowed[None, None] & ~top; smin = jnp.min(jnp.where(top, s, 1e9), -1); smax = jnp.max(jnp.where(rest, s, -1e9), -1); g = jnp.where(rest.any(-1), smin - smax, 1e3); out.append(g[:, :, 1:])   # (B,H,L)
    return out
def family(sub, task, draws=150, s0=1e-5):
    cfg, params = F.load_submission(sub); ev = E.FamilyEval(task, cfg, params); P = ev.params; tops = {}
    for L in ev.lens:
        _, S, al = scores_forward(P, cfg, ev.tok[L]); tops[L] = [(s >= jnp.max(jnp.where(al[None, None], s, -1e9), -1, keepdims=True) - 0.01) & al[None, None] for s in S]
    def margins(p, with_sat):
        out = []
        for L in ev.lens:
            lg, S, al = scores_forward(p, cfg, ev.tok[L]); o = lg[:, 1:]; e = ev.exp_arr[L]
            if ev.t["kind"] == "cat": corr = jnp.take_along_axis(o, e[..., None], -1)[..., 0]; oth = jnp.where(jax.nn.one_hot(e, o.shape[-1]) > 0, -1e9, o).max(-1); m = corr - oth
            else: m = ev.t["atol"] - jnp.abs(o[..., 0] - e)
            parts = [m] + ([g.reshape(m.shape[0], -1) for g in gaps(S, al, tops[L])] if with_sat else []); out.append(jnp.concatenate(parts, 1).ravel())
        return jnp.concatenate(out)
    res = {}
    for name, ws in (("v3", False), ("v4", True)):
        mj = jax.jit(lambda p, ws=ws: margins(p, ws)); m0 = np.asarray(mj(P)); seqstart = []; pos = 0
        for L in ev.lens:
            width = (m0.size and 0)  # placeholder
        # per-sequence widths
        wl = []; tot = 0
        for L in ev.lens:
            nb = len(ev.tok[L]); w = L + (cfg["n_layers"] * cfg["n_heads"] * L if ws else 0); wl.append((nb, w))
        for nb, w in wl:
            for b in range(nb): seqstart.append(pos); pos += w
        seqstart = np.array(seqstart); assert (m0 > 0).all(); rng = np.random.default_rng(0); keys = sorted(P); fp = []
        for d in range(draws):
            q = {k: P[k] + s0 * jnp.asarray(rng.standard_normal(P[k].shape).astype(np.float32)) for k in keys}; fp.append((np.asarray(mj(q)) - m0) / s0)
        fp = np.stack(fp, 1); curve = {}
        for s in E.SIG: mm = np.minimum.reduceat(m0[:, None] + s * fp, seqstart, axis=0); curve[str(s)] = [float((mm > 0).mean())]
        res[name] = E.s50(curve)
    return res
def run_family(subdir, resdir, pattern, prefix, out):
    res = json.load(open(out)) if os.path.exists(out) else {}
    for f in sorted(glob.glob(subdir + "/" + pattern)):
        base = f.split("/")[-1].rsplit(".", 1)[0]
        if base in res or "__3b__" not in base: continue
        try: r = json.load(open(f"{resdir}/{base}.json"))
        except Exception: continue
        if r.get("status") != "correct" or "noise" not in r: continue
        t = re.match(prefix + r"(\w+?)__(constrained|unconstrained)__", base).group(1)
        try: pr = family(f, t)
        except AssertionError: res[base] = dict(error="nonpositive clean margin"); continue
        res[base] = dict(task=t, meas=r["noise"]["abs"]["s50"], **pr); print(base[:46].ljust(46), f"meas {res[base]['meas']:.2e} v3 {pr['v3']:.2e} v4 {pr['v4']:.2e}", flush=True); json.dump(res, open(out, "w"), indent=1)
if __name__ == "__main__":
    if sys.argv[1] == "fresh": run_family(REPO + "/llm_synthesis/fresh_run/submissions", REPO + "/llm_synthesis/fresh_run/results", "*__none__a*", "", "v4_fresh.json")

def sim(mj, params, seqstart, draws=150, s0=1e-5):
    m0 = np.asarray(mj(params)); assert (m0 > 0).all(), f"{(m0 <= 0).sum()} non-positive clean margins"; leaves, tree = jax.tree_util.tree_flatten(params); rng = np.random.default_rng(0); fp = []
    for d in range(draws):
        q = [jnp.asarray(np.asarray(w) + s0 * rng.standard_normal(np.asarray(w).shape).astype(np.float32)) for w in leaves]; fp.append((np.asarray(mj(jax.tree_util.tree_unflatten(tree, q))) - m0) / s0)
    fp = np.stack(fp, 1); curve = {}
    for s in E.SIG if sim.grid is None else sim.grid: mm = np.minimum.reduceat(m0[:, None] + s * fp, seqstart, axis=0); curve[str(s)] = [float((mm > 0).mean())]
    return E.s50(curve)
sim.grid = None
def tracr_v4():
    import fragility as fr, tracr.compiler.assemble as AS
    class Shim:
        def __getattr__(self, k): return getattr(jnp, k)
        @staticmethod
        def argmax(x, axis=-1): return x
    meas = json.load(open("cross_tracr.json")); out = json.load(open("v4_tracr.json")) if os.path.exists("v4_tracr.json") else {}
    for name, (prog, kind, atol, samp) in fr.build_programs().items():
        if name in out: continue
        t = fr.Task(name, prog, kind, atol, samp); enc = t.model.output_encoder
        if type(enc).__name__ == "CategoricalEncoder": kind = "cat"
        tgt = {L: (jnp.array([[enc.encoding_map[v] for v in e] for e in t.exp[L]]) if kind == "cat" else jnp.array(np.array([list(e) for e in t.exp[L]], np.float32))) for L in t.enc}
        AS.jnp = Shim()
        def run(p, L, toks):
            o = t.model.forward(p, toks); return o.unembedded_output[:, 1:], o.transformer_output.attn_logits
        try: _, lg0 = run(t.params, list(t.enc)[0], t.enc[list(t.enc)[0]]); AS.jnp = jnp
        except Exception as e: AS.jnp = jnp; out[name] = dict(error=f"attn_logits unavailable: {e!r}"[:200]); print(name, out[name]); continue
        AS.jnp = Shim(); tops = {}
        for L, toks in t.enc.items():
            _, lgs = run(t.params, L, toks); tops[L] = [s >= jnp.max(s, -1, keepdims=True) - 0.01 for s in lgs]
        def margins(p, ws):
            ms = []
            for L, toks in t.enc.items():
                o, lgs = run(p, L, toks)
                if kind != "cat" and o.ndim == 3: o = o[..., 0]
                if kind == "cat": corr = jnp.take_along_axis(o, tgt[L][..., None], -1)[..., 0]; oth = jnp.where(jax.nn.one_hot(tgt[L], o.shape[-1]) > 0, -1e9, o).max(-1); m = corr - oth
                else: m = atol - jnp.abs(o - tgt[L])
                parts = [m]
                if ws:
                    for s, top in zip(lgs, tops[L]):
                        rest = ~top; g = jnp.where(rest.any(-1), jnp.min(jnp.where(top, s, 1e9), -1) - jnp.max(jnp.where(rest, s, -1e9), -1), 1e3)[:, :, 1:]; parts.append(g.reshape(m.shape[0], -1))
                ms.append(jnp.concatenate(parts, 1).ravel())
            return jnp.concatenate(ms)
        res = {}
        for nm, ws in (("v3", False), ("v4", True)):
            mj = jax.jit(lambda p, ws=ws: margins(p, ws)); seqstart = []; pos = 0
            for L, toks in t.enc.items():
                Lo = int(toks.shape[1]) - 1; n_extra = (len(tops[L]) * int(tops[L][0].shape[1]) * Lo) if ws else 0
                for b in range(len(toks)): seqstart.append(pos); pos += Lo + n_extra
            try: res[nm] = sim(mj, t.params, np.array(seqstart))
            except AssertionError as e: res[nm] = None; print(name, nm, "ERR", e)
        AS.jnp = jnp; out[name] = dict(meas=meas[name]["meas"]["abs"], **res); print(name.ljust(12), out[name], flush=True); json.dump(out, open("v4_tracr.json", "w"), indent=1)
def bilinear_v4():
    import baselines as B, fw_lib as W
    meas = json.load(open("cross_bilinear.json")); out = {}; sim.grid = B.SIGB
    for pr in W.pool("mixed", 12, 21):
        ev = B.Evaluator([pr], 40); p0 = {k: jnp.asarray(v) for k, v in ev.ind[0].items()}; ref = jnp.asarray(ev.refs[0]); K = pr["K"]; xs = jnp.swapaxes(jax.nn.one_hot(ev.Xj, W.V), 0, 1)
        def run(p):
            s = jnp.broadcast_to(p["s0"], (ev.Xj.shape[0], p["s0"].shape[0]))
            def f(s, x): z = jnp.einsum("bv,vij,bj->bi", x, p["W"], s); s2 = jnp.clip(2.0 * z - 0.5, 0.0, 1.0); return s2, (s2 @ p["T"].T, z)
            _, (y, z) = jax.lax.scan(f, s, xs); return jnp.swapaxes(y, 0, 1), jnp.swapaxes(z, 0, 1)
        _, z0 = run(p0); on = z0 > 0.5
        def margins(p, ws):
            y, z = run(p); corr = jnp.take_along_axis(y, ref[..., None], -1)[..., 0]; oth = jnp.where(jax.nn.one_hot(ref, K) > 0, -1e9, y).max(-1); m = (corr - oth)[..., None]
            if ws: m = jnp.concatenate([m, jnp.where(on, z - 0.75, 0.25 - z)], -1)
            return m.ravel()
        res = {}
        for nm, ws in (("v3", False), ("v4", True)):
            mj = jax.jit(lambda p, ws=ws: margins(p, ws)); w = ev.Xj.shape[1] * (1 + (p0["s0"].shape[0] if ws else 0)); seqstart = np.arange(ev.Xj.shape[0]) * w
            try: res[nm] = sim(mj, p0, seqstart)
            except AssertionError as e: res[nm] = None; print(pr["name"], nm, "ERR", e)
        out[pr["name"]] = dict(meas=meas[pr["name"]]["meas_abs"], **res); print(pr["name"].ljust(16), out[pr["name"]], flush=True); json.dump(out, open("v4_bilinear.json", "w"), indent=1)
if __name__ == "__main__" and sys.argv[1] == "tracr": tracr_v4()
if __name__ == "__main__" and sys.argv[1] == "bilinear": bilinear_v4()
