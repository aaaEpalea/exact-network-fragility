import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))   # repository root
import sys, json, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, REPO + "/code"); sys.path.insert(0, REPO + "/llm_synthesis/harness")
import numpy as np, jax, jax.numpy as jnp
import fragility as fr, evaluate as E
import tracr.compiler.assemble as AS
class _Shim:
    def __getattr__(self, k): return getattr(jnp, k)
    @staticmethod
    def argmax(x, axis=-1): return x          # expose the logits instead of the decoded class index (the unembedding matrix itself is fixed)

MODES = ("abs", "rel", "nonzero_only", "zero_only")
def tangent(margins_fn, params, seqstart, grid, draws=150, s0=1e-5, seed=0):
    mj = jax.jit(margins_fn); m0 = np.asarray(mj(params)); leaves, tree = jax.tree_util.tree_flatten(params); res = {}
    assert (m0 > 0).all(), f"{(m0 <= 0).sum()} non-positive clean margins"
    for mode in MODES:
        rng = np.random.default_rng(seed); fp = []
        for d in range(draws):
            q = []
            for w in leaves:
                w = np.asarray(w); z = rng.standard_normal(w.shape).astype(np.float32)
                z = z * (np.abs(w) if mode == "rel" else (w != 0) if mode == "nonzero_only" else (w == 0) if mode == "zero_only" else 1.0); q.append(jnp.asarray(w + s0 * z))
            fp.append((np.asarray(mj(jax.tree_util.tree_unflatten(tree, q))) - m0) / s0)
        fp = np.stack(fp, 1); curve = {}
        for s in grid: mm = np.minimum.reduceat(m0[:, None] + s * fp, seqstart, axis=0); curve[str(s)] = [float((mm > 0).mean())]
        res[mode] = E.s50(curve)
    return res
def tracr_part():
    out = json.load(open("cross_tracr.json")) if os.path.exists("cross_tracr.json") else {}
    for name, (prog, kind, atol, samp) in fr.build_programs().items():
        if name in out: continue
        t = fr.Task(name, prog, kind, atol, samp); enc = t.model.output_encoder; tgt = {}
        if type(enc).__name__ == 'CategoricalEncoder': kind = 'cat'          # numeric values with a categorical output encoding (length, hist): argmax class must equal the value
        for L in t.enc:
            tgt[L] = jnp.array([[enc.encoding_map[v] for v in e] for e in t.exp[L]]) if kind == "cat" else jnp.array(np.array([list(e) for e in t.exp[L]], np.float32))
        def margins(p):
            ms = []
            for L, toks in t.enc.items():
                o = t.model.forward(p, toks).unembedded_output[:, 1:]
                if kind != "cat" and o.ndim == 3: o = o[..., 0]
                if kind == "cat": corr = jnp.take_along_axis(o, tgt[L][..., None], -1)[..., 0]; oth = jnp.where(jax.nn.one_hot(tgt[L], o.shape[-1]) > 0, -1e9, o).max(-1); m = corr - oth
                else: m = atol - jnp.abs(o - tgt[L])
                ms.append(m.ravel())
            return jnp.concatenate(ms)
        seqstart = []; pos = 0
        for L in t.enc:
            Lout = int(t.enc[L].shape[1]) - 1
            for b in range(len(t.enc[L])): seqstart.append(pos); pos += Lout
        class Ev:
            params = t.params; groups = []
            def acc(self, p): return t.accuracy(p)
            def group_pred(self, g, n): return False
        meas = E.noise_suite(Ev(), groups=False)
        AS.jnp = _Shim()
        try: pred = tangent(margins, t.params, np.array(seqstart), E.SIG)
        except AssertionError as e: AS.jnp = jnp; out[name] = dict(error=str(e)); print(name, 'ERR', e, flush=True); json.dump(out, open('cross_tracr.json', 'w'), indent=1); continue
        AS.jnp = jnp
        out[name] = dict(kind=kind, pred=pred, meas={k: meas[k]["s50"] for k in MODES}, n_params=int(sum(np.asarray(a).size for a in jax.tree_util.tree_leaves(t.params))))
        print(name.ljust(12), {k: f"{out[name]['meas'][k]:.2e}/{pred[k]:.2e}" for k in ("abs", "rel")}, flush=True); json.dump(out, open("cross_tracr.json", "w"), indent=1)
if __name__ == "__main__" and len(sys.argv) == 1: tracr_part()

def bilinear_part():
    import baselines as B, fw_lib as W
    out = json.load(open("cross_bilinear.json")) if os.path.exists("cross_bilinear.json") else {}
    for pr in W.pool("mixed", 12, 21):
        if pr["name"] in out: continue
        ev = B.Evaluator([pr], 40); p0 = {k: jnp.asarray(v) for k, v in ev.ind[0].items()}; ref = jnp.asarray(ev.refs[0]); K = pr["K"]
        def margins(p):
            y = B.run_bil(p, ev.Xj); corr = jnp.take_along_axis(y, ref[..., None], -1)[..., 0]; oth = jnp.where(jax.nn.one_hot(ref, K) > 0, -1e9, y).max(-1); return (corr - oth).ravel()
        seqstart = np.arange(ev.Xj.shape[0]) * ev.Xj.shape[1]
        try: pred = tangent(margins, p0, seqstart, B.SIGB)
        except AssertionError as e: out[pr["name"]] = dict(error=str(e)); continue
        meas = B.baseline_s50(ev, "indep")["per_program"]; out[pr["name"]] = dict(fam=pr["fam"], pred_abs=pred["abs"], meas_abs=meas); print(pr["name"].ljust(18), f"{meas:.3g} / {pred['abs']:.3g}", flush=True); json.dump(out, open("cross_bilinear.json", "w"), indent=1)
if len(sys.argv) > 1 and sys.argv[1] == "bilinear": bilinear_part()
