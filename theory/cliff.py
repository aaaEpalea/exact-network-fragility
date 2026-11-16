import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))   # repository root
import sys, glob, json, re, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, REPO + "/llm_synthesis/harness")
import tasks as T, family as F, evaluate as E
D = REPO + "/llm_synthesis"
def tapped(params, cfg, tokens):
    """residual stream after each sub-block: [embed, attn0, mlp0, attn1, mlp1, ...] and outputs"""
    T_ = tokens.shape[1]; x = params["embed"][tokens] + params["pos_embed"][:T_][None]; dh = cfg["d_head"]; mask = jnp.tril(jnp.ones((T_, T_), bool)) if cfg["causal"] else jnp.ones((T_, T_), bool); taps = [x]
    for i in range(cfg["n_layers"]):
        p = lambda n: params[f"l{i}.{n}"]
        q = jnp.einsum("btd,hde->bhte", x, p("wq")) + p("bq")[None, :, None]; k = jnp.einsum("btd,hde->bhte", x, p("wk")) + p("bk")[None, :, None]; v = jnp.einsum("btd,hde->bhte", x, p("wv")) + p("bv")[None, :, None]
        s = jnp.einsum("bhte,bhse->bhts", q, k) / jnp.sqrt(dh * 1.0); s = jnp.where(mask[None, None], s, -1e9); a = jax.nn.softmax(s, -1); o = jnp.einsum("bhts,bhse->bhte", a, v)
        x = x + jnp.einsum("bhte,hed->btd", o, p("wo")) + p("bo"); taps.append(x); x = x + jax.nn.relu(x @ p("w1") + p("b1")) @ p("w2") + p("b2"); taps.append(x)
    return taps, x @ params["unembed.w"] + params["unembed.b"]
def setup(sub, task):
    cfg, params = F.load_submission(sub); ev = E.FamilyEval(task, cfg, params); names = sorted(ev.params); shapes = {k: ev.params[k].shape for k in names}; sizes = [int(np.prod(shapes[k])) for k in names]; off = np.concatenate([[0], np.cumsum(sizes)])
    flat0 = jnp.concatenate([ev.params[k].ravel() for k in names]); unflat = lambda v: {k: v[off[i]:off[i + 1]].reshape(shapes[k]) for i, k in enumerate(names)}
    def margins(v):
        p = unflat(v); out = []
        for L in ev.lens:
            o = F.forward(p, cfg, ev.tok[L])[:, 1:]; e = ev.exp_arr[L]
            if ev.t["kind"] == "cat": corr = jnp.take_along_axis(o, e[..., None], -1)[..., 0]; oth = jnp.where(jax.nn.one_hot(e, o.shape[-1]) > 0, -1e9, o).max(-1); m = corr - oth
            else: m = ev.t["atol"] - jnp.abs(o[..., 0] - e)
            out.append(m.ravel())
        return jnp.concatenate(out)
    grp = {g: np.concatenate([np.full(sizes[i], F.GROUPS[g](k)) for i, k in enumerate(names)]).astype(bool) for g in F.GROUPS}
    return cfg, ev, names, unflat, flat0, margins, grp
def attack_dir(margins, flat0, eps, steps=30, seed=0):
    rng = np.random.default_rng(seed); obj = lambda d: -0.05 * jax.nn.logsumexp(-margins(flat0 + d) / 0.05); g = jax.jit(jax.value_and_grad(obj)); mj = jax.jit(lambda d: margins(flat0 + d).min())
    d = eps * jnp.asarray(rng.choice([-1.0, 1.0], size=flat0.shape), jnp.float32)
    for _ in range(steps): _, gr = g(d); d = jnp.clip(d - eps / 4 * jnp.sign(jnp.nan_to_num(gr)), -eps, eps)
    return d, float(mj(d))
def analyze(sub, task, atk):
    cfg, ev, names, unflat, flat0, margins, grp = setup(sub, task); m0 = np.asarray(margins(flat0)); G = np.asarray(jax.jacrev(margins)(flat0)); eps = 1.3 * atk
    for _ in range(6):
        d, mm = attack_dir(margins, flat0, eps)
        if mm <= 0: break
        eps *= 1.5
    ts = np.linspace(0, 1.6, 33); act = np.array([np.asarray(margins(flat0 + t * d)) for t in ts]); lin = m0[None] + ts[:, None] * (G @ np.asarray(d))[None]
    fa, fl = act.min(1), lin.min(1); tx = ts[np.argmax(fa <= 0)] if (fa <= 0).any() else None; tl = ts[np.argmax(fl <= 0)] if (fl <= 0).any() else None
    half = float(np.interp(0.5 * tx, ts, fa) / fa[0]) if tx else None
    # attribution: remove / keep only each group in the failing perturbation
    base = float(margins(flat0 + d).min()); attr = {}
    for g, mask in grp.items():
        mk = jnp.asarray(mask.astype(np.float32)); attr[g] = dict(remove_min_margin=float(margins(flat0 + d * (1 - mk)).min()), only_min_margin=float(margins(flat0 + d * mk).min()))
    # block gains on the worst sample (longest length block containing argmin)
    pos = int(np.argmin(np.asarray(margins(flat0 + d)))); tot = 0; gains = None
    for L in ev.lens:
        n_ = len(ev.tok[L]) * L
        if pos < tot + n_:
            b = (pos - tot) // L; tok = ev.tok[L][b:b + 1]; t0, _ = tapped(unflat(flat0), cfg, tok); t1, _ = tapped(unflat(flat0 + d), cfg, tok); dev = [float(jnp.linalg.norm(a - c)) for a, c in zip(t0, t1)]
            gains = dict(deviation=dev, blocks=["embed"] + sum([[f"attn{i}", f"mlp{i}"] for i in range(cfg["n_layers"])], []), gain=[dev[i + 1] / max(dev[i], 1e-12) for i in range(len(dev) - 1)]); break
        tot += n_
    return dict(eps=eps, fail_margin=base, t_x=tx, t_lin=tl, half_ratio=half, margin0=float(m0.min()), rho_lin=float((m0 / (np.abs(G).sum(1) + 1e-30)).min()), attr=attr, blocks=gains)
if __name__ == "__main__":
    out = {}
    for f in sorted(glob.glob(D + "/submissions/contam__*__3b__none__a1.py")):
        base = f.split("/")[-1][:-3]; t = re.match(r"contam__(\w+?)__(constrained|unconstrained)__", base).group(1); r = json.load(open(f"{D}/results/{base}.json")); out[base] = analyze(f, t, r["attack_first_break"]); o = out[base]
        print(base[8:], f"t_x={o['t_x']} t_lin={o['t_lin']} half_ratio={o['half_ratio']} rho_lin/atk={o['rho_lin']/r['attack_first_break']:.1f}", flush=True); json.dump(out, open("cliff_dev.json", "w"), indent=1)
