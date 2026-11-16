"""Try to FALSIFY the IBP noise certificate with a gradient (PGD) attack on parameters and tube states."""
import json, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
import digital_compiler as dc
R = json.load(open("compiler_tests.json"))
specs = {"majority(N=8)": dc.majority(8), "majority(N=16)": dc.majority(16), "dyck_valid_prefix(N=16)": dc.dyck(16),
         "dyck_valid_prefix(N=32)": dc.dyck(32), "parity_of_token0": dc.parity(), "count_mod3": dc.mod_count(3)}
PERT = ["A", "B", "b", "C", "S1", "bs1", "S2", "bs2", "C3", "C4", "T0", "T1", "T2", "T3", "I3", "Bt", "b3", "R"]
def attack(spec, eps, delta, steps=150, restarts=4, seed=0):
    p, meta = dc.compile_spec(spec); V = spec.V
    cfgs = list(dc._configs(spec)); S = jnp.array(np.stack([c[1] for c in cfgs]), jnp.float32); Xo = jnp.array(np.stack([np.eye(V)[c[0]] for c in cfgs]), jnp.float32)
    ES = jnp.array(np.stack([c[2] for c in cfgs]), jnp.float32); EY = jnp.array(np.array([c[3] for c in cfgs]), jnp.int32)
    keys = [k for k in p if k in PERT]
    def viol(delta_p, eta):
        q = {k: (p[k] + delta_p[k] if k in delta_p else p[k]) for k in p}
        s2, ys = dc.step(q, meta, S + eta, Xo)
        v = jnp.abs(s2 - ES).max(-1) - delta                                 # tube-invariance violation
        for h, y in enumerate(ys):
            correct = jnp.take_along_axis(y, EY[:, h:h + 1] if len(ys) > 1 or meta["type"] != "selected" else EY[:, :1], 1)[:, 0]
            masked = y.at[jnp.arange(y.shape[0]), EY[:, h] if len(ys) > 1 or meta["type"] != "selected" else EY[:, 0]].set(-1e9)
            v = jnp.maximum(v, masked.max(-1) - correct)                      # readout violation (>= 0 means wrong/tie)
        return v
    obj = lambda dp_, eta: jax.nn.logsumexp(viol(dp_, eta) / 0.02) * 0.02
    g = jax.jit(jax.value_and_grad(obj, argnums=(0, 1))); vj = jax.jit(viol)
    rng = np.random.default_rng(seed); worst = -1e9
    for r in range(restarts):
        dp_ = {k: eps * jnp.asarray(rng.choice([-1.0, 1.0], size=p[k].shape), jnp.float32) for k in keys}
        eta = delta * jnp.asarray(rng.choice([-1.0, 1.0], size=S.shape), jnp.float32)
        for t in range(steps):
            _, (gp, ge) = g(dp_, eta)
            dp_ = {k: jnp.clip(dp_[k] + (eps / 8) * jnp.sign(gp[k]), -eps, eps) for k in keys}; eta = jnp.clip(eta + (delta / 8) * jnp.sign(ge), -delta, delta)
        worst = max(worst, float(vj(dp_, eta).max()))
    return worst                                                              # > 0 means the attack BROKE the guarantee
out = {}
for name, spec in specs.items():
    eps, delta = R[name]["certified_radius"], R[name]["tube"]; line = f"{name:24s} certified eps*={eps:.1e} (tube {delta}): "
    first_break = None
    for mult in [1, 2, 4, 8, 16, 32, 64]:
        w = attack(spec, mult * eps, delta)
        line += f"{mult}x:{'BREAK' if w > 0 else 'ok'}({w:+.2f}) "
        if w > 0 and first_break is None: first_break = mult
    out[name] = {"eps_cert": eps, "first_break_multiple": first_break}; print(line, flush=True)
    json.dump(out, open("cert_attack.json", "w"))
print("DONE", flush=True)
