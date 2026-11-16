"""Kink test: relu'(0) = 1 (the sensitive one-sided derivative) instead of autodiff's 0.  Does the first-order L-inf radius then match the attack radius for the 'cliff' networks?"""
import sys, glob, json, re, numpy as np, jax, jax.numpy as jnp
@jax.custom_jvp
def relu1(x): return jnp.maximum(x, 0.0)
@relu1.defjvp
def _(p, t): (x,), (dx,) = p, t; return jnp.maximum(x, 0.0), jnp.where(x >= 0, 1.0, 0.0) * dx
jax.nn.relu = relu1
import cliff
D = cliff.D; out = {}
for f in sorted(glob.glob(D + "/submissions/contam__*__3b__none__a1.py")):
    base = f.split("/")[-1][:-3]; t = re.match(r"contam__(\w+?)__(constrained|unconstrained)__", base).group(1); r = json.load(open(f"{D}/results/{base}.json"))
    cfg, ev, names, unflat, flat0, margins, grp = cliff.setup(f, t); m0 = np.asarray(margins(flat0)); G = np.asarray(jax.jacrev(margins)(flat0)); n1 = np.abs(G).sum(1) + 1e-30; n2 = np.sqrt((G ** 2).sum(1)) + 1e-30
    out[base] = dict(rho_lin_kink=float((m0 / n1).min()), sigma_lin_kink=float((m0 / n2).min()), atk=r["attack_first_break"], s50=r["noise"]["abs"]["s50"], cert=r["cert_radius_sample"])
    o = out[base]; print(f"{base[8:44]:36s} rho_lin(kink)/atk = {o['rho_lin_kink']/o['atk']:7.2f} | sigma_lin(kink)/sigma50 = {o['sigma_lin_kink']/o['s50']:6.2f}", flush=True); json.dump(out, open("kink_dev.json", "w"), indent=1)
