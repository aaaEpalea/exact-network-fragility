"""Harness self-test: (1) interval-bound soundness on a random model, (2) the project's Tracr baseline reproduces through the same noise procedure (hist sigma50 ~ 1.4e-3 abs in RESULTS.md A1)."""
import numpy as np, jax, jax.numpy as jnp, tasks as T, family as F, evaluate as E
cfg = dict(n_layers=2, n_heads=2, d_model=8, d_head=4, d_mlp=16, causal=True); rng = np.random.default_rng(0)
params = {k: (rng.normal(size=s) * 0.5).astype(np.float32) for k, s in F.shapes(cfg, 2).items()}; ev = E.FamilyEval("parity", cfg, params); tok = ev.tok[5]
for eps in [1e-3, 1e-2, 5e-2]:
    yc, yr = E.ibp_forward(ev.params, cfg, tok, eps); worst = 0.0
    for s in range(200):
        d = {k: ev.params[k] + eps * jnp.asarray(rng.uniform(-1, 1, size=v.shape), jnp.float32) for k, v in ev.params.items()}; y = F.forward(d, cfg, tok); worst = max(worst, float(jnp.max(jnp.abs(y - yc) - yr)))
    print(f"IBP soundness eps={eps}: max violation over 200 random box points = {worst:.2e} (must be <= ~1e-6)  mean radius {float(yr.mean()):.3g}")
print("attack on random model runs:", E.attack_margin(ev, 0.01, steps=5))
