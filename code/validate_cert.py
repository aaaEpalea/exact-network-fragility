import json, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
import digital_compiler as dc
R = json.load(open("compiler_tests.json")); rng = np.random.default_rng(0)
specs = {"majority(N=8)": dc.majority(8), "majority(N=16)": dc.majority(16), "dyck_valid_prefix(N=16)": dc.dyck(16),
         "dyck_valid_prefix(N=32)": dc.dyck(32), "parity_of_token0": dc.parity(), "count_mod3": dc.mod_count(3), "prefix_hist(N=4)": dc.prefix_hist(4)}
for name, spec in specs.items():
    eps = R[name]["certified_radius"]; p, meta = dc.compile_spec(spec)
    X = rng.choice(spec.V, size=(64, 150), p=[0.3, 0.3] + [0.4 / (spec.V - 2)] * (spec.V - 2)); exp = np.array([dc.reference(spec, list(r)) for r in X]); Xj = jnp.array(X)
    f = jax.jit(lambda P: dc.run(P, meta, Xj, spec.V)); line = f"{name:24s} eps*={eps:.1e}: "
    for mult in [1, 3, 10, 30, 100]:
        worst = 1.0
        for t in range(12):
            ks = dict(zip(p.keys(), jax.random.split(jax.random.PRNGKey(t), len(p))))
            P = {k: v + mult * eps * jnp.sign(jax.random.normal(ks[k], v.shape)) for k, v in p.items()}   # extreme corners of the eps-box
            worst = min(worst, float((np.asarray(f(P)) == exp).all(1).mean()))
        line += f"{mult}x->{worst:.2f}  "
    print(line, flush=True)
