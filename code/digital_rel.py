"""Digital prefix-hist under the SAME relative protocol as trained/compiled transformers: W + alpha*rms(W)*noise per tensor."""
import json, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
from scaling import s50
from digital_prefix_hist import build_digital, run_digital, prefix_hist, V, BATCH
ALPH = [float(x) for x in np.logspace(-3, 0.3, 15)]; rng = np.random.default_rng(0); res = {}
for N in [8, 16, 32, 64]:
    p = build_digital(N); L = N - 1
    X = np.stack([rng.integers(0, V, size=L) for _ in range(BATCH)]); exp = np.array([prefix_hist([int(t) + 1 for t in r]) for r in X])
    f = jax.jit(lambda P, X: run_digital(P, X, True))
    def noisy(P, a, key):
        ks = dict(zip(P.keys(), jax.random.split(key, len(P))))
        return {k: v + a * jnp.sqrt(jnp.mean(v ** 2)) * jax.random.normal(ks[k], v.shape) for k, v in P.items()}
    curve = {str(a): [float((np.asarray(f(noisy(p, a, jax.random.PRNGKey(k)), jnp.array(X))) == exp).all(1).mean()) for k in range(4)] for a in ALPH}
    res[N] = s50(curve); print(f"digital prefix-hist N={N:3d}  alpha50 (relative, RMS protocol) = {res[N]:.2e}", flush=True)
json.dump(res, open("digital_rel.json", "w"))
