"""Hardening transforms for Tracr-compiled models.

V1: scale attention query weights by c (attention temperature / margin knob).
V2: insert 'snap' (restoring) MLP layers that clean one-hot feature dims after the
    embedding and after every attention / MLP sublayer. Snap = clip(x, 0, 1) via ReLUs,
    exact on {0,1}, restores deviations < 0.25.
Both are exact-preserving on clean inputs (verified empirically in run_hardening.py).
"""
import numpy as np
import jax, jax.numpy as jnp


class CompiledNet:
    """Re-implementation of the compiled forward pass operating on a params dict
    (so we can insert extra layers). Output = argmax over output dims (categorical)
    or the raw output dim (numerical)."""

    def __init__(self, task):
        m = task.model
        self.task, self.m = task, m
        self.L = m.model_config.num_layers
        self.H, self.K = m.model_config.num_heads, m.model_config.key_size
        labels = m.residual_labels
        self.labels = labels
        self.binary_dims = [i for i, l in enumerate(labels) if ":" in l]
        prog_label = task.prog.label
        self.out_dims = [i for i, l in enumerate(labels) if l.startswith(prog_label + ":")]
        self.out_numeric = len(self.out_dims) == 0
        if self.out_numeric:  # numerical output lives in a single dim named by program label
            self.out_dims = [i for i, l in enumerate(labels) if l == prog_label]
        self.d = len(labels)

    # ---- snap params ------------------------------------------------------
    def make_snap(self, lo=0.25, hi=0.75):
        """Return snap-layer params restoring all binary dims."""
        n, d = len(self.binary_dims), self.d
        W1 = np.zeros((d, 4 * n), np.float32); b1 = np.zeros(4 * n, np.float32)
        W2 = np.zeros((4 * n, d), np.float32)
        s = 1.0 / (hi - lo)
        for k, j in enumerate(self.binary_dims):
            a = 4 * k
            W1[j, a] = s;      b1[a] = -s * lo        # relu(s(x-lo))
            W1[j, a + 1] = s;  b1[a + 1] = -s * hi    # relu(s(x-hi))
            W1[j, a + 2] = 1.0                        # relu(x)
            W1[j, a + 3] = -1.0                       # relu(-x)
            W2[a, j], W2[a + 1, j], W2[a + 2, j], W2[a + 3, j] = 1.0, -1.0, -1.0, 1.0   # clip(x)-x
        return {"w1": W1, "b1": b1, "w2": W2}

    def harden(self, params, c=1.0, snap=False, snap_sites="all", lo=0.25, hi=0.75, balance=False):
        """Return a new params pytree {'base': ..., 'snaps': [...]} with attention scale c and optional snaps."""
        base = {mod: {n: np.asarray(a).copy() for n, a in d.items()} for mod, d in params.items()}
        if balance:   # exact rescaling symmetries: q.k invariant under (Wq/s, Wk*s); relu(W1x/s)*s*W2/... invariant under (W1/s, b1/s, W2*s)
            for l in range(self.L):
                pre = f"transformer/layer_{l}/"
                q, k = base[pre + "attn/query"], base[pre + "attn/key"]
                mq, mk = np.abs(q["w"]).max(), np.abs(k["w"]).max()
                if mq > 0 and mk > 0:
                    sc = np.sqrt(mq / mk)
                    q["w"], q["b"], k["w"], k["b"] = q["w"] / sc, q["b"] / sc, k["w"] * sc, k["b"] * sc
                m1, m2 = base[pre + "mlp/linear_1"], base[pre + "mlp/linear_2"]
                a1, a2 = np.abs(m1["w"]).max(), np.abs(m2["w"]).max()
                if a1 > 0 and a2 > 0:
                    sc = np.sqrt(a1 / a2)
                    m1["w"], m1["b"], m2["w"] = m1["w"] / sc, m1["b"] / sc, m2["w"] * sc
        if c != 1.0:
            for mod in base:
                if mod.endswith("attn/query"):
                    base[mod]["w"] = base[mod]["w"] * c
        sites = []
        if snap:
            n_sites = 1 + 2 * self.L           # after embed, then after each attn and each mlp
            sites = [self.make_snap(lo, hi) for _ in range(n_sites)] if snap_sites == "all" else \
                    [self.make_snap(lo, hi) for _ in range(1 + self.L)]   # 'layer': embed + after each layer
        return {"base": base, "snaps": sites}

    # ---- forward ----------------------------------------------------------
    def forward(self, hp, toks, ideal=False):
        P, snaps = hp["base"], hp["snaps"]
        mode = "all" if len(snaps) == 1 + 2 * self.L else "layer"
        B, T = toks.shape
        x = P["token_embed"]["embeddings"][toks] + P["pos_embed"]["embeddings"][jnp.arange(T)][None]
        si = 0
        bd = jnp.array(self.binary_dims)
        def snap(x):
            nonlocal si
            if ideal:   # weight-free clip on binary dims (upper bound on what restoring could do)
                return x.at[..., bd].set(jnp.clip(x[..., bd], 0.0, 1.0))
            if si < len(snaps):
                s = snaps[si]; si += 1
                x = x + jax.nn.relu(x @ s["w1"] + s["b1"]) @ s["w2"]
            return x
        x = snap(x)
        for l in range(self.L):
            g = lambda part, k: P[f"transformer/layer_{l}/{part}"][k]
            q = (x @ g("attn/query", "w") + g("attn/query", "b")).reshape(B, T, self.H, self.K)
            k = (x @ g("attn/key", "w") + g("attn/key", "b")).reshape(B, T, self.H, self.K)
            v = (x @ g("attn/value", "w") + g("attn/value", "b")).reshape(B, T, self.H, self.K)
            a = jax.nn.softmax(jnp.einsum("bthd,bThd->bhtT", q, k) / np.sqrt(self.K), -1)
            o = jnp.einsum("bhtT,bThd->bthd", a, v).reshape(B, T, -1)
            x = x + o @ g("attn/linear", "w") + g("attn/linear", "b")
            if mode == "all": x = snap(x)
            h = jax.nn.relu(x @ g("mlp/linear_1", "w") + g("mlp/linear_1", "b"))
            x = x + h @ g("mlp/linear_2", "w") + g("mlp/linear_2", "b")
            x = snap(x)
        out = x[..., self.out_dims]
        return out

    def predict(self, hp, toks, ideal=False):
        out = self.forward(hp, toks, ideal)
        return out.squeeze(-1) if self.out_numeric else jnp.argmax(out, -1)


def accuracy_hp(net, hp, jit_fwd=None):
    """Exact-sequence accuracy for the hardened/perturbed params, using the Task's eval set."""
    T = net.task
    fwd = jit_fwd or jax.jit(net.predict)
    ok = tot = 0
    for L, toks in T.enc.items():
        outs = np.asarray(fwd(hp, toks))
        for row, exp in zip(outs, T.exp[L]):
            if net.out_numeric:
                got = row[1:]
                good = bool(np.allclose(np.asarray(got, float), np.asarray(exp, float), atol=T.atol))
            else:
                got = T.model.output_encoder.decode(row.tolist())[1:]
                good = list(got) == list(exp)
            ok += good; tot += 1
    return ok / tot


def perturb(hp, sigma, key, mode="abs"):
    """Add Gaussian noise to every array in the hardened params (base + snaps)."""
    leaves, tree = jax.tree_util.tree_flatten({"base": hp["base"], "snaps": hp["snaps"]})
    keys = jax.random.split(key, len(leaves)); out = []
    for a, k in zip(leaves, keys):
        a = jnp.asarray(a); n = jax.random.normal(k, a.shape)
        if mode == "abs": out.append(a + sigma * n)
        elif mode == "nz": out.append(a + sigma * n * (a != 0))
        else: out.append(a + sigma * jnp.sqrt(jnp.mean(a ** 2)) * n)
    d = jax.tree_util.tree_unflatten(tree, out)
    return {"base": d["base"], "snaps": d["snaps"]}
