"""Fragility harness for Tracr-compiled transformers.

Compiles small RASP programs, evaluates exact-correctness, and measures how
accuracy degrades under weight perturbations (Gaussian noise, relative noise,
quantization, dtype casts, per-component noise).
"""
import itertools, json, sys, time, warnings
import numpy as np
import jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
from tracr.rasp import rasp
from tracr.compiler import compiling, lib

VOCAB = {1, 2, 3, 4, 5, 6}
MAX_LEN = 8
BOS = "BOS"
rng0 = np.random.default_rng(0)


def build_programs():
    """name -> (rasp program, kind, atol, input sampler type)."""
    progs = {}
    progs["reverse"] = (lib.make_reverse(rasp.tokens), "cat", None, "free")
    progs["sort_unique"] = (lib.make_sort_unique(rasp.tokens, rasp.tokens), "cat", None, "unique")
    progs["hist"] = (lib.make_hist(), "num", 0.5, "free")
    progs["length"] = (lib.make_length(), "num", 0.5, "free")
    progs["frac_prevs"] = (lib.make_frac_prevs(rasp.tokens == 1), "num", 0.05, "ones")
    progs["sort_freq"] = (lib.make_sort_freq(MAX_LEN), "cat", None, "free")
    return progs


def sample_inputs(kind, n_per_len=40, lens=(3, 4, 5, 6, 7), seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for L in lens:
        xs = []
        for _ in range(n_per_len):
            if kind == "ones":
                while True:
                    x = list(rng.choice(sorted(VOCAB), size=L, p=[0.4, 0.12, 0.12, 0.12, 0.12, 0.12]))
                    if 1 in x[:-1]:
                        break
                xs.append(x)
            elif kind == "unique":
                xs.append(list(rng.choice(sorted(VOCAB), size=min(L, len(VOCAB)), replace=False)))
            else:
                xs.append(list(rng.choice(sorted(VOCAB), size=L)))
        out[L] = [[int(v) for v in x] for x in xs]
    return out


class Task:
    """A compiled program + cached eval data + a jitted batched forward."""

    def __init__(self, name, prog, kind, atol, sampler):
        self.name, self.prog, self.kind, self.atol = name, prog, kind, atol
        self.model = compiling.compile_rasp_to_model(
            prog, vocab=VOCAB, max_seq_len=MAX_LEN, compiler_bos=BOS)
        m = self.model
        self.params = m.params
        self.inputs = sample_inputs(sampler)
        self.enc, self.exp = {}, {}
        for L, xs in self.inputs.items():
            self.enc[L] = jnp.array([m.input_encoder.encode([BOS] + x) for x in xs])
            self.exp[L] = [prog(x) for x in xs]
        self._fwd = jax.jit(lambda p, t: m.forward(p, t).unembedded_output)

    def _decode_row(self, row):
        row = np.asarray(row).tolist()
        if self.model.output_encoder is not None:
            row = self.model.output_encoder.decode(row)
        return row[1:]  # drop BOS position

    def accuracy(self, params):
        """Fraction of sequences with every output position correct."""
        ok, tot = 0, 0
        for L, toks in self.enc.items():
            outs = np.asarray(self._fwd(params, toks))
            for row, exp in zip(outs, self.exp[L]):
                got = self._decode_row(row)
                if self.kind == "cat":
                    good = list(got) == list(exp)
                else:
                    try:
                        good = bool(np.allclose(np.asarray(got, dtype=float),
                                                np.asarray(exp, dtype=float), atol=self.atol))
                    except Exception:
                        good = False
                ok += good; tot += 1
        return ok / tot


# ---- parameter perturbations --------------------------------------------
def flat_items(params):
    for mod, d in params.items():
        for n, a in d.items():
            yield mod, n, a


def map_params(params, fn):
    return {mod: {n: fn(mod, n, jnp.asarray(a)) for n, a in d.items()} for mod, d in params.items()}


def add_gaussian(params, sigma, key):
    keys = iter(jax.random.split(key, 1000))
    return map_params(params, lambda m, n, a: a + sigma * jax.random.normal(next(keys), a.shape))


def add_relative(params, sigma, key):
    """Multiplicative noise on nonzero entries only: w * (1 + sigma*eps)."""
    keys = iter(jax.random.split(key, 1000))
    return map_params(params, lambda m, n, a: a * (1 + sigma * jax.random.normal(next(keys), a.shape)))


def quantize(params, bits):
    def q(m, n, a):
        mx = jnp.max(jnp.abs(a))
        if mx == 0:
            return a
        levels = 2 ** (bits - 1) - 1
        scale = mx / levels
        return jnp.round(a / scale) * scale
    return map_params(params, q)


def cast_dtype(params, dtype):
    return map_params(params, lambda m, n, a: a.astype(dtype).astype(jnp.float32))


GROUPS = {
    "embeddings": lambda m, n: "embed" in m,
    "attn_query/key": lambda m, n: ("attn/query" in m or "attn/key" in m),
    "attn_value/out": lambda m, n: ("attn/value" in m or "attn/linear" in m),
    "mlp_in": lambda m, n: "mlp/linear_1" in m,
    "mlp_out": lambda m, n: "mlp/linear_2" in m,
}


def add_gaussian_group(params, sigma, key, group):
    keys = iter(jax.random.split(key, 1000))
    sel = GROUPS[group]
    return map_params(params, lambda m, n, a: a + sigma * jax.random.normal(next(keys), a.shape) if sel(m, n) else a)


def add_gaussian_masked(params, sigma, key, where):
    """Additive noise only on entries that are 'zero' or 'nonzero' in the compiled weights."""
    keys = iter(jax.random.split(key, 1000))
    def f(m, n, a):
        noise = sigma * jax.random.normal(next(keys), a.shape)
        mask = (a == 0) if where == "zero" else (a != 0)
        return a + noise * mask
    return map_params(params, f)
