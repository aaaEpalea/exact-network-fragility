"""The permitted model family: a decoder/encoder transformer with explicit parameters (no layer norm, ReLU MLP, softmax attention scaled by 1/sqrt(d_head)).
   x = embed[tok] + pos_embed[pos];  per layer:  x += sum_h softmax(q_h k_h^T / sqrt(d_head)) v_h W_o[h] + b_o ;  x += relu(x W1 + b1) W2 + b2 ;  y = x Wu + bu.
Parameters (flat dict name -> float32 array):
   embed [7,d]  pos_embed [8,d]
   l{i}.wq/wk/wv [H,d,dh]  l{i}.bq/bk/bv [H,dh]  l{i}.wo [H,dh,d]  l{i}.bo [d]  l{i}.w1 [d,dm]  l{i}.b1 [dm]  l{i}.w2 [dm,d]  l{i}.b2 [d]
   unembed.w [d,d_out]  unembed.b [d_out]
Config: n_layers, n_heads, d_model, d_head, d_mlp, causal (bool)."""
import json, importlib.util, numpy as np, jax, jax.numpy as jnp
from tasks import TASKS, MAXPOS
LIMITS = dict(n_layers=4, n_heads=4, d_model=64, d_head=32, d_mlp=256, max_params=60000)
CONSTRAINED = {   # exact architecture for the constrained-synthesis condition
    "length":    dict(n_layers=1, n_heads=1, d_model=16, d_head=8,  d_mlp=32,  causal=False),
    "hist":      dict(n_layers=1, n_heads=1, d_model=16, d_head=8,  d_mlp=32,  causal=False),
    "reverse":   dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=64,  causal=False),
    "parity":    dict(n_layers=2, n_heads=2, d_model=16, d_head=8,  d_mlp=64,  causal=True),
    "majority8": dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=128, causal=True),
    "dyck8":     dict(n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=128, causal=True),
}
def shapes(cfg, d_out):
    d, H, dh, dm = cfg["d_model"], cfg["n_heads"], cfg["d_head"], cfg["d_mlp"]; s = {"embed": (7, d), "pos_embed": (MAXPOS, d)}
    for i in range(cfg["n_layers"]):
        s.update({f"l{i}.wq": (H, d, dh), f"l{i}.wk": (H, d, dh), f"l{i}.wv": (H, d, dh), f"l{i}.bq": (H, dh), f"l{i}.bk": (H, dh), f"l{i}.bv": (H, dh), f"l{i}.wo": (H, dh, d), f"l{i}.bo": (d,),
                  f"l{i}.w1": (d, dm), f"l{i}.b1": (dm,), f"l{i}.w2": (dm, d), f"l{i}.b2": (d,)})
    s.update({"unembed.w": (d, d_out), "unembed.b": (d_out,)}); return s
def n_params(cfg, d_out): return int(sum(int(np.prod(v)) for v in shapes(cfg, d_out).values()))
def validate(task, cfg, params, condition):
    """returns list of problems (empty = valid submission)"""
    errs = []; d_out = TASKS[task]["d_out"]
    for k in ("n_layers", "n_heads", "d_model", "d_head", "d_mlp", "causal"):
        if k not in cfg: errs.append(f"config missing {k}")
    if errs: return errs
    if condition == "constrained":
        if cfg != CONSTRAINED[task]: errs.append(f"constrained condition requires config {CONSTRAINED[task]}, got {cfg}")
    else:
        for k, lim in LIMITS.items():
            if k != "max_params" and cfg[k] > lim: errs.append(f"{k}={cfg[k]} exceeds limit {lim}")
        if n_params(cfg, d_out) > LIMITS["max_params"]: errs.append(f"{n_params(cfg, d_out)} parameters exceed {LIMITS['max_params']}")
    if errs: return errs
    sh = shapes(cfg, d_out)
    for k, s in sh.items():
        if k not in params: errs.append(f"missing tensor {k} {s}")
        elif tuple(np.shape(params[k])) != s: errs.append(f"tensor {k} has shape {np.shape(params[k])}, expected {s}")
        elif not np.all(np.isfinite(params[k])): errs.append(f"tensor {k} has non-finite values")
    for k in params:
        if k not in sh: errs.append(f"unexpected tensor {k}")
    return errs
def load_submission(path):
    """.py: module with CONFIG (dict) and build_params() -> {name: array}   (category 3: code computes the weights)
       .json: {"config":{...}, "tensors":{name:{"shape":[..],"default":0.0,"entries":[[i,j,..,value],..]}}}  (category 4: weights written as literal values)"""
    if path.endswith(".json"):
        j = json.load(open(path)); cfg = j["config"]; params = {}
        for k, t in j["tensors"].items():
            a = np.full(t["shape"], t.get("default", 0.0), np.float32)
            for e in t["entries"]: a[tuple(int(i) for i in e[:-1])] = float(e[-1])
            params[k] = a
        return cfg, params
    spec = importlib.util.spec_from_file_location("submission", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return dict(m.CONFIG), {k: np.asarray(v, np.float32) for k, v in m.build_params().items()}
def forward(params, cfg, tokens):
    """tokens (B,T) int -> (B,T,d_out)"""
    T = tokens.shape[1]; x = params["embed"][tokens] + params["pos_embed"][:T][None]; dh = cfg["d_head"]; mask = jnp.tril(jnp.ones((T, T), bool)) if cfg["causal"] else jnp.ones((T, T), bool)
    for i in range(cfg["n_layers"]):
        p = lambda n: params[f"l{i}.{n}"]
        q = jnp.einsum("btd,hde->bhte", x, p("wq")) + p("bq")[None, :, None]; k = jnp.einsum("btd,hde->bhte", x, p("wk")) + p("bk")[None, :, None]; v = jnp.einsum("btd,hde->bhte", x, p("wv")) + p("bv")[None, :, None]
        s = jnp.einsum("bhte,bhse->bhts", q, k) / jnp.sqrt(dh * 1.0); s = jnp.where(mask[None, None], s, -1e9); a = jax.nn.softmax(s, -1); o = jnp.einsum("bhts,bhse->bhte", a, v)
        x = x + jnp.einsum("bhte,hed->btd", o, p("wo")) + p("bo"); x = x + jax.nn.relu(x @ p("w1") + p("b1")) @ p("w2") + p("b2")
    return x @ params["unembed.w"] + params["unembed.b"]
GROUPS = {"embeddings": lambda n: n in ("embed", "pos_embed"), "attn_query/key": lambda n: any(n.endswith(s) for s in (".wq", ".bq", ".wk", ".bk")),
          "attn_value/out": lambda n: any(n.endswith(s) for s in (".wv", ".bv", ".wo", ".bo")), "mlp_in": lambda n: n.endswith((".w1", ".b1")), "mlp_out": lambda n: n.endswith((".w2", ".b2")), "unembed": lambda n: n.startswith("unembed")}
