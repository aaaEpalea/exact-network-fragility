"""Exhaustive domain verification for compiled/hardened categorical models."""
import itertools, numpy as np, jax, jax.numpy as jnp
from fragility import VOCAB

def domain(task_name, max_len=7):
    """Yield (L, x[int array N x L], expected class idx [N x L])."""
    for L in range(1, max_len + 1):
        if task_name == "sort_unique":
            if L > 6: continue
            x = np.array(list(itertools.permutations(range(1, 7), L)), dtype=np.int32)
        else:
            x = np.array(list(itertools.product(range(1, 7), repeat=L)), dtype=np.int32)
        if task_name == "reverse": y = x[:, ::-1] - 1
        elif task_name == "sort_unique": y = np.sort(x, 1) - 1
        elif task_name == "hist": y = (x[:, :, None] == x[:, None, :]).sum(-1)
        yield L, x, y

def verify(net, hp, task_name, chunk=20000, fwd=None):
    T = net.task; enc = T.model.input_encoder
    e = enc.encode(["BOS"] + list(range(1, 7))); bos = e[0]; ids = {v: e[v] for v in range(1, 7)}
    lut = np.zeros(7, np.int32)
    for v in range(1, 7): lut[v] = ids[v]
    fwd = fwd or jax.jit(net.predict)
    n_ok = n_tot = 0; worst = None
    for L, x, y in domain(task_name):
        toks = np.concatenate([np.full((len(x), 1), bos, np.int32), lut[x]], 1)
        for i in range(0, len(x), chunk):
            pred = np.asarray(fwd(hp, jnp.array(toks[i:i + chunk])))[:, 1:]
            ok = (pred == y[i:i + chunk]).all(1)
            n_ok += int(ok.sum()); n_tot += len(ok)
    return n_ok, n_tot
