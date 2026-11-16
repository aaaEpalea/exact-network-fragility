"""B: error-correcting state codes for a restoring finite-state network, evaluated under a full matrix of noise conditions.
Constructions (all: linear transition memory M_v, restoring unit s' = clip(g (z - tau) + .5, 0, 1), linear readout):
   onehot   one unit per state (baseline)                                            m = n,        w = 1
   rep3/rep5 every state replicated r times, M entries 1/r (majority cleanup)         m = r n,      w = r
   willshaw random constant-weight sparse codes (w=3, pairwise overlap <= 1), binary OR-superposed memory (threshold cleanup)
   grid2    two-hot product code (row, column of a sqrt(n) x sqrt(n) grid), same memory/cleanup as willshaw (w=2)
Noise conditions: static parameter noise (abs / rel), FRESH parameter noise every step, FRESH activation noise every step, FIXED activation offset (constant per run),
   single BURST (once) vs the same magnitude REPEATED every step; sequence-length scaling of the repeated-noise failure rate; interval-bound certificates (activation tube / parameter box);
   mechanism numbers (switching slack / row norm)."""
import json, time, os, sys, numpy as np, jax, jax.numpy as jnp
import warnings; warnings.filterwarnings("ignore")
from fw_lib import V, majority_ab, dyck_ab, ref_run, s50
OUT = "../results/B_codes.json"; t0 = time.time(); SIGC = [float(x) for x in np.logspace(-3, 0.5, 14)]

def make_codes(n, m, w, rng, max_overlap=1, tries=4000):
    codes = []
    for _ in range(tries):
        c = np.zeros(m, bool); c[rng.choice(m, w, replace=False)] = True
        if all((c & d).sum() <= max_overlap for d in codes): codes.append(c)
        if len(codes) == n: return np.array(codes, np.float32)
    return None
def grid_codes(n):
    g = int(np.ceil(np.sqrt(n))); C = np.zeros((n, 2 * g), np.float32)
    for i in range(n): C[i, i // g] = 1; C[i, g + i % g] = 1
    return C
def _assemble(pr, kind, C, w, g_, tau, scale):
    n, K = pr["n"], pr["K"]; d, tb, init = pr["delta"], pr["table"], pr["init"]; m = C.shape[1]; M = np.zeros((V, m, m), np.float32); T = np.zeros((K, m), np.float32); dense = kind == "onehot" or kind.startswith("rep")
    for v in range(V):
        for j in range(n): M[v] = (M[v] + np.outer(C[d[v, j]], C[j]) * scale) if dense else np.maximum(M[v], np.outer(C[d[v, j]], C[j]) * scale)
    if dense:
        for i in range(n): T[tb[i]] = np.maximum(T[tb[i]], C[i] / w)
    else:                                                  # sparse codes: an OR-readout is ambiguous when many states share a class -> least-squares readout memory
        Ycls = np.zeros((K, n)); Ycls[tb, np.arange(n)] = 1.0; T = (Ycls @ np.linalg.pinv(C.astype(np.float64).T)).astype(np.float32)
    p = dict(M=M, T=T, s0=C[init].copy()); cfg = dict(g=g_, tau=tau, m=m, w=w, n=n, C=C); snap = lambda z: np.clip(g_ * (z - tau) + 0.5, 0, 1)
    for v in range(V):                                     # exhaustive one-step exactness (=> exact for any length, noise-free)
        for i in range(n):
            s2 = snap(M[v] @ C[i])
            if not np.array_equal(s2, C[d[v, i]]): return None
            y = T @ C[d[v, i]]; ty = tb[d[v, i]]
            if y[ty] - np.delete(y, ty).max() < 0.25: return None
    return p, cfg
def build_code(pr, kind, seed=0):
    n = pr["n"]
    if kind == "onehot": return _assemble(pr, kind, np.eye(n, dtype=np.float32), 1, 2.0, 0.5, 1.0)
    if kind.startswith("rep"): r = int(kind[3:]); return _assemble(pr, kind, np.kron(np.eye(n, dtype=np.float32), np.ones((1, r), np.float32)), r, 2.0, 0.5, 1.0 / r)
    if kind == "grid2": w = 2; return _assemble(pr, kind, grid_codes(n), w, 2.0 * w, (w - 0.5) / w, 1.0 / w)
    if kind == "willshaw":
        w = 3
        for m in [16, 24, 32, 48, 64, 96, 128, 192, 256]:
            for sd in range(3):
                C = make_codes(n, m, w, np.random.default_rng(100 * sd + m))
                if C is None: continue
                out = _assemble(pr, kind, C, w, 2.0 * w, (w - 0.5) / w, 1.0 / w)
                if out is not None: return out
        return None

def make_stream(L, B=48, seed=0):
    rng = np.random.default_rng(seed); X = rng.choice(V, size=(B, L))
    for q in range(B // 2): v = rng.integers(V); mk = rng.random(L) < 0.8; X[B // 2 + q] = np.where(mk, v, rng.integers(V, size=L))
    return X
class Runner:
    def __init__(self, pr, p, cfg, L, B=48, seed=0):
        self.X = make_stream(L, B, seed); self.ref = ref_run(pr, self.X); self.Xj = jnp.array(self.X); self.R = jnp.array(self.ref); g_, tau, m = cfg["g"], cfg["tau"], cfg["m"]; self.p = {k: jnp.asarray(v) for k, v in p.items()}; self.L = L; self.B = B
        xs = jnp.swapaxes(jax.nn.one_hot(self.Xj, V), 0, 1); tt = jnp.arange(L)
        @jax.jit
        def run(P, key, sf, sa, sx, sb, tb):
            kfix, kk = jax.random.split(key); eps_fix = sx * jax.random.normal(kfix, (B, m)); s = jnp.broadcast_to(P["s0"], (B, m)); ks = jax.random.split(kk, L)
            def f(s, inp):
                x, t, k = inp; k1, k2, k3, k4 = jax.random.split(k, 4)
                sin = s + sa * jax.random.normal(k1, s.shape) + eps_fix + (t == tb)[:, None] * sb * jax.random.normal(k2, s.shape)
                z = jnp.einsum("bv,vij,bj->bi", x, P["M"], sin) + sf * jnp.sqrt((sin ** 2).sum(-1, keepdims=True) + 1e-12) * jax.random.normal(k3, s.shape)
                s2 = jnp.clip(g_ * (z - tau) + 0.5, 0.0, 1.0); y = s2 @ P["T"].T + sf * jnp.sqrt((s2 ** 2).sum(-1, keepdims=True) + 1e-12) * jax.random.normal(k4, (B, P["T"].shape[0])); return s2, jnp.argmax(y, -1)
            _, ys = jax.lax.scan(f, s, (xs, tt, ks)); return (jnp.swapaxes(ys, 0, 1) == self.R).all(1).mean()
        self.run = run; rng = np.random.default_rng(1); self.tb_rand = jnp.array(rng.integers(2, max(3, L // 2), size=B)); self.tb_none = jnp.full((B,), -1)
    def acc(self, P=None, key=0, sf=0., sa=0., sx=0., sb=0., burst=False):
        return float(self.run(P or self.p, jax.random.PRNGKey(key), sf, sa, sx, sb, self.tb_rand if burst else self.tb_none))
    def curve(self, mode, rel=False, NS=3):
        c = {}
        for s in SIGC:
            v = []
            for i in range(NS):
                if mode == "static": 
                    P = {k: self.p[k] + s * (jnp.sqrt(jnp.mean(self.p[k] ** 2)) if rel else 1.0) * jax.random.normal(jax.random.PRNGKey(50 + 7 * i + j), self.p[k].shape) for j, k in enumerate(self.p)}; v.append(self.acc(P, i))
                elif mode == "fresh_par": v.append(self.acc(key=i, sf=s))
                elif mode == "fresh_act": v.append(self.acc(key=i, sa=s))
                elif mode == "fixed_act": v.append(self.acc(key=i, sx=s))
                elif mode == "burst_act": v.append(self.acc(key=i, sb=s, burst=True))
            c[str(s)] = v
        return c

# ---------------------------------------------------------------- interval certificates + mechanism numbers
def _snap_i(l, u, g, tau): return np.clip(g * (l - tau) + 0.5, 0, 1), np.clip(g * (u - tau) + 0.5, 0, 1)
def cert_code(pr, p, cfg, eps, delta):
    M, T, s0 = [np.asarray(p[k], np.float64) for k in ("M", "T", "s0")]; C = cfg["C"].astype(np.float64); g, tau = cfg["g"], cfg["tau"]; d, tb = pr["delta"], pr["table"]
    if np.any(np.abs(s0 - C[pr["init"]]) + eps > delta): return False
    for v in range(V):
        a, b = np.abs(M[v]), M[v]
        for i in range(pr["n"]):
            sl, su = np.clip(C[i] - delta, 0, 1), np.clip(C[i] + delta, 0, 1); sc, sr = (sl + su) / 2, (su - sl) / 2
            zc = b @ sc; zr = a @ sr + eps * (np.abs(sc) + sr).sum(); l, u = _snap_i(zc - zr, zc + zr, g, tau); ni = d[v, i]; E = C[ni]
            if np.any(l < E - delta) or np.any(u > E + delta): return False
            yc = T @ ((l + u) / 2); yr = np.abs(T) @ ((u - l) / 2) + eps * (np.abs((l + u) / 2) + (u - l) / 2).sum(); ty = tb[ni]
            if (yc - yr)[ty] <= np.delete(yc + yr, ty).max(): return False
    return True
def bis(fn, lo=1e-5, hi=0.6, it=16):
    if not fn(lo): return 0.0
    if fn(hi): return hi
    for _ in range(it):
        mid = float(np.sqrt(lo * hi)); lo, hi = (mid, hi) if fn(mid) else (lo, mid)
    return lo
def mechanism(pr, p, cfg):
    M = np.asarray(p["M"], np.float64); C = cfg["C"].astype(np.float64); g, tau = cfg["g"], cfg["tau"]; on, off = [], []; rown = 0.0
    for v in range(V):
        rown = max(rown, np.sqrt((M[v] ** 2).sum(1)).max())
        for i in range(pr["n"]):
            z = M[v] @ C[i]; E = C[pr["delta"][v, i]] > 0; on.append((z[E] - (tau + 0.5 / g)).min()); off.append(((tau - 0.5 / g) - z[~E]).min())
    return dict(slack_on=float(min(on)), slack_off=float(min(off)), row_l2=float(rown), slack_over_row=float(min(min(on), min(off)) / rown))

def main():
    res = json.load(open(OUT)) if (len(sys.argv) > 1 and sys.argv[1] == "resume" and os.path.exists(OUT)) else {}
    progs = [(fam, N, (majority_ab(N, 0, 1) if fam == "majority" else dyck_ab(N, 0, 1))) for fam in ["majority", "dyck"] for N in [8, 16, 32]]
    for fam, N, pr in progs:
        for kind in ["onehot", "rep3", "rep5", "willshaw", "grid2"]:
            key = f"{fam}|N{N}|{kind}"
            if key in res: continue
            out = build_code(pr, kind)
            if out is None: res[key] = dict(exact=False); print(f"{key:24s}: no exact code found", flush=True); json.dump(res, open(OUT, "w"), indent=1); continue
            p, cfg = out; L = 3 * N if N <= 16 else 2 * N; rn = Runner(pr, p, cfg, L); params = int(sum(v.size for v in p.values())); r = dict(exact=True, m=cfg["m"], w=cfg["w"], n=pr["n"], params=params, L=L, clean=rn.acc())
            for name, mode, rel in [("static_abs", "static", False), ("static_rel", "static", True), ("fresh_par", "fresh_par", False), ("fresh_act", "fresh_act", False), ("fixed_act", "fixed_act", False), ("burst_act", "burst_act", False)]:
                r["sigma50_" + name] = s50(rn.curve(mode, rel))
            # once vs repeated at the repeated-noise failure point ; failure rate vs sequence length at a fixed noise level
            sa = r["sigma50_fresh_act"]
            if np.isfinite(sa) and sa > 0:
                r["hazard"] = {}
                for Lh in [16, 64, 256]:
                    rh = Runner(pr, p, cfg, Lh, seed=3); sg = 0.6 * sa; a = np.mean([rh.acc(key=i, sa=sg) for i in range(4)]); r["hazard"][str(Lh)] = dict(sigma=sg, acc=float(a), per_step_error=float(1 - a ** (1.0 / Lh)))
            # certificates
            r["cert_delta_act"] = bis(lambda dl: cert_code(pr, p, cfg, 0.0, dl)); r["cert_eps_par"] = max([bis(lambda e, dl=dl: cert_code(pr, p, cfg, e, dl), hi=dl) for dl in (0.05, 0.1, 0.2, 0.3, 0.4)])
            r["mechanism"] = mechanism(pr, p, cfg); res[key] = r; json.dump(res, open(OUT, "w"), indent=1)
            print(f"{key:24s}: m={r['m']:4d} params={params:7d} | sigma50 static {r['sigma50_static_abs']:.3f} (rel {r['sigma50_static_rel']:.3f}) fresh-par {r['sigma50_fresh_par']:.3f} | act: fresh {r['sigma50_fresh_act']:.3f} fixed {r['sigma50_fixed_act']:.3f} burst {r['sigma50_burst_act']:.3f} | cert act-tube {r['cert_delta_act']:.3f} par-eps {r['cert_eps_par']:.3f} | slack/row {r['mechanism']['slack_over_row']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)
if __name__ == "__main__": main()
