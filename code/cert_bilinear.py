"""Certified parameter-noise radius for the BILINEAR and WEIGHT-TIED designs (IBP, exact for all lengths), dense vs zeros-protected, + PGD attack validation."""
import json, time, warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings("ignore")
import digital_compiler as dc
from zeros_empirical import majority_ab, dyck_ab
from bilinear_lib import build_bil, build_tied
V = 6; DELTAS = (0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3)

def ok_bil(spec, p, eps, delta, protect):
    if eps > delta: return False
    W, T = np.asarray(p["W"], np.float64), np.asarray(p["T"], np.float64)
    for v, s, exp_s, exp_y in dc._configs(spec):
        Wv = W[v]; e = eps * (Wv != 0) if protect else eps
        zl, zu = dc._imat(Wv, s - delta, s + delta, e)
        s2l, s2u = np.clip(2 * zl - 0.5, 0, 1), np.clip(2 * zu - 0.5, 0, 1)
        if np.any(s2l < exp_s - delta) or np.any(s2u > exp_s + delta): return False
        yl, yu = dc._imat(T, s2l, s2u, eps * (T != 0) if protect else eps); ey = exp_y[0]; oth = np.delete(yu, ey)
        if oth.size and yl[ey] <= oth.max(): return False
    return True

def ok_tied(p, N, eps, delta, protect):
    if eps > delta: return False
    P, T = np.asarray(p["P"], np.float64), np.asarray(p["T"], np.float64); n = P.shape[1]
    for i0 in range(n):
        for r in range(3):
            s = np.eye(n)[i0]; nxt = [min(i0 + 1, n - 1), max(i0 - 1, 0), i0][r]; exp_s = np.eye(n)[nxt]; ey = int(nxt > N)
            g = np.eye(3)[r]; eg = eps * (g != 0) if protect else eps * np.ones(3); gl, gu = g - eg, g + eg
            zl = np.zeros(n); zu = np.zeros(n)
            for rr in range(3):
                ql, qu = dc._imat(P[rr], s - delta, s + delta, eps * (P[rr] != 0) if protect else eps)
                c = np.stack([gl[rr] * ql, gl[rr] * qu, gu[rr] * ql, gu[rr] * qu]); zl += c.min(0); zu += c.max(0)
            s2l, s2u = np.clip(2 * zl - 0.5, 0, 1), np.clip(2 * zu - 0.5, 0, 1)
            if np.any(s2l < exp_s - delta) or np.any(s2u > exp_s + delta): return False
            yl, yu = dc._imat(T, s2l, s2u, eps * (T != 0) if protect else eps); oth = np.delete(yu, ey)
            if oth.size and yl[ey] <= oth.max(): return False
    return True

def refined(ok):
    best = (0.0, None)
    for d in DELTAS:
        lo, hi = 1e-6, min(d, 0.45)
        if not ok(lo, d): continue
        if ok(hi, d): lo = hi
        else:
            for _ in range(16):
                mid = float(np.sqrt(lo * hi))
                if ok(mid, d): lo = mid
                else: hi = mid
        if lo > best[0]: best = (lo, d)
    return best

# ---------------- PGD attack (parameters + tube states) ----------------
def attack_bil(spec, p, eps, delta, protect, steps=150, restarts=4, seed=0):
    cfgs = list(dc._configs(spec)); S = jnp.array(np.stack([c[1] for c in cfgs]), jnp.float32); Xo = jnp.array(np.stack([np.eye(V)[c[0]] for c in cfgs]), jnp.float32)
    ES = jnp.array(np.stack([c[2] for c in cfgs]), jnp.float32); EY = jnp.array([c[3][0] for c in cfgs], jnp.int32)
    mW, mT = (p["W"] != 0).astype(jnp.float32), (p["T"] != 0).astype(jnp.float32)
    if not protect: mW, mT = jnp.ones_like(mW), jnp.ones_like(mT)
    def viol(dW, dT, eta):
        z = jnp.einsum("mv,vij,mj->mi", Xo, p["W"] + dW, S + eta); s2 = jnp.clip(2 * z - 0.5, 0, 1); y = s2 @ (p["T"] + dT).T
        v = jnp.abs(s2 - ES).max(-1) - delta; corr = jnp.take_along_axis(y, EY[:, None], 1)[:, 0]; oth = y.at[jnp.arange(y.shape[0]), EY].set(-1e9).max(-1)
        return jnp.maximum(v, oth - corr)
    g = jax.jit(jax.value_and_grad(lambda a, b, c: jax.nn.logsumexp(viol(a, b, c) / 0.02) * 0.02, argnums=(0, 1, 2))); vj = jax.jit(viol)
    rng = np.random.default_rng(seed); worst = -1e9
    for r in range(restarts):
        dW = eps * mW * jnp.asarray(rng.choice([-1., 1.], size=p["W"].shape), jnp.float32); dT = eps * mT * jnp.asarray(rng.choice([-1., 1.], size=p["T"].shape), jnp.float32)
        eta = delta * jnp.asarray(rng.choice([-1., 1.], size=S.shape), jnp.float32)
        for t in range(steps):
            _, (gW, gT, ge) = g(dW, dT, eta)
            dW = jnp.clip(dW + eps / 8 * jnp.sign(gW), -eps, eps) * mW; dT = jnp.clip(dT + eps / 8 * jnp.sign(gT), -eps, eps) * mT; eta = jnp.clip(eta + delta / 8 * jnp.sign(ge), -delta, delta)
        worst = max(worst, float(vj(dW, dT, eta).max()))
    return worst

def attack_tied(p, N, eps, delta, protect, steps=150, restarts=4, seed=0):
    n = p["P"].shape[1]; cf = [(i0, r) for i0 in range(n) for r in range(3)]
    S = jnp.array(np.stack([np.eye(n)[i0] for i0, r in cf]), jnp.float32); Gn = jnp.array(np.stack([np.eye(3)[r] for i0, r in cf]), jnp.float32)
    nxt = [([min(i0 + 1, n - 1), max(i0 - 1, 0), i0][r]) for i0, r in cf]; ES = jnp.array(np.stack([np.eye(n)[x] for x in nxt]), jnp.float32); EY = jnp.array([int(x > N) for x in nxt], jnp.int32)
    mP, mT, mG = (p["P"] != 0).astype(jnp.float32), (p["T"] != 0).astype(jnp.float32), (Gn != 0).astype(jnp.float32)
    if not protect: mP, mT, mG = jnp.ones_like(mP), jnp.ones_like(mT), jnp.ones_like(mG)
    def viol(dP, dT, dG, eta):
        z = jnp.einsum("mr,rij,mj->mi", Gn + dG, p["P"] + dP, S + eta); s2 = jnp.clip(2 * z - 0.5, 0, 1); y = s2 @ (p["T"] + dT).T
        v = jnp.abs(s2 - ES).max(-1) - delta; corr = jnp.take_along_axis(y, EY[:, None], 1)[:, 0]; oth = y.at[jnp.arange(y.shape[0]), EY].set(-1e9).max(-1)
        return jnp.maximum(v, oth - corr)
    g = jax.jit(jax.value_and_grad(lambda a, b, c, d: jax.nn.logsumexp(viol(a, b, c, d) / 0.02) * 0.02, argnums=(0, 1, 2, 3))); vj = jax.jit(viol)
    rng = np.random.default_rng(seed); worst = -1e9; sg = lambda shp: jnp.asarray(rng.choice([-1., 1.], size=shp), jnp.float32)
    for r in range(restarts):
        dP, dT, dG, eta = eps * mP * sg(p["P"].shape), eps * mT * sg(p["T"].shape), eps * mG * sg(Gn.shape), delta * sg(S.shape)
        for t in range(steps):
            _, (gP, gT, gG, ge) = g(dP, dT, dG, eta)
            dP = jnp.clip(dP + eps / 8 * jnp.sign(gP), -eps, eps) * mP; dT = jnp.clip(dT + eps / 8 * jnp.sign(gT), -eps, eps) * mT
            dG = jnp.clip(dG + eps / 8 * jnp.sign(gG), -eps, eps) * mG; eta = jnp.clip(eta + delta / 8 * jnp.sign(ge), -delta, delta)
        worst = max(worst, float(vj(dP, dT, dG, eta).max()))
    return worst

if __name__ == "__main__":
    res = {}; t0 = time.time()
    for fam, mk in [("majority", majority_ab), ("dyck", dyck_ab)]:
        for N in [8, 16, 32, 64]:
            spec = mk(N, 0, 1); p = build_bil(spec); D = int(p["s0"].shape[0])
            e0, d0 = refined(lambda e, d: ok_bil(spec, p, e, d, False)); e1, d1 = refined(lambda e, d: ok_bil(spec, p, e, d, True))
            res[f"bilinear|{fam}|{N}"] = {"D": D, "dense_eps": e0, "dense_tube": d0, "protected_eps": e1, "protected_tube": d1}
            print(f"BILINEAR {fam:8s} N={N:2d} D={D:3d}: certified eps*  dense={e0:.2e} (tube {d0})  zeros-protected={e1:.2e} (tube {d1})  ({time.time()-t0:.0f}s)", flush=True); json.dump(res, open("cert_bilinear.json", "w"))
    for N in [16, 32, 64]:
        p = build_tied(N, [(0, 1)]); n = 2 * N + 1
        e0, d0 = refined(lambda e, d: ok_tied(p, N, e, d, False)); e1, d1 = refined(lambda e, d: ok_tied(p, N, e, d, True))
        res[f"tied|N{N}"] = {"n": n, "dense_eps": e0, "dense_tube": d0, "protected_eps": e1, "protected_tube": d1}
        print(f"TIED (valid for ANY k) N={N:2d} n={n:3d}: certified eps*  dense={e0:.2e} (tube {d0})  zeros-protected={e1:.2e} (tube {d1})  ({time.time()-t0:.0f}s)", flush=True); json.dump(res, open("cert_bilinear.json", "w"))
    for name, runner in [("bilinear majority N=16", lambda e, d, pr: attack_bil(majority_ab(16, 0, 1), build_bil(majority_ab(16, 0, 1)), e, d, pr)),
                         ("bilinear dyck N=16", lambda e, d, pr: attack_bil(dyck_ab(16, 0, 1), build_bil(dyck_ab(16, 0, 1)), e, d, pr)),
                         ("tied N=16", lambda e, d, pr: attack_tied(build_tied(16, [(0, 1)]), 16, e, d, pr))]:
        key = {"bilinear majority N=16": "bilinear|majority|16", "bilinear dyck N=16": "bilinear|dyck|16", "tied N=16": "tied|N16"}[name]
        for protect in [False, True]:
            e, d = (res[key]["protected_eps"], res[key]["protected_tube"]) if protect else (res[key]["dense_eps"], res[key]["dense_tube"])
            row = {m: runner(m * e, d, protect) for m in [1.0, 1.1, 1.25, 1.5]}; res[f"attack|{name}|{'protected' if protect else 'dense'}"] = {str(k): v for k, v in row.items()}
            print(f"PGD attack {name} {'zeros-protected' if protect else 'dense'} (eps*={e:.2e}, tube {d}): " + "  ".join(f"{m}x:{v:+.6f}" for m, v in row.items()), flush=True); json.dump(res, open("cert_bilinear.json", "w"))
    print("DONE", flush=True)
