"""Fast-weight universal interpreter (FWI) library.

One shared interpreter (slow weights theta) executes DIFFERENT finite-state programs whose transition tables live only in fast weights phi.
   key   k(x,s,g) = (A x) o (B s) o (C g) / sqrt(dk)      (o = Hadamard product: multiplicative binding of token, state, program address)
   step  s' = clip(a * F k - b, 0, 1)                      (F = fast transition memory; clip = restoring snap, a=2, b=.5)
   read  y  = R r(s',g),  r = (B s') o (C g)/sqrt(dk)      (R = fast readout memory)
   init  s0 = clip(a * S0 (C g)/sqrt(dk) - b, 0, 1)        (S0 = fast init memory)
theta = {A,B,C,a,b}  (slow; identical for every program)      phi = {F,R,S0} (fast; written from a program tape by a delta rule)
"""
import numpy as np, jax, jax.numpy as jnp
V = 6

# ------------------------------------------------------------------ programs (genuinely different transition structures)
def _prog(delta, table, init, K, name, fam): return dict(delta=np.array(delta, int), table=np.array(table, int), init=int(init), K=int(K), n=delta.shape[1] if hasattr(delta, "shape") else len(delta[0]), name=name, fam=fam)
def sat(n, i, d): return int(min(max(i + d, 0), n - 1))
def majority_ab(N, a, b):
    n = 2 * N + 1; d = np.array([[sat(n, i, 1) if t == a else sat(n, i, -1) if t == b else i for i in range(n)] for t in range(V)])
    return _prog(d, [int(i > N) for i in range(n)], N, 2, f"maj{N}({a}>{b})", "majority")
def dyck_ab(N, a, b):
    n = N + 2; dead = N + 1
    d = np.array([[dead if i == dead else (min(i + 1, N) if t == a else (i - 1 if i > 0 else dead) if t == b else i) for i in range(n)] for t in range(V)])
    return _prog(d, [1] * (N + 1) + [0], 0, 2, f"dyck{N}({a},{b})", "dyck")
def mod_counter(m, a):
    d = np.array([[(i + 1) % m if t == a else i for i in range(m)] for t in range(V)]); return _prog(d, list(range(m)), 0, m, f"mod{m}({a})", "modcount")
def reset_counter(n, a, b, thr):
    d = np.array([[sat(n, i, 1) if t == a else 0 if t == b else i for i in range(n)] for t in range(V)]); return _prog(d, [int(i >= thr) for i in range(n)], 0, 2, f"reset{n}({a},{b})", "reset")
def parity_set(a, b):
    d = np.array([[1 - i if t == a else i for i in range(2)] if t != b else [0, 0] for t in range(V)]); return _prog(d, [0, 1], 0, 2, f"parity({a};rst{b})", "parity")
def random_fsm(n, K, seed):
    r = np.random.default_rng(seed); d = r.integers(0, n, size=(V, n)); tb = r.integers(0, K, size=n); tb[:K] = np.arange(K); return _prog(d, tb, 0, K, f"rand{n}_{seed}", "random")

PAIRS = [(a, b) for a in range(V) for b in range(V) if a != b]
def pool(kind, P, nmax=21, seed=0):
    """Program sets of increasing structural diversity."""
    N = (nmax - 1) // 2; out = []
    if kind == "template":                      # D0: same transition structure, only the token roles differ (what existing weight-tied design can share)
        out = [majority_ab(N, *PAIRS[i % len(PAIRS)]) for i in range(P)]
    elif kind == "family":                      # D1: two families (counter with saturation / bracket automaton), varied sizes and tokens
        for i in range(P):
            Ni = [N, N - 2, N - 4, N - 6][(i // 2) % 4]; a, b = PAIRS[(i * 7) % len(PAIRS)]
            out.append(majority_ab(Ni, a, b) if i % 2 == 0 else dyck_ab(min(Ni * 2, nmax - 2), a, b))
    elif kind == "mixed":                       # D2: six structurally different families
        for i in range(P):
            a, b = PAIRS[(i * 7 + 3) % len(PAIRS)]; f = i % 6; s = i // 6
            if f == 0: out.append(majority_ab(N - s % 4, a, b))
            elif f == 1: out.append(dyck_ab(nmax - 2 - 2 * (s % 3), a, b))
            elif f == 2: out.append(mod_counter(5 + (s * 3) % (nmax - 5), a))
            elif f == 3: out.append(reset_counter(nmax - (s % 4), a, b, 3 + s % 5))
            elif f == 4: out.append(parity_set(a, b))
            else: out.append(random_fsm(nmax - (s % 3), 4, 1000 + 17 * i + seed))
    elif kind == "random":                      # D3: unstructured random automata (incompressible)
        out = [random_fsm(nmax, 4, 5000 + 13 * i + seed) for i in range(P)]
    return out

def ref_run(pr, X):
    """X (B,L) -> (B,L) readout symbols"""
    B, L = X.shape; s = np.full(B, pr["init"]); out = np.zeros((B, L), int)
    for t in range(L): s = pr["delta"][X[:, t], s]; out[:, t] = pr["table"][s]
    return out

def diversity_stats(progs):
    """number of distinct per-token transition functions (up to padding) and description length in bits"""
    rows = {tuple(pr["delta"][v]) for pr in progs for v in range(V)}; bits = sum(V * pr["n"] * np.log2(max(pr["n"], 2)) for pr in progs)
    return dict(P=len(progs), distinct_token_maps=len(rows), total_token_maps=len(progs) * V, bits=float(bits), families=sorted({pr["fam"] for pr in progs}))

# ------------------------------------------------------------------ interpreter: build keys, write fast weights, verify
def make_theta(dk, nmax, P, seed, hashed=False):
    r = np.random.default_rng(seed); g = lambda *s: r.choice([-1.0, 1.0], size=s).astype(np.float32)
    return dict(A=g(dk, V), B=g(dk, nmax), C=g(dk, P), a=np.float32(2.0), b=np.float32(0.5))

def triples(progs, nmax):
    """tape entries (p,v,j,i,class)"""
    T = [(p, v, j, int(pr["delta"][v, j])) for p, pr in enumerate(progs) for v in range(V) for j in range(pr["n"])]; return np.array(T)

def keys_np(th, p, v, j):
    dk = th["A"].shape[0]; return th["A"][:, v] * th["B"][:, j] * th["C"][:, p] / np.sqrt(dk)
def rkeys_np(th, p, i):
    dk = th["A"].shape[0]; return th["B"][:, i] * th["C"][:, p] / np.sqrt(dk)
def ikeys_np(th, p):
    dk = th["A"].shape[0]; return th["C"][:, p] / np.sqrt(dk)

def tape(progs, th, nmax, Kmax):
    """Program tape -> (keys, target one-hots) for the three fast memories"""
    T = triples(progs, nmax); dk = th["A"].shape[0]
    Kf = (th["A"][:, T[:, 1]] * th["B"][:, T[:, 2]] * th["C"][:, T[:, 0]]).T / np.sqrt(dk); Yf = np.eye(nmax, dtype=np.float32)[T[:, 3]]
    Rk, Ry = [], []
    for p, pr in enumerate(progs):
        for i in range(pr["n"]): Rk.append(th["B"][:, i] * th["C"][:, p] / np.sqrt(dk)); Ry.append(np.eye(Kmax, dtype=np.float32)[pr["table"][i]])
    Ik = np.stack([ikeys_np(th, p) for p in range(len(progs))]); Iy = np.stack([np.eye(nmax, dtype=np.float32)[pr["init"]] for pr in progs])
    return dict(F=(Kf.astype(np.float32), Yf), R=(np.array(Rk, np.float32), np.array(Ry, np.float32)), S0=(Ik.astype(np.float32), Iy))

HI, LO = 0.85, 0.15     # write-rule target band (snap switches at z=.75 / .25)
def write_memory(Kmat, Y, steps=1500, lr=0.02, seed=0, F0=None, opt="adam"):
    """Fast-weight write phase: clipped delta rule iterated over the tape.  Per tape entry (k,y): F += eta * (target - F k) k^T, applied only when the entry is
    not yet inside its margin band (positives must reach z>=HI, negatives z<=LO).  Mini-batched with Adam step sizes for speed; start from the 1-pass Hebbian/ridge write."""
    M, dk = Kmat.shape; n = Y.shape[1]; Kj = jnp.array(Kmat); Yj = jnp.array(Y)
    lam = 1e-3
    if F0 is not None: F0 = jnp.asarray(F0)
    elif M <= dk: F0 = jnp.linalg.solve(Kj @ Kj.T + lam * jnp.eye(M), Yj).T @ Kj
    else: F0 = jnp.linalg.solve(Kj.T @ Kj + lam * jnp.eye(dk), Kj.T @ Yj).T
    def loss(F):
        Z = Kj @ F.T; e = jnp.where(Yj > 0, jnp.maximum(HI - Z, 0.0), jnp.maximum(Z - LO, 0.0)); return (e ** 2).sum()
    vg = jax.jit(jax.value_and_grad(loss)); m = jnp.zeros_like(F0); v = jnp.zeros_like(F0); F = F0; b1, b2 = 0.9, 0.999
    ok = lambda F: bool(band_ok(np.asarray(Kj @ F.T), Y))
    if ok(F): return np.asarray(F), 0
    best = None
    if opt == "sgd":      # plain clipped delta rule (no per-weight step normalisation): the fast-weight-programmer semantics
        for t in range(1, steps + 1):
            l, g = vg(F); F = F - 0.25 * g
            if t % 10 == 0 and ok(F): return np.asarray(F), t
        return np.asarray(F), -1
    for t in range(1, steps + 1):
        l, g = vg(F); m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g; F = F - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + 1e-8)
        if t % 25 == 0 and ok(F): return np.asarray(F), t
    return np.asarray(F), -1
def margins_ok(Z, Y): return Z[Y > 0].min() >= 0.75 + 1e-4 and Z[Y == 0].max() <= 0.25 - 1e-4      # exactness (snap saturates to exact 0/1)
def band_ok(Z, Y): return Z[Y > 0].min() >= HI - 0.02 and Z[Y == 0].max() <= LO + 0.02            # write-rule success: exactness WITH a safety band of >= 0.08 on both sides

def build_interpreter(progs, dk, seed=0, steps=1500, nmax=None, Kmax=None, th=None):
    """slow weights theta (random binding codes, width dk) + fast memories written from the program tape.  The transition memory F uses the full key width dk;
    the readout memory R and init memory S0 store far fewer associations, so each uses the smallest key PREFIX (width d_R, d_S <= dk) that still stores its tape exactly."""
    nmax = nmax or max(pr["n"] for pr in progs); Kmax = Kmax or max(pr["K"] for pr in progs); P = len(progs)
    th = th if th is not None else make_theta(dk, nmax, P, seed); tp = tape(progs, th, nmax, Kmax); phi = {}; info = {}; allok = True
    for name, (Km, Y) in tp.items():
        widths = [dk] if name == "F" else sorted({w for w in (8, 16, 32, 64, 128, 256, 512, 1024) if w < dk} | {dk})
        for w in widths:
            Fm, t = write_memory(Km[:, :w], Y, steps=steps)
            if t >= 0: break
        phi[name] = Fm.astype(np.float32); info[name] = t; allok &= (t >= 0)
    return th, phi, dict(write_steps=info, written=allok, nmax=nmax, Kmax=Kmax, P=P, dk=dk, widths={k: int(v.shape[1]) for k, v in phi.items()})

def verify_exact(progs, th, phi, nmax, Kmax):
    """Exhaustive one-step certificate: every (program, token, state) transition lands on its exact one-hot successor, every readout argmax is right, every init is right.
    Because the snap returns exact 0/1 vectors, the one-hot invariant holds for sequences of ANY length (induction)."""
    tp = tape(progs, th, nmax, Kmax); res = {}
    for name, (Km, Y) in tp.items():
        Z = Km[:, :phi[name].shape[1]] @ phi[name].T
        if name == "R": res[name] = bool(all(((Z[idx] if False else Z)[k].argmax() == Y[k].argmax()) for k in range(len(Y))))
        else: res[name] = bool(margins_ok(Z, Y))
        res[name + "_min_pos"] = float(Z[Y > 0].min()); res[name + "_max_neg"] = float(Z[Y == 0].max())
    res["exact"] = bool(res["F"] and res["R"] and res["S0"]); return res

# ------------------------------------------------------------------ execution (jax)
def to_j(d): return {k: jnp.asarray(v) for k, v in d.items()}
def _nrm(x): return jnp.sqrt((x ** 2).sum(-1, keepdims=True) + 1e-12)      # NaN-safe norm (gradient-safe at 0)
def run_fw(th, phi, X, nprog, nmax, noise=None, key=None):
    """X (B,L) shared token stream -> logits (P,B,L,Kmax); each program p addressed by code C[:,p].
    noise = dict(kind: 'F_fresh'|'theta_fresh'|'act_fresh'|'act_fixed'..., sigma) -- additive noise sampled in key-component space (exactly equivalent in law to iid Gaussian entry noise)."""
    dk = th["A"].shape[0]; B, L = X.shape; xs = jax.nn.one_hot(X, V); sq = jnp.sqrt(dk * 1.0); noise = noise or {}
    sF, sT, sA = noise.get("F", 0.0), noise.get("theta", 0.0), noise.get("act", 0.0)      # may be traced scalars (noise is always applied, sigma=0 -> no effect)
    gcode = th["C"].T                                                         # (P,dk)
    def snap(z): return jnp.clip(th["a"] * z - th["b"], 0.0, 1.0)
    s0 = snap(gcode[:, :phi["S0"].shape[1]] @ phi["S0"].T / sq)                                      # (P,nmax)
    s = jnp.broadcast_to(s0[:, None], (nprog, B, nmax)); key = key if key is not None else jax.random.PRNGKey(0)
    def step(s, inp):
        x, kk = inp; k1, k2, k3, k4, k5, k6 = jax.random.split(kk, 6)
        s_in = s + sA * jax.random.normal(k1, s.shape)
        Ax = (x @ th["A"].T)[None]                                            # (1,B,dk)
        Bs = s_in @ th["B"].T                                                 # (P,B,dk)
        Cg = gcode[:, None]                                                   # (P,1,dk)
        # fresh slow-weight noise: (dA x)_d ~ N(0,s^2), (dB s)_d ~ N(0,s^2|s|^2), (dC g)_d ~ N(0,s^2)
        Ax = Ax + sT * jax.random.normal(k2, (nprog, B, dk)); Bs = Bs + sT * _nrm(s_in) * jax.random.normal(k3, (nprog, B, dk)); Cg = Cg + sT * jax.random.normal(k4, (nprog, B, dk))
        kv = Ax * Bs * Cg / sq
        z = kv @ phi["F"].T
        dR = phi["R"].shape[1]
        z = z + sF * _nrm(kv) * jax.random.normal(k5, z.shape)
        s2 = snap(z)
        rk = ((s2 @ th["B"].T) * gcode[:, None] / sq)[..., :dR]; y = rk @ phi["R"].T
        y = y + sF * _nrm(rk) * jax.random.normal(k6, y.shape)
        return s2, y
    ks = jax.random.split(key, L); _, ys = jax.lax.scan(step, s, (jnp.swapaxes(xs, 0, 1), ks)); return jnp.transpose(ys, (1, 2, 0, 3))   # (P,B,L,K)

def predict(th, phi, X, nprog, nmax, Kmax_valid=None, **kw):
    lg = run_fw(th, phi, X, nprog, nmax, **kw)
    return lg.argmax(-1)

# ------------------------------------------------------------------ noise protocols on stored tensors
THETA_KEYS = ("A", "B", "C"); PHI_KEYS = ("F", "R", "S0")
def perturb(th, phi, key, sig_theta=0.0, sig_phi=0.0, rel=False, theta_scalars=True):
    """static (fixed-through-time) Gaussian perturbation of slow and/or fast tensors; rel=True -> sigma * RMS(tensor)"""
    th2, phi2 = dict(th), dict(phi); ks = jax.random.split(key, 8); i = 0
    for k in THETA_KEYS:
        if sig_theta > 0: th2[k] = th[k] + sig_theta * (jnp.sqrt(jnp.mean(th[k] ** 2)) if rel else 1.0) * jax.random.normal(ks[i], th[k].shape)
        i += 1
    if sig_theta > 0 and theta_scalars:
        th2["a"] = th["a"] + sig_theta * (th["a"] if rel else 1.0) * jax.random.normal(ks[6]); th2["b"] = th["b"] + sig_theta * (th["b"] if rel else 1.0) * jax.random.normal(ks[7])
    for k in PHI_KEYS:
        if sig_phi > 0: phi2[k] = phi[k] + sig_phi * (jnp.sqrt(jnp.mean(phi[k] ** 2)) if rel else 1.0) * jax.random.normal(ks[i], phi[k].shape)
        i += 1
    return th2, phi2

def s50(curve):
    xs = np.array([float(k) for k in curve]); ys = np.array([np.mean(v) for v in curve.values()])
    if ys[0] < 0.5: return float("nan")
    for i in range(1, len(xs)):
        if ys[i] < 0.5:
            f = (ys[i - 1] - 0.5) / (ys[i - 1] - ys[i]); return float(np.exp(np.log(xs[i - 1]) + f * (np.log(xs[i]) - np.log(xs[i - 1]))))
    return float("inf")

def make_eval(progs, nmax, Kmax, L, B=32, seed=0):
    """returns X, refs (P,B,L) and a jitted function  acc(th,phi,key,noise) -> (per-program exact-sequence accuracy (P,), joint accuracy)"""
    rng = np.random.default_rng(seed); X = rng.choice(V, size=(B, L)); refs = np.stack([ref_run(pr, X) for pr in progs]); Xj = jnp.array(X); R = jnp.array(refs); P = len(progs)
    @jax.jit
    def _acc(th, phi, key, sF, sT, sA):
        pred = predict(th, phi, Xj, P, nmax, key=key, noise=dict(F=sF, theta=sT, act=sA)); ok = (pred == R).all(-1)      # (P,B)
        return ok.mean(1), ok.all(0).mean()
    def acc(th, phi, key, noise=None):
        nz = noise or {}; return _acc(th, phi, key, jnp.float32(nz.get("F", 0.0)), jnp.float32(nz.get("theta", 0.0)), jnp.float32(nz.get("act", 0.0)))
    return X, refs, acc

SIG = [float(x) for x in np.logspace(-3, 0.3, 14)]
def sigma50_curve(accf, th, phi, mode, rel=False, NS=3, sigmas=SIG, noise_mode="static"):
    """mode in {'theta','phi','both'}; static = perturb stored tensors once per run; fresh = new noise every step.  Returns per-program-mean s50 and joint s50 (+ curves)."""
    pp, jj = {}, {}
    for s in sigmas:
        pv, jv = [], []
        for i in range(NS):
            key = jax.random.PRNGKey(1000 + i)
            if noise_mode == "static":
                th2, phi2 = perturb(th, phi, key, s if mode in ("theta", "both") else 0.0, s if mode in ("phi", "both") else 0.0, rel); a, j = accf(th2, phi2, key)
            else:
                nz = dict(F=s if mode in ("phi", "both") else 0.0, theta=s if mode in ("theta", "both") else 0.0); a, j = accf(th, phi, key, nz)
            pv.append(float(np.mean(np.asarray(a)))); jv.append(float(j))
        pp[str(s)] = pv; jj[str(s)] = jv
    return dict(per_program=s50(pp), joint=s50(jj), curve_pp={k: float(np.mean(v)) for k, v in pp.items()}, curve_joint={k: float(np.mean(v)) for k, v in jj.items()})
