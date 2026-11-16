import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))   # repository root
import json, csv, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
P = REPO + "/"; J = lambda p: json.load(open(P + p)); T = ["length", "hist", "reverse", "parity", "majority8", "dyck8"]
# ---- K1: sigma50 by task
fr = [d for d in J("llm_synthesis/fresh_run/results/fresh_summary.json") if d.get("s50") and d["cat"] == "3b" and d["fb"] == "none" and d["rev"] is None]; co = [d for d in J("llm_synthesis/results/contaminated_summary.json") if d["category"] == "3b"]; tr = J("theory/cross_tracr.json")
fig, ax = plt.subplots(figsize=(11, 4.2)); xt = []; xl = []
for i, t in enumerate(T):
    for j, c in enumerate(["constrained", "unconstrained"]):
        x = i * 3 + j; xt.append(x); xl.append(f"{t}\n{c[:5]}.")
        v = [d["s50"]["abs"] for d in fr if d["task"] == t and d["cond"] == c]; ax.scatter(np.full(len(v), x) + np.linspace(-.12, .12, len(v)), v, c="C0", s=28, label="fresh run" if (i, j) == (0, 0) else None)
        u = [d["s50"]["abs"] for d in co if d["task"] == t and d["condition"] == c]; ax.scatter(np.full(len(u), x) + .2, u, c="C3", marker="D", s=34, label="contaminated run" if (i, j) == (0, 0) else None)
    if t in tr and tr[t]["meas"]["abs"] == tr[t]["meas"]["abs"]: ax.scatter([i * 3 + .5], [tr[t]["meas"]["abs"]], c="k", marker="s", s=44, label="Tracr compiled" if t == "length" else None)
ax.set_yscale("log"); ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7); ax.set_ylabel("sigma50, absolute noise"); ax.legend(fontsize=8); ax.set_title("LLM-built transformers: noise tolerance per attempt (attempt-to-attempt spread is large)"); plt.tight_layout(); plt.savefig("figures/K1_sigma50_by_task.png", dpi=140); plt.close()
# ---- L1: predictors v3 and v4 across constructions
v4f = {k: v for k, v in J("theory/v4_fresh.json").items() if "meas" in v}; v4t = J("theory/v4_tracr.json"); v4b = J("theory/v4_bilinear.json")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for a, ver in zip(ax, ("v3", "v4")):
    x = [v["meas"] for v in v4f.values()]; y = [v[ver] for v in v4f.values()]; a.scatter(x, y, c="C0", s=22, label="LLM-built (fresh run, held-out)")
    xt = [(v["meas"], v[ver], k) for k, v in v4t.items() if v["meas"] == v["meas"] and v[ver] == v[ver]]; a.scatter([q[0] for q in xt], [q[1] for q in xt], c="k", marker="s", s=40, label="Tracr compiled")
    for m, p, k in xt:
        if k in ("length", "hist", "reverse"): a.annotate(k, (m, p), fontsize=8, xytext=(4, -10), textcoords="offset points")
    a.scatter([v["meas"] for v in v4b.values()], [v[ver] for v in v4b.values()], c="C2", marker="^", s=30, label="bilinear restoring-state")
    lo, hi = 2e-4, 5; a.plot([lo, hi], [lo, hi], "k-", lw=.8); a.fill_between([lo, hi], [lo / 2, hi / 2], [lo * 2, hi * 2], color="grey", alpha=.15); a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("measured sigma50"); a.set_title({"v3": "v3: tangent-cone model", "v4": "v4: plus saturation margins"}[ver])
ax[0].set_ylabel("predicted sigma50"); ax[0].legend(fontsize=8, loc="upper left"); fig.suptitle("Parameter-free predictors of the failure level (grey band: within a factor of 2)"); plt.tight_layout(); plt.savefig("figures/L1_predictors.png", dpi=140); plt.close()
# ---- L2: first-order L-inf radius vs attack radius (autodiff vs kink-aware)
kd = J("theory/kink_dev.json"); ld = J("theory/lin_dev.json"); names = list(kd); rho_k = [kd[n]["rho_lin_kink"] / kd[n]["atk"] for n in names]; rho_a = [ld[n]["pred"]["rho_linf"] / ld[n]["atk"] for n in names if n in ld]
fig, ax = plt.subplots(figsize=(10, 4.2)); idx = np.arange(len(names)); ax.bar(idx - .2, [ld[n]["pred"]["rho_linf"] / ld[n]["atk"] if n in ld else np.nan for n in names], .4, label="autodiff gradient (relu'(0) = 0)", color="C1"); ax.bar(idx + .2, rho_k, .4, label="one-sided derivative (relu'(0) = 1)", color="C0")
ax.axhline(1, c="k", lw=.8); ax.set_yscale("log"); ax.set_xticks(idx); ax.set_xticklabels([n.replace("contam__", "").replace("__3b__none__a1", "").replace("constrained", "con.").replace("unconstrained", "unc.") for n in names], rotation=40, ha="right", fontsize=8); ax.set_ylabel("first-order L-inf radius / attack radius"); ax.set_title("Kinks at the operating point defeat gradients\n(one-sided derivative: 10 of 12 networks match the attack, ratio 1)", fontsize=10); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("figures/L2_first_order_vs_attack.png", dpi=140); plt.close()
# ---- I0: certificate vs attack
At = J("results/A_tied.json"); C4 = J("results/C4_cert.json"); pts = {"tied (bilinear/ReLU, dense/protected)": [], "fast-weight interpreter": [], "LLM-built transformers (sample-certified)": []}
for k, v in At.items():
    if k.startswith("attack|") and v["any"]["first_break_multiple"]: pts["tied (bilinear/ReLU, dense/protected)"].append((v["eps_cert"], v["eps_cert"] * v["any"]["first_break_multiple"]))
for P_ in (1, 4):
    c = C4[f"cert|P{P_}"]; a = C4[f"attack|P{P_}"]; pts["fast-weight interpreter"] += [(c["eps_F"], c["eps_F"] * a["fast"]["first_break_t"]), (c["eps_theta"], c["eps_theta"] * a["slow"]["first_break_t"])]
for d in fr + [dict(cert=x["cert"], atk=x["attack"], s50={}) for x in co]:
    c = d.get("cert"); a = d.get("atk", d.get("attack"))
    if c and a: pts["LLM-built transformers (sample-certified)"].append((c, a))
fig, ax = plt.subplots(figsize=(6.6, 5.6))
for (n, v), m in zip(pts.items(), "osD"): ax.scatter([p[0] for p in v], [p[1] for p in v], marker=m, s=26, label=n, alpha=.75)
r = np.array([1e-6, 1e-1]); ax.plot(r, r, "k-", lw=.8); ax.plot(r, r * 10, "k:", lw=.8); ax.text(2e-6, 2.5e-5, "10x", fontsize=8); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("certified radius (lower bound)"); ax.set_ylabel("first failure found by attack (upper bound)"); ax.legend(fontsize=7, loc="upper left"); ax.set_title("How tight are the certificates?"); plt.tight_layout(); plt.savefig("figures/I0_certificate_gap.png", dpi=140); plt.close()
# ---- X1: overview
rows = list(csv.DictReader(open(P + "results/implementation_dataset.csv"))); grp = {"Tracr compiled\n(project tasks)": [], "LLM-built\n(code, 3b)": [], "LLM-built\n(literal, 4)": [], "bilinear\nrestoring-state": [], "state codes\n(static abs)": []}
for r_ in rows:
    try: v = float(r_["s50_abs"])
    except Exception: continue
    if not (v > 0 and np.isfinite(v)): continue
    f, c = r_["family"], r_["category"]
    if f.startswith("transformer(Tracr)"): grp["Tracr compiled\n(project tasks)"].append(v)
    elif f.startswith("transformer(LLM)"): grp["LLM-built\n(literal, 4)" if c == "4" else "LLM-built\n(code, 3b)"].append(v)
    elif f.startswith("bilinear"): grp["bilinear\nrestoring-state"].append(v)
    elif f.startswith("state"): grp["state codes\n(static abs)"].append(v)
fig, ax = plt.subplots(figsize=(9.5, 4.4)); rng = np.random.default_rng(0)
for i, (k, v) in enumerate(grp.items()): ax.scatter(i + rng.uniform(-.18, .18, len(v)), v, s=14, alpha=.7); ax.text(i, 1.3e-4, f"n={len(v)}", ha="center", fontsize=8)
ax.set_xticks(range(len(grp))); ax.set_xticklabels(list(grp), fontsize=8); ax.set_yscale("log"); ax.set_ylabel("sigma50, absolute noise"); ax.set_ylim(1e-4, 1); ax.set_title("Noise tolerance by construction method (absolute noise)\n(trained transformers are measured in the relative metric, see Part A)", fontsize=10); plt.tight_layout(); plt.savefig("figures/X1_overview_construction.png", dpi=140); plt.close()
# ---- diagrams
def box(ax, x, y, w, h, t, c="#dfe9f5", fs=8): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=c, ec="k", lw=.8)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
def arr(ax, a, b): ax.add_patch(FancyArrowPatch(a, b, arrowstyle="->", mutation_scale=10, lw=.9))
fig, ax = plt.subplots(figsize=(9.5, 3.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
box(ax, .2, 2.5, 1.5, .9, "token t\n(one-hot)"); box(ax, .2, 1.2, 1.5, .9, "state s\n(one-hot)"); box(ax, 2.4, 1.85, 2.2, 1.0, "role gates g(program, token)\n-> weights over {up, down, hold}", "#f6e6c8"); box(ax, 5.2, 1.85, 2.2, 1.0, "z = sum_r g_r * P_r s\n(shared matrices P_up, P_down, P_hold)", "#d8efd8"); box(ax, 8.0, 1.85, 1.7, 1.0, "clip(2z - .5, 0, 1)\n= new state", "#efd8d8")
arr(ax, (1.7, 2.95), (2.4, 2.5)); arr(ax, (1.7, 1.65), (5.2, 2.2)); arr(ax, (4.6, 2.35), (5.2, 2.35)); arr(ax, (7.4, 2.35), (8.0, 2.35)); ax.text(5, .5, "k programs reuse the same three matrices; only the gates differ per program (parameters about 3,400-3,900 for 1-30 programs)", ha="center", fontsize=8); ax.set_title("Weight-tied bilinear composition (Part D)"); plt.tight_layout(); plt.savefig("figures/diagram_tied.png", dpi=140); plt.close()
fig, ax = plt.subplots(figsize=(10, 4.2)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 4.4)
box(ax, .2, 3.2, 1.6, .8, "token x"); box(ax, .2, 2.1, 1.6, .8, "state s"); box(ax, .2, 1.0, 1.6, .8, "program address g"); box(ax, 2.6, 2.0, 2.7, 1.2, "key k = (A x) o (B s) o (C g) / sqrt(dk)\nslow weights A, B, C\n(shared by all programs)", "#f6e6c8"); box(ax, 6.0, 2.0, 2.0, 1.2, "fast memory F\n(written from the\nprogram tape)", "#d8efd8"); box(ax, 8.6, 2.0, 2.0, 1.2, "s' = clip(2 F k - .5, 0, 1)\nrestoring state", "#efd8d8"); box(ax, 6.0, .3, 2.0, .9, "fast readout R\nfast init S0", "#d8efd8")
for y in (3.6, 2.5, 1.4): arr(ax, (1.8, y), (2.6, 2.6))
arr(ax, (5.3, 2.6), (6.0, 2.6)); arr(ax, (8.0, 2.6), (8.6, 2.6)); arr(ax, (9.6, 2.0), (8.0, .8)); ax.text(5.5, 3.9, "program tape (token, state, successor) is written into F, R, S0 by a clipped delta rule", fontsize=8, ha="center"); ax.set_title("Fast-weight universal interpreter (Part I)"); plt.tight_layout(); plt.savefig("figures/diagram_fwi.png", dpi=140); plt.close()
print("figures done")
