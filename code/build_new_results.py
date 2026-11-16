"""Regenerates figures and the tables of RESULTS.md section I (fast-weight interpreter + plan-completion experiments) from results/*.json."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "../results/"; F = "../figures/"; L = lambda n: json.load(open(R + n))
C1, A, B, C2, C4, C5 = L("C1_scaling.json"), L("A_tied.json"), L("B_codes.json"), L("C2_robust.json"), L("C4_cert.json"), L("C5_comp.json")
md = {}
# ---------------- fig I1
kinds = ["template", "family", "mixed", "random"]; Ps = {k: [1, 2, 4, 8, 16] if k != "random" else [1, 2, 4, 8, 12] for k in kinds}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k in kinds: ax[0].plot(Ps[k], [C1[f"{k}|P{p}"]["dk"] / C1[f"{k}|P{p}"]["M_tape"] for p in Ps[k]], "o-", label=k)
ax[0].set_xscale("log", base=2); ax[0].set_ylim(0, 1); ax[0].set_xlabel("programs P"); ax[0].set_ylabel("min key width dk / tape entries M"); ax[0].legend(); ax[0].set_title("capacity of one shared fast memory")
m = "mixed"; ax[1].plot(Ps[m], [C1[f"{m}|P{p}"]["params_fast"] for p in Ps[m]], "o-", label="FWI shared fast memory")
ax[1].plot([p for p in Ps[m] if p <= 8], [C1[f"{m}|P{p}"]["private_fast_params"] for p in Ps[m] if p <= 8], "s--", label="FWI private memory per program")
ax[1].plot(Ps[m], [C1[f"{m}|P{p}"]["params_indep_dense"] for p in Ps[m]], "^-", label="independent bilinear (dense)"); ax[1].plot(Ps[m], [C1[f"{m}|P{p}"]["params_blockdiag"] for p in Ps[m]], "v-", label="block-diagonal stack")
ax[1].plot(Ps[m], [C1[f"{m}|P{p}"]["params_indep_nonzero"] for p in Ps[m]], "x:", label="independent (non-zeros only)"); ax[1].plot(Ps[m], [C1[f"{m}|P{p}"]["params_tied_dictionary"] for p in Ps[m]], "d-", label="role-dictionary tied")
ax[1].set_xscale("log", base=2); ax[1].set_yscale("log"); ax[1].set_xlabel("programs P (mixed families)"); ax[1].set_ylabel("parameters"); ax[1].legend(fontsize=7); ax[1].set_title("parameter accounting (n<=21)")
plt.tight_layout(); plt.savefig(F + "I1_fwi_scaling.png", dpi=130); plt.close()
# ---------------- fig I2
fig, ax = plt.subplots(figsize=(5.5, 4))
for P in [1, 4]:
    ns = [8, 12, 16, 24, 32, 48, 64, 96]; ax.plot(ns, [C1[f"nscale|P{P}|n{n}"]["dk_over_M"] for n in ns], "o-", label=f"P={P}")
ax.set_xscale("log", base=2); ax.set_ylim(0, 1); ax.set_xlabel("states n (random automata)"); ax.set_ylabel("dk_min / M"); ax.legend(); ax.set_title("no compression gain with n"); plt.tight_layout(); plt.savefig(F + "I2_nscale.png", dpi=130); plt.close()
# ---------------- fig I3 tied attacks
ks = [1, 2, 4, 8, 15, 30]; fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for j, N in enumerate([16, 32]):
    for d, c in [("bil", "C0"), ("relu", "C3")]:
        kk = [k for k in ks if f"attack|{d}|N{N}|dense|k{k}" in A]; ax[j].plot(kk, [A[f"attack|{d}|N{N}|dense|k{k}"]["any"]["first_break_multiple"] for k in kk], "o-", color=c, label=f"{d} any-head"); ax[j].plot(kk, [A[f"attack|{d}|N{N}|dense|k{k}"]["head"]["first_break_multiple"] for k in kk], "s--", color=c, label=f"{d} single head")
    ax[j].axhline(1, color="k", lw=0.8); ax[j].set_xscale("log", base=2); ax[j].set_xlabel("heads k"); ax[j].set_ylabel("first failure / certified radius"); ax[j].set_title(f"weight-tied, N={N} (dense)"); ax[j].legend(fontsize=7); ax[j].set_ylim(0.8, 5)
plt.tight_layout(); plt.savefig(F + "I3_tied_attack.png", dpi=130); plt.close()
# ---------------- fig I4 cert vs P
fig, ax = plt.subplots(figsize=(5.5, 4)); P_ = [1, 2, 4, 8, 16]
ax.plot(P_, [C4[f"cert|P{p}"]["eps_F"] for p in P_], "o-", label="fast weights"); ax.plot(P_, [C4[f"cert|P{p}"]["eps_theta"] for p in P_], "s-", label="slow weights")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("programs P (shared memory)"); ax.set_ylabel("certified entrywise radius"); ax.legend(); ax.set_title("certified radius shrinks with P"); plt.tight_layout(); plt.savefig(F + "I4_cert_vs_P.png", dpi=130); plt.close()
# ---------------- fig I5 codes
fig, ax = plt.subplots(1, 3, figsize=(14, 4)); cols = {"onehot": "k", "rep3": "C0", "rep5": "C1", "willshaw": "C3"}
for j, (key, ttl) in enumerate([("sigma50_static_abs", "static parameter noise"), ("sigma50_fresh_act", "fresh activation noise"), ("sigma50_burst_act", "single activation burst")]):
    for kind, c in cols.items():
        pts = [(B[f"majority|N{N}|{kind}"]["params"], B[f"majority|N{N}|{kind}"][key]) for N in [8, 16, 32] if B.get(f"majority|N{N}|{kind}", {}).get("exact")]
        ax[j].plot(*zip(*pts), "o-", color=c, label=kind)
    ax[j].set_xscale("log"); ax[j].set_yscale("log"); ax[j].set_xlabel("parameters (N = 8, 16, 32)"); ax[j].set_ylabel("sigma50"); ax[j].set_title(ttl + " (majority)"); ax[j].legend(fontsize=7)
plt.tight_layout(); plt.savefig(F + "I5_codes.png", dpi=130); plt.close()
# ---------------- fig I6 targets
fig, ax = plt.subplots(figsize=(8, 4)); names = [("static_slow_after_write", "slow"), ("static_fast", "fast"), ("static_joint", "both"), ("fresh_slow", "fresh slow"), ("fresh_fast", "fresh fast"), ("fresh_joint", "fresh both"), ("act_fresh", "activation")]
w = 0.25
for i, P in enumerate([1, 4, 16]): ax.bar(np.arange(len(names)) + i * w, [C2[f"targets|P{P}"][k]["joint"] for k, _ in names], w, label=f"P={P} (all programs exact)")
ax.set_xticks(np.arange(len(names)) + w); ax.set_xticklabels([n for _, n in names], rotation=20); ax.set_ylabel("sigma50"); ax.legend(); ax.set_title("perturbation targets (joint exactness)"); plt.tight_layout(); plt.savefig(F + "I6_targets.png", dpi=130); plt.close()
# ---------------- spearman mechanism
from scipy.stats import spearmanr
cells = [v for v in B.values() if v.get("exact") and np.isfinite(v["sigma50_fresh_act"])]
rho = spearmanr([v["mechanism"]["slack_over_row"] for v in cells], [v["sigma50_fresh_act"] for v in cells]); rho2 = spearmanr([v["cert_eps_par"] for v in cells], [v["sigma50_static_abs"] for v in cells])
print("spearman slack/row vs fresh_act sigma50", rho, " cert eps vs static sigma50", rho2, "n cells", len(cells))
json.dump(dict(rho_slack_fresh_act=float(rho[0]), p1=float(rho[1]), rho_cert_static=float(rho2[0]), p2=float(rho2[1]), n=len(cells)), open(R + "I_mechanism_corr.json", "w"))
