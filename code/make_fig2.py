import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = json.load(open("compare.json")); N = json.load(open("noln.json"))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
for a, task in zip(ax, ["hist", "sort_unique", "reverse"]):
    mean = lambda d: [np.mean(v) for v in d.values()]
    xs = [float(k) for k in C["compiled"][task]["rms"]]
    a.plot(xs, mean(C["compiled"][task]["rms"]), "r-o", ms=4, lw=2, label="compiled (Tracr)")
    for i, (s, r) in enumerate(C["trained"][task].items()):
        a.plot(xs, mean(r["rms"]), "b-", alpha=.6, lw=1.5, label="trained (LayerNorm), 3 seeds" if i == 0 else None)
    for i, (s, r) in enumerate(N.get(task, {}).items()):
        a.plot(xs, mean(r["rms"]), "g--", alpha=.8, lw=1.5, label="trained (no LayerNorm), 2 seeds" if i == 0 else None)
    a.set_xscale("log"); a.axhline(.5, ls=":", c="gray"); a.set_title(task); a.set_xlabel("relative noise α (σ = α · RMS of each tensor)")
ax[0].set_ylabel("exact-sequence accuracy"); ax[0].legend(fontsize=8, loc="lower left")
plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/compiled_vs_trained.png", dpi=140)
