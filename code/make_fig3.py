import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from compare_noise import alpha50
C = json.load(open("compare.json")); N = json.load(open("noise_train.json"))
tasks = ["hist", "sort_unique", "reverse"]
base = [alpha50(C["compiled"][t]["rms"]) for t in tasks]
hard = [alpha50(N[t]["from_balance+c=0.25"]["rms"]) for t in tasks]
trn = [np.mean([alpha50(v["rms"]) for v in C["trained"][t].values()]) for t in tasks]
x = np.arange(3); w = 0.26
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.bar(x - w, base, w, label="compiled (Tracr)", color="#d62728")
ax.bar(x, hard, w, label="compiled + hardening\n(rescale + attn temp + noise-aware fine-tune)", color="#ff7f0e")
ax.bar(x + w, trn, w, label="trained from scratch (3 seeds)", color="#1f77b4")
for i, (b, h, t) in enumerate(zip(base, hard, trn)):
    ax.text(i - w, b * 1.15, f"{b:.0e}", ha="center", fontsize=8); ax.text(i, h * 1.15, f"{h:.0e}" + ("*" if tasks[i] == "hist" else ""), ha="center", fontsize=8); ax.text(i + w, t * 1.15, f"{t:.2f}", ha="center", fontsize=8)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(tasks); ax.set_ylabel("α50: relative noise at which accuracy < 50%  (higher = more robust)")
ax.set_title("Hardening closes ~2–14× of the 15–170× robustness gap"); ax.legend(fontsize=8, loc="upper left")
ax.text(0.99, 0.02, "* hist: 6 of 335,922 sequences wrong after fine-tuning", transform=ax.transAxes, ha="right", fontsize=7, color="gray")
plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/hardening_summary.png", dpi=140)
print(list(zip(tasks, base, hard, trn)))
