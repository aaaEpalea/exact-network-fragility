import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open("results2.json")); names = list(R)
def s50(curve):
    xs=np.array([float(k) for k in curve]); ys=np.array([np.mean(v) for v in curve.values()])
    if ys[0]<0.5: return np.nan
    for i in range(1,len(xs)):
        if ys[i]<0.5:
            f=(ys[i-1]-0.5)/(ys[i-1]-ys[i]); return float(np.exp(np.log(xs[i-1])+f*(np.log(xs[i])-np.log(xs[i-1]))))
    return np.inf
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios":[1.15,1,1.05]})
# (a) accuracy vs sigma
for n in names:
    xs=[float(k) for k in R[n]["abs"]]; m=[np.mean(v) for v in R[n]["abs"].values()]; sd=[np.std(v) for v in R[n]["abs"].values()]
    ax[0].plot(xs,m,marker="o",ms=3,label=f"{n} ({R[n]['layers']}L)"); ax[0].fill_between(xs,np.array(m)-sd,np.array(m)+sd,alpha=.12)
ax[0].set_xscale("log"); ax[0].set_xlabel("Gaussian noise σ added to every weight (compiled weights are 0, ~1, or up to 100)")
ax[0].set_ylabel("exact-sequence accuracy"); ax[0].axhline(.5,ls=":",c="gray"); ax[0].set_title("(a) Accuracy collapses at σ ≈ 1e-4 – 1e-3"); ax[0].legend(fontsize=8)
# (b) zero-only vs nonzero-only
w=0.26; x=np.arange(len(names))
for j,(k,lab) in enumerate([("abs","all weights"),("zero_only","only the zero entries"),("nonzero_only","only the nonzero entries")]):
    v=[s50(R[n][k]) for n in names]; v=[1e-5 if (isinstance(t,float) and np.isnan(t)) else t for t in v]
    ax[1].bar(x+(j-1)*w,v,w,label=lab)
ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels(names,rotation=30,ha="right",fontsize=8)
ax[1].set_ylabel("σ at which accuracy falls below 50%  (higher = more robust)")
ax[1].set_title("(b) Noise on the ~97% zero entries causes the failure"); ax[1].legend(fontsize=8)
ax[1].text(0.02,0.02,"sort_freq: below grid (<1e-5) for 'all'/'zero'",transform=ax[1].transAxes,fontsize=7,color="gray")
# (c) component heatmap
groups=list(R[names[0]]["group"]); M=np.array([[s50(R[n]["group"][g]) for n in names] for g in groups])
Mp=np.where(np.isinf(M),0.2,np.where(np.isnan(M),5e-5,M))
im=ax[2].imshow(np.log10(Mp),cmap="RdYlGn",aspect="auto")
ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(names,rotation=30,ha="right",fontsize=8)
ax[2].set_yticks(range(len(groups))); ax[2].set_yticklabels(groups,fontsize=8)
for i in range(len(groups)):
    for j in range(len(names)):
        t="≥0.1" if np.isinf(M[i,j]) else ("<1e-4" if np.isnan(M[i,j]) else f"{M[i,j]:.0e}")
        ax[2].text(j,i,t,ha="center",va="center",fontsize=7)
ax[2].set_title("(c) σ50 by component (noise on that component only)"); plt.colorbar(im,ax=ax[2],label="log10 σ50")
plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/fragility_results.png",dpi=140)
