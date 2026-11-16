import json, numpy as np
R = json.load(open("hardening.json"))
def s50(curve):
    xs=np.array([float(k) for k in curve]); ys=np.array([np.mean(v) for v in curve.values()])
    if ys[0]<0.5: return float("nan")
    for i in range(1,len(xs)):
        if ys[i]<0.5:
            f=(ys[i-1]-0.5)/(ys[i-1]-ys[i]); return float(np.exp(np.log(xs[i-1])+f*(np.log(xs[i])-np.log(xs[i-1]))))
    return float("inf")
progs=list(R); variants=list(R[progs[0]])
print("sigma50 (absolute noise on every weight; higher=better). '--' = clean accuracy < 100% (not exact)")
print(f"{'variant':14s}"+"".join(f"{p:>13s}" for p in progs))
for v in variants:
    row=[]
    for p in progs:
        r=R[p][v]
        row.append("--" if "abs" not in r else f"{s50(r['abs']):.2e}")
    print(f"{v:14s}"+"".join(f"{x:>13s}" for x in row))
print("\nGain vs base (x):")
for v in variants[1:]:
    row=[]
    for p in progs:
        r=R[p][v]; b=s50(R[p]['base']['abs'])
        row.append("--" if "abs" not in r else f"{s50(r['abs'])/b:.1f}x")
    print(f"{v:14s}"+"".join(f"{x:>13s}" for x in row))
