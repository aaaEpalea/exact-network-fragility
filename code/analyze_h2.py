import json, numpy as np
R = json.load(open("hardening2.json"))
def s50(curve):
    xs=np.array([float(k) for k in curve]); ys=np.array([np.mean(v) for v in curve.values()])
    if ys[0]<0.5: return float("nan")
    for i in range(1,len(xs)):
        if ys[i]<0.5:
            f=(ys[i-1]-0.5)/(ys[i-1]-ys[i]); return float(np.exp(np.log(xs[i-1])+f*(np.log(xs[i])-np.log(xs[i-1]))))
    return float("inf")
progs=list(R); variants=list(R[progs[0]])
for mode,title in [("abs","DENSE noise (every entry, incl. zeros)"),("nz","STRUCTURAL-ZEROS-PRESERVED noise (nonzero entries only)")]:
    print(f"\n=== sigma50, {title}; gain vs base in parentheses ===")
    print(f"{'variant':20s}"+"".join(f"{p:>20s}" for p in progs))
    for v in variants:
        row=[]
        for p in progs:
            r=R[p][v]
            if mode not in r: row.append("-- (not exact)"); continue
            b=s50(R[p]['base'][mode]); x=s50(r[mode])
            row.append(f"{x:.1e} ({x/b:.1f}x)")
        print(f"{v:20s}"+"".join(f"{x:>20s}" for x in row))
