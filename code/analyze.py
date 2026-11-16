import json, numpy as np
R = json.load(open("results2.json"))
names = list(R)
def sigma50(curve):
    xs = np.array([float(k) for k in curve]); ys = np.array([np.mean(v) for v in curve.values()])
    if ys[0] < 0.5: return float("nan")   # already failing at smallest noise
    for i in range(1, len(xs)):
        if ys[i] < 0.5:
            f = (ys[i-1]-0.5)/(ys[i-1]-ys[i])
            return float(np.exp(np.log(xs[i-1]) + f*(np.log(xs[i])-np.log(xs[i-1]))))
    return float("inf")
print(f"{'program':12s} {'layers':>6s} {'d':>4s} {'params':>7s} {'nz%':>5s} | {'s50 abs':>9s} {'s50 rel':>9s} {'s50 zero-only':>13s} {'s50 nonzero-only':>16s}")
rows = {}
for n in names:
    r = R[n]
    s = [sigma50(r[k]) for k in ["abs","rel","zero_only","nonzero_only"]]
    rows[n] = s
    print(f"{n:12s} {r['layers']:6d} {r['d_model']:4d} {r['params']:7d} {100*r['nonzero_frac']:5.1f} | " + " ".join(f"{x:>9.2e}" if k<2 else f"{x:>{13 if k==2 else 16}.2e}" for k,x in enumerate(s)))
print("\nPer-component sigma50 (abs Gaussian noise on that component only):")
print(f"{'component':16s} " + " ".join(f"{n[:10]:>10s}" for n in names))
for g in R[names[0]]["group"]:
    print(f"{g:16s} " + " ".join(f"{sigma50(R[n]['group'][g]):10.2e}" for n in names))
print("\nbaseline:", {n: R[n]['baseline'] for n in names})
