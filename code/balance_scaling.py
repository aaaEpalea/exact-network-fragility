
import json, warnings
warnings.filterwarnings("ignore")
from balance import *
res = {}
cfgs = [("plain c=0.25", lambda N: lib.make_length(), lambda N: int(1.2 * N * N), 0.25, [16, 32, 48]),
        ("chunk g=4 c=1", lambda N: chunked_length(N, 4), lambda N: 100, 1.0, [16, 48])]
for tag, pf, sf, c, Ns in cfgs:
    for N in Ns:
        r = run(f"{tag} N={N}", pf(N), N, sf(N), c=c)
        res[f"{tag}|{N}"] = r
        json.dump(res, open("balance_scaling.json", "w"))
print("DONE", flush=True)
