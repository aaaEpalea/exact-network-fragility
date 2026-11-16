import json, warnings
warnings.filterwarnings("ignore")
from chunk_count import *
res = {}
for N in [32, 48]:
    res[N] = {}
    res[N]["plain"] = evaluate(lib.make_length(), N, int(1.2 * N * N), "plain (S=1.2N^2)")
    res[N]["chunk4"] = evaluate(chunked_length(N, 4), N, 100, "chunked g=4")
    json.dump(res, open("chunk_big.json", "w"))
print("DONE", flush=True)
