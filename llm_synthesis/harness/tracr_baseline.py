"""Category 1 (compiler-generated) reference: the project's Tracr models for length / hist / reverse, evaluated through the SAME noise_suite as the LLM submissions."""
import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "../../code")
import numpy as np, jax, jax.numpy as jnp
import fragility as fr, evaluate as E
class TracrEval:
    def __init__(self, name):
        prog, kind, atol, samp = fr.build_programs()[name]; self.task = fr.Task(name, prog, kind, atol, samp); self.params = self.task.params; self.groups = list(fr.GROUPS)
    def acc(self, p): return self.task.accuracy(p)
    def group_pred(self, g, name): parts = name.split("/"); return fr.GROUPS[g]("/".join(parts[:-1]), parts[-1])
if __name__ == "__main__":
    out = {}
    for name in ["length", "hist", "reverse"]:
        ev = TracrEval(name); r = E.noise_suite(ev, seeds=5); r["params"] = E.param_stats(ev.params); r["clean"] = ev.acc(ev.params); out[name] = r
        print(name, "clean", r["clean"], "params", r["params"]["n_params"], {k: f"{v['s50']:.3g}" for k, v in r.items() if isinstance(v, dict) and "s50" in v}, flush=True)
    json.dump(out, open("../results/tracr_baseline_same_harness.json", "w"), indent=1, default=str)
