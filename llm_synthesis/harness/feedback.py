"""Feedback shown to the LLM in the clean_check condition (and ONLY this): validation errors, accuracy on the 200-sequence DEVELOPMENT set (seed 100), up to 3 failing dev examples.
usage: python feedback.py --task parity --submission ../submissions/x.py --condition constrained"""
import argparse, numpy as np, jax.numpy as jnp
import tasks as T, family as F, evaluate as E
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--submission", required=True); ap.add_argument("--condition", required=True); a = ap.parse_args()
try: cfg, params = F.load_submission(a.submission)
except Exception as e: print("LOAD ERROR:", repr(e)); raise SystemExit
errs = F.validate(a.task, cfg, params, a.condition)
if errs: print("VALIDATION ERRORS:"); [print(" -", e) for e in errs[:10]]; raise SystemExit
ev = E.FamilyEval(a.task, cfg, params); data = T.sample(a.task, "dev"); P = ev.params; t = T.TASKS[a.task]; tot = ok = 0; fails = []
for L, xs in data.items():
    out = np.asarray(F.forward(P, cfg, jnp.array(T.tokens_with_bos(xs))))[:, 1:]
    for x, o, e in zip(xs, out, T.expected(a.task, xs)):
        got = [int(v) for v in o.argmax(-1)] if t["kind"] == "cat" else [round(float(v), 2) for v in o[:, 0]]
        good = got == list(e) if t["kind"] == "cat" else bool(np.all(np.abs(np.array(got) - np.array(e)) <= t["atol"])); tot += 1; ok += good
        if not good and len(fails) < 3: fails.append((x, list(e), got))
print(f"development-set accuracy: {ok}/{tot}"); [print(f" failing example: input={x} expected={e} got={g}") for x, e, g in fails]
