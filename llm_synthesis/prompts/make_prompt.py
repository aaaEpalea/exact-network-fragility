"""Renders the frozen prompts: python make_prompt.py  ->  prompts/rendered/{task}__{condition}__{category}__{feedback}.md"""
import os, inspect, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "harness"))
import tasks as T, family as F
DESC = {
 "length": "For every position i of the input sequence the output is the length L of the input sequence (a number; the BOS token is not counted). Numeric output: |y - L| <= 0.5 counts as correct.",
 "hist": "For every position i the output is the number of positions j (including i itself) whose token equals x[i] (a number). Numeric output: |y - target| <= 0.5 counts as correct.",
 "reverse": "The output at position i (1..L) is the token x[L+1-i], i.e. the input sequence reversed. Categorical output: the predicted token id is argmax over the 7 output logits (index = token id).",
 "parity": "Causal streaming task. The output at position t is the parity (0/1) of the number of tokens equal to 1 among x[1..t]. Categorical output over 2 classes: predicted class = argmax of the 2 logits. The model may use causal or bidirectional attention but the output at position t must depend only on x[1..t].",
 "majority8": "Causal streaming task. A counter starts at 8 and is saturated to the range [0,16]; token 1 increments it, token 2 decrements it, all other tokens leave it unchanged. The output at position t is 1 if the counter after x[1..t] is > 8 (more 1s than 2s so far), else 0. Categorical output over 2 classes (argmax). The output at position t must depend only on x[1..t].",
 "dyck8": "Causal streaming task (valid-prefix check). Token 1 is an opening bracket, token 2 a closing bracket, tokens 3..6 are neutral. The depth starts at 0 and is capped at 8 (token 1 gives min(depth+1, 8)); token 2 gives depth-1; if token 2 arrives at depth 0 the sequence becomes invalid forever (absorbing). The output at position t is 1 while the prefix x[1..t] is still valid, else 0. Categorical output over 2 classes (argmax). The output at position t must depend only on x[1..t].",
}
def src(fn): return inspect.getsource(fn)
def render(task, condition, category, feedback):
    t = T.TASKS[task]; cfgs = F.CONSTRAINED[task]
    fam = inspect.getsource(F.forward); shapes = F.__doc__.strip()
    ref = {"length": T._length, "hist": T._hist, "reverse": T._reverse, "parity": T._parity, "majority8": T._majority, "dyck8": T._dyck}[task]
    cond = (f"CONSTRAINED SYNTHESIS. You must use exactly this architecture (no other choice is allowed):\n```\n{json.dumps(cfgs)}\n```\nThis fixes every tensor shape ({F.n_params(cfgs, t['d_out'])} parameters in total)." if condition == "constrained" else
            f"UNCONSTRAINED SYNTHESIS. You choose the architecture yourself within these limits: n_layers <= {F.LIMITS['n_layers']}, n_heads <= {F.LIMITS['n_heads']}, d_model <= {F.LIMITS['d_model']}, d_head <= {F.LIMITS['d_head']}, d_mlp <= {F.LIMITS['d_mlp']}, causal = true or false, and at most {F.LIMITS['max_params']} parameters in total.")
    if category == "3b":
        deliver = ("DELIVERABLE (one Python file `submission.py`, numpy only):\n```python\nimport numpy as np\nCONFIG = dict(n_layers=..., n_heads=..., d_model=..., d_head=..., d_mlp=..., causal=...)\ndef build_params():\n    # return a dict {tensor_name: np.ndarray} containing EVERY tensor with the exact shapes above\n    ...\n```\n"
                   "You write the program that computes the weights; the weights must come from your own analytical construction (loops, formulas, structured fills are fine). Do NOT train, fit, search or optimise: no gradient descent, no random search, no least-squares fitting to enumerated inputs, no solver calls. Only numpy may be imported.")
    else:
        deliver = ("DELIVERABLE (one JSON document `weights.json`; every weight value must be written by you as a literal number in your answer):\n```json\n{\"config\": {\"n_layers\": 1, \"n_heads\": 1, \"d_model\": 16, \"d_head\": 8, \"d_mlp\": 32, \"causal\": false},\n \"tensors\": {\"embed\": {\"shape\": [7,16], \"default\": 0.0, \"entries\": [[3,0,1.0],[4,2,-0.5]]}, \"...\": {}}}\n```\n"
                   "`default` is the value of all entries not listed; `entries` lists [index..., value] for the others. You must include every tensor with its exact shape. Values must be literal decimal numbers that you wrote down yourself (no formulas, no code generating the numbers, no training, no fitting).")
    fb = ("FEEDBACK AND REVISION. You may revise your submission up to 3 times. After each submission you will be told: any validation errors, the accuracy on a development set of 200 sequences, and up to 3 failing development examples (input, expected output, your model's output). You will not see any other evaluation." if feedback == "clean_check" else
          "NO FEEDBACK. This is a single attempt: you will not receive any information about how your submission performs. Think carefully, verify your construction by reasoning, and submit one final answer.")
    return f"""# Construct a transformer that computes `{task}` (no training)

You are asked to construct the parameters of a small transformer that exactly implements a specified function. The parameters must be derived by your own reasoning, not obtained by training.

## The model family (fixed)
{shapes}

The forward pass is exactly this code (JAX):
```python
{fam}```
Token ids: BOS = 0 (always prepended at position 0), input tokens are 1..6 at positions 1..L, with 1 <= L <= 7. Outputs are read at positions 1..L (the output at position 0 is ignored). The output width is d_out = {t['d_out']}.

## The function
{DESC[task]}

Reference implementation (the ground truth, Python):
```python
{src(ref)}```

## Correctness requirement
Your network must be exactly correct on ALL 335,922 input sequences of length 1..7 over tokens 1..6 (every output position of every sequence). Nothing else is evaluated about correctness; only exactly correct submissions are accepted.

## Condition
{cond}

## Deliverable
{deliver}

## {('Feedback' if feedback == 'clean_check' else 'No feedback')}
{fb}

Report: the final deliverable, the number of parameters, and in a few sentences how the construction works.
"""
if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "rendered"); os.makedirs(out, exist_ok=True); n = 0
    for task in T.TASKS:
        for cond in ["constrained", "unconstrained"]:
            for cat in (["3b", "4"] if task in ("length", "parity") else ["3b"]):
                for fb in ["none", "clean_check"]:
                    open(os.path.join(out, f"{task}__{cond}__{cat}__{fb}.md"), "w").write(render(task, cond, cat, fb)); n += 1
    print(n, "prompts rendered")
