"""Task definitions for the LLM-synthesis experiment. They reuse the task semantics, token conventions and input distributions of the existing project:
 * 'length', 'hist', 'reverse'  = the Tracr lib programs used in fragility.py (vocabulary 1..6, BOS prepended, outputs read at positions 1..L, MAX_LEN 8, bidirectional attention allowed)
 * 'parity', 'majority8', 'dyck8' = the digital_compiler specs (parity_of_token0, majority(N=8), dyck_valid_prefix(N=8)); digital token t is token id t+1 here; causal streaming semantics
Token ids: BOS = 0, input tokens 1..6.  Sequences have length L = 1..7 (positions 1..L; position 0 = BOS)."""
import itertools, numpy as np
BOS = 0; VOCAB = [1, 2, 3, 4, 5, 6]; MAXPOS = 8
def _length(x): return [len(x)] * len(x)
def _hist(x): return [x.count(v) for v in x]
def _reverse(x): return list(x[::-1])
def _parity(x):
    c, out = 0, []
    for v in x: c += (v == 1); out.append(c % 2)
    return out
def _majority(x, N=8):
    n = 2 * N + 1; s = N; out = []
    for v in x: s = min(s + 1, n - 1) if v == 1 else max(s - 1, 0) if v == 2 else s; out.append(int(s > N))
    return out
def _dyck(x, N=8):
    dead = N + 1; s = 0; out = []
    for v in x:
        if s == dead: pass
        elif v == 1: s = min(s + 1, N)
        elif v == 2: s = s - 1 if s > 0 else dead
        out.append(int(s != dead))
    return out
TASKS = {
    "length":    dict(ref=_length,   kind="num", atol=0.5, d_out=1, causal=False, probs=None),
    "hist":      dict(ref=_hist,     kind="num", atol=0.5, d_out=1, causal=False, probs=None),
    "reverse":   dict(ref=_reverse,  kind="cat", atol=None, d_out=7, causal=False, probs=None),
    "parity":    dict(ref=_parity,   kind="cat", atol=None, d_out=2, causal=True,  probs=[0.3, 0.3, 0.1, 0.1, 0.1, 0.1]),
    "majority8": dict(ref=_majority, kind="cat", atol=None, d_out=2, causal=True,  probs=[0.3, 0.3, 0.1, 0.1, 0.1, 0.1]),
    "dyck8":     dict(ref=_dyck,     kind="cat", atol=None, d_out=2, causal=True,  probs=[0.3, 0.3, 0.1, 0.1, 0.1, 0.1]),
}
def sample(task, split="test", n_per_len=40, lens=(3, 4, 5, 6, 7)):
    """split: 'test' (seed 0, hidden evaluation set), 'dev' (seed 100, may be shown to the LLM in the feedback condition)."""
    t = TASKS[task]; rng = np.random.default_rng({"test": 0, "dev": 100}[split]); out = {}
    for L in lens: out[L] = [list(map(int, rng.choice(VOCAB, size=L, p=t["probs"]))) for _ in range(n_per_len)]
    return out
def exhaustive(task, lens=range(1, 8)):
    """ALL 6^L sequences, L = 1..7 (335,922 sequences): the clean-correctness gate."""
    for L in lens: yield L, np.array(list(itertools.product(VOCAB, repeat=L)), dtype=np.int32)
def expected(task, xs): return [TASKS[task]["ref"](list(map(int, x))) for x in xs]
def tokens_with_bos(xs): return np.concatenate([np.zeros((len(xs), 1), np.int32), np.asarray(xs, np.int32)], 1)
