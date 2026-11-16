# Figure index

**Figure A1.**  (a) Exact-sequence accuracy collapses at 1e-4 to 1e-3 for six compiled programs (bands show seeds). (b) sigma50 with noise on all weights, only zero entries, or only non-zero entries. (c) sigma50 per parameter group (noise on one group at a time; darker is more fragile).

**Figure A2.**  Exact-sequence accuracy against relative noise for compiled (red) and trained transformers (blue, with LayerNorm solid, without dashed) on `hist`, `sort_unique` and `reverse`. Compiled models fail one to two orders of magnitude earlier.

**Figure B1.**  Left: tolerance of Tracr's attention counter falls roughly as N to the minus 1.5 (default sharpness, best exact sharpness, and with attention scale 0.25). Right: the default compiler silently becomes inexact for N of 10 and above; raising the sharpness restores exactness.

**Figure C.**  Hardening closes about 2 to 14 times of the 15 to 170 times gap: compiled (red), compiled plus hardening (orange: rescale, attention temperature, noise-aware fine-tune; the asterisk marks a case with 6 of 335,922 sequences wrong after fine-tuning) and trained from scratch (blue, 3 seeds).

**Figure D1.**  Same task (counting tokens), different designs: digital one-hot counter with per-step snapping stays high; a linear-state accumulator decays as 1/N squared; Tracr decays roughly as N to the minus 1.5.

**Figure D2.**  Prefix histogram: the digital design with snapping is nearly flat in N; removing the snapping makes it decay; Tracr causal models are about 1e-4 at N=32.

**Figure D8.**  Weight-tied design: the transition matrices (up, down, hold) are shared; each program only selects which role each token uses.

**Figure I1.**  First failure found by the attack, as a multiple of the certified radius, against the number of heads k, for the tied-bilinear and the ReLU-gated tied design (N=16 and 32, dense noise). 'Any head' and 'single head' budgets are nearly equal.

**Figure I2.**  Noise tolerance against parameter count for one-hot, replication and Willshaw codes on the majority program (N=8, 16, 32): static parameter noise, fresh activation noise and a single activation burst.

**Figure I3.**  Left: the minimum key width of one shared fast memory divided by the number of stored transitions, against the number of programs, for four sets of increasing diversity. Right: parameter accounting against the number of programs (mixed families).

**Figure I3b.**  The fast-weight interpreter: a key built from token, state and program address (Hadamard product of random codes) reads the fast transition memory F; the clip restores the state; readout and initial state are also fast memories.

**Figure I4.**  Noise tolerance for each perturbation target (slow, fast, both, fresh versions, activation) at 1, 4 and 16 programs, in the joint metric where every program must be exact.

**Figure I5.**  Certified entrywise radius of the interpreter shrinks as programs are added.

**Figure I5b.**  Certified radius against first failure found by the attack, for the tied designs, the interpreter and the LLM-built transformers. Every point lies on or above the diagonal; the vertical distance to it is the looseness of the certificate.

**Figure K1.**  Absolute sigma50 for every attempt, by task and condition: fresh run (dots), contaminated run (diamonds) and the Tracr compiled baseline (squares, shared tasks only).

**Figure L2.**  First-order L-infinity radius divided by the attack-found radius for the 12 development networks, with autodiff gradients (relu'(0)=0) and with the one-sided derivative (relu'(0)=1). A ratio of 1 means the first-order model predicts the attack exactly.

**Figure L5.**  Predicted against measured sigma50 for predictor v3 (left) and v4 (right) on three construction methods. v4 brings the LLM-built networks and the bilinear networks close to the diagonal; the compiled `length` and `hist` programs stay far above it (the predictor is too optimistic).

**Figure S1.**  Absolute noise tolerance (sigma50) for every network in the consolidated dataset, by construction method. Compiled and LLM-built transformers lie in the same broad range; restoring-state designs are an order of magnitude higher; the within-method spread is large.
