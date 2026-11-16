## 1. Background and related work

### 1.1 Why parameter noise on exact networks

A neural network that is *exactly* correct on a finite computation is a different object from one that is *usually* correct on a data distribution. For such networks a natural question is how far the parameters can move before an output is wrong. Perturbation of the parameters is the physical situation in analog and in-memory hardware (programming errors, drift, noise), in quantization, and in model editing. It is also a clean mathematical stand-in for "how redundant, or how sharply tuned, is this realization of the computation". Two networks that compute the same function can differ completely in this respect, which is the central observation behind this report.

### 1.2 Compiled transformers

Tracr (Lindner et al., 2023) compiles programs written in RASP (Weiss et al., 2021) into transformer weights, with no training, as a laboratory for interpretability. Compiled weights are very sparse and use large values (1 to 100 in the models studied here), and the compiled models are exact by construction. We found no earlier study of how such compiled networks react to parameter noise (Part M). A hand-built-transformer write-up reports numerical instability of Tracr models beyond roughly 400 positions, which is consistent with our finding that the default compiler becomes inexact for counting at length 12 and above (Part B).

### 1.3 Finite-state automata in recurrent networks

Encoding a finite-state machine in a recurrent network is a classical topic. Giles et al. (1992) trained and extracted automata with second-order recurrent networks. Omlin and Giles (1996) gave a construction of deterministic finite-state automata in second-order recurrent networks with sigmoid units, using weights of plus or minus H, and proved that the encoding is stable for strings of arbitrary length, including conditions for networks whose non-programmed weights carry random noise. A US patent (5,706,400, NEC) describes fault-tolerant variants: relative perturbation of non-zero weights, stuck-at-zero faults, k-fold replication of weights or of the whole network with a majority vote, and distributed "n choose k" encodings. **The digital and bilinear designs, their any-length exactness argument, the zero-weight noise tolerance, replication and sparse-code variants in Parts D, E, F, H and I all overlap this prior work**, and we present them as reproductions and extensions with new measurements (certificates compared to attacks, comparison with compiled transformers and trained GRUs).

### 1.4 Fast weights and programmable transformers

Schmidhuber (1992) introduced fast-weight programmers: one network writes the weights of another. Schlag, Irie and Schmidhuber (2021) showed that linear transformers are fast weight programmers, derived a capacity limit for such memories and proposed a delta-rule write. Giannou et al. (2023) showed that a fixed, hand-programmed looped transformer can execute programs supplied as part of the input. Part I builds a "universal interpreter" that executes different programs from fast weights; its capacity limit, and the absence of a shared-capacity gain, are consistent with this literature.

### 1.5 Robustness certificates and sensitivity

Interval bound propagation (Gowal et al., 2018) is a standard tool for certifying robustness to *input* perturbations. Certification against *weight* perturbations also exists (Weng et al., 2020, AAAI), as does a body of work on noisy analog hardware (e.g. Zhou et al., 2020). Verification of recurrent networks mostly addresses input perturbations over finite unrolled sequences. For any-length parameter-perturbation certificates based on a restoring invariant we found no source other than the analytic conditions of Omlin and Giles.

### 1.6 Same function, different robustness

Dinh et al. (2017) showed that rectifier networks computing the same function can be made arbitrarily sharp by reparameterization, which motivated scale-invariant measures of flatness (e.g. Kwon et al., 2021; Andriushchenko et al., 2023; "minimum sharpness"). Our question is a discrete, exact-computation version of this: how much does robustness depend on the *construction method* and on *design choices* when the computation is identical.

### 1.7 LLMs writing network weights

Directly generating network parameters with a language model has appeared in early forms: NeuroGen trains an LLM to emit parameters; Drag-and-Drop LLMs map prompts to adapter weights; a recent blog-level "weight-space programming" claim has a general LLM read and write small-network weights from a plain-language goal; TracrBench has LLMs write RASP programs for Tracr (GPT-4-turbo was correct on 57 of 101 programs); torchwright is a compiler from a typed computation graph to transformer weights. We found no study comparing the parameter-noise robustness of such networks with other construction methods (Part M).

## 2. Methods common to all experiments

### 2.1 Computations and tasks

Programs are small and completely specified so that exactness can be verified.

* **Tracr programs** (Parts A to C): `reverse`, `sort_unique`, `hist` (count of each token), `length`, `frac_prevs` (fraction of previous tokens equal to the current), `sort_freq`; vocabulary of 6 tokens, a beginning-of-sequence token, and a maximum length N (default 8).
* **Digital / bilinear programs** (Parts D to I): counting, prefix histogram, majority-so-far (`majority(N)`: counter with saturation), bracket matching (`dyck(N)`: valid-prefix check), mod-counters, reset counters, parity, random finite-state machines.
* **LLM-synthesis tasks** (Parts J, K): `length`, `hist`, `reverse`, `parity`, `majority8`, `dyck8` with exact semantics given in the repository.

### 2.2 Exactness

An output sequence is correct only if *every* output position is correct (**exact-sequence accuracy**). Numeric outputs use an absolute tolerance of 0.5 (0.05 for `frac_prevs`). Clean exactness is verified exhaustively when the domain is small (all 335,922 sequences of length 1 to 7 over 6 tokens in Parts J, K) and by induction for the restoring-state designs (the one-hot invariant holds after every step).

### 2.3 Noise protocols

Noise is Gaussian and static unless stated: one draw is added to the parameters and held fixed through the whole sequence.

* **Absolute (`abs`)**: every parameter gets independent N(0, sigma^2) noise. **sigma50** is the sigma at which exact-sequence accuracy falls below 50%, interpolated on a log scale (grid 1e-5 to 1, 16 points, usually 5 seeds).
* **Relative / scale-free**: noise proportional to each tensor's root-mean-square (RMS) value (**alpha50**) or to each weight (`rel`, multiplicative).
* **Zero-only / non-zero-only**: noise only on entries that are exactly zero or non-zero in the clean network (separates "crosstalk" from "programmed weights").
* **Parameter groups**: embeddings, attention query/key, attention value/output, MLP input, MLP output (and unembedding for the Part J, K transformer family).
* **Fresh-per-step** and **activation** noise (Parts H, I): new noise at every step, or noise on the state, which are different mathematical problems from static parameter noise.

### 2.4 Certificates and attacks

* **Interval-bound certificate.** Given a tube around each valid state and a box of parameter perturbations, interval arithmetic proves that the next state stays in the tube. By induction this proves exactness for sequences of *any length*. The **certified radius** is the largest box for which the proof goes through.
* **Function-level attack.** A sign-gradient attack searches for one fixed parameter perturbation inside a box that produces a wrong output. The smallest box where it succeeds is an upper bound on the true robustness radius. The certified radius is a lower bound. Their ratio measures how tight the certificate is.
* For transformers (Parts J, K) the certificate is *sample-certified* (on the evaluation inputs), not a domain-wide proof.

### 2.5 Statistical practice

Most experiments use 3 to 5 noise seeds and one trained or constructed model per cell; differences below a factor of about 2 should not be over-interpreted, and in Part K we measured that identical constructions at different widths can differ by 2.3 times in sigma50. Pre-registered experiments are marked; everything else is exploratory.
