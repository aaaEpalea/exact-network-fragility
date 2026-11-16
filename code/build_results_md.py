"""Packages the project and regenerates RESULTS.md (tables are rendered directly from the JSON result files).
Usage:  python build_results_md.py        (run from the directory that holds the .py/.json files)
Add new experiments by appending to ENTRIES below, then re-run."""
import json, math, os, shutil, glob

SRC = os.environ.get("SRC", ".")
OUT = os.environ.get("OUT", "/mnt/user-data/outputs/project")
FIG_SRC = os.environ.get("FIG_SRC", "/mnt/user-data/outputs")


# ------------------------------------------------------------------ table rendering helpers
def fmt(x):
    if isinstance(x, bool) or x is None or isinstance(x, str): return str(x)
    if isinstance(x, int): return str(x)
    if isinstance(x, float):
        if math.isnan(x): return "nan"
        if math.isinf(x): return "inf"
        if x == 0: return "0"
        if abs(x) < 1e-3 or abs(x) >= 1e4: return f"{x:.2e}"
        return f"{x:.4g}"
    return str(x)


def is_num(s):
    try: float(s); return True
    except Exception: return False


def curve_s50(d):
    """dict {sigma: [acc,...] or acc}  ->  noise level at which mean accuracy first falls below 0.5."""
    xs = sorted(d, key=float); ys = []
    for k in xs:
        v = d[k]; ys.append(sum(v) / len(v) if isinstance(v, list) else v)
    xs = [float(k) for k in xs]
    if ys[0] < 0.5: return float("nan")
    for i in range(1, len(xs)):
        if ys[i] < 0.5:
            f = (ys[i - 1] - 0.5) / (ys[i - 1] - ys[i]); return math.exp(math.log(xs[i - 1]) + f * (math.log(xs[i]) - math.log(xs[i - 1])))
    return float("inf")


def collapse(o):
    """Replace accuracy-vs-noise curves by their sigma50; lists of numbers by their mean (short lists kept)."""
    if isinstance(o, dict):
        if len(o) >= 6 and all(is_num(k) for k in o) and all(isinstance(v, (list, int, float)) for v in o.values()):
            return {"s50(curve)": curve_s50(o)}
        return {k: collapse(v) for k, v in o.items()}
    if isinstance(o, list):
        if all(isinstance(v, (int, float)) for v in o):
            if len(o) <= 4: return ", ".join(fmt(v) for v in o)
            return f"[{len(o)} values, mean {fmt(sum(o) / len(o))}]"
        return {str(i): collapse(v) for i, v in enumerate(o)}
    return o


def flatten(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flatten(v, path + (str(k),))
    else:
        yield path, o


def table(obj, row_depth=None, max_cols=9, max_rows=70):
    leaves = list(flatten(collapse(obj)))
    for rd in ([row_depth] if row_depth else [1, 2, 3]):
        rows, cols = {}, []
        for p, v in leaves:
            if len(p) <= rd: r, c = " / ".join(p[:-1]) or "value", p[-1]
            else: r, c = " / ".join(p[:rd]), " / ".join(p[rd:])
            rows.setdefault(r, {})[c] = v
            if c not in cols: cols.append(c)
        if len(cols) <= max_cols or rd == 3: break
    if len(cols) > max_cols:                                   # fall back to a two-column listing
        out = ["| path | value |", "|---|---|"] + [f"| {' / '.join(p)} | {fmt(v)} |" for p, v in leaves[:max_rows * 3]]
        return "\n".join(out) + ("\n\n*(truncated; see the JSON file for everything)*" if len(leaves) > max_rows * 3 else "")
    lines = ["| " + " | ".join(["item"] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for r, d in list(rows.items())[:max_rows]: lines.append("| " + " | ".join([r] + [fmt(d.get(c, "")) for c in cols]) + " |")
    if len(rows) > max_rows: lines.append(f"\n*(showing {max_rows} of {len(rows)} rows; see the JSON file)*")
    return "\n".join(lines)


# ------------------------------------------------------------------ the experiments (id, title, code, results, explanation)
E = []
def entry(id_, title, code, results, text, **kw): E.append(dict(id=id_, title=title, code=code, results=results, text=text, kw=kw))

SECTIONS = {
    "A": "Audit: compiled programs are fragile",
    "B": "Mechanism and scaling with sequence length",
    "C": "Hardening Tracr-compiled models",
    "D": "Digital (restoring-state) designs",
    "E": "Compiler and certificates",
    "F": "Zeros / crosstalk and composition",
    "G": "Trained-transformer comparisons and hybrids",
    "H": "Wild ideas",
}

entry("A1", "Tracr programs under Gaussian weight noise (baseline fragility)", ["run_sweeps2.py", "fragility.py"], ["results2.json"],
      "Six Tracr programs compiled at N=8. `abs` = accuracy-vs-noise curve with noise on every weight (sigma50 = noise level where exact-sequence accuracy falls below 50%); `zero_only` / `nonzero_only` add noise only to the entries that are zero / nonzero in the compiled weights. Compiled weights are very sparse (0.3-6% nonzero) with values from 1 to 100. Noise of ~1e-4 to 1e-3 already breaks most programs; noise on the zero entries alone gives almost the same failure as noise on everything.")
entry("A2", "Compiled vs trained transformers, scale-free noise", ["trained_baseline.py", "compare_noise.py", "noln.py"], ["compare.json", "noln.json"],
      "alpha50 = relative noise (sigma = alpha x RMS of each tensor) at which accuracy falls below 50%. Compiled models fail at alpha around 3e-3 to 6e-3, trained transformers of the same depth at 0.09 to 0.28 (roughly 15-170x more tolerant). `noln.json` repeats the trained models without LayerNorm (not the explanation).")
entry("A3", "Which programs are fragile? static analysis + noise", ["audit_counting.py"], ["audit_counting.json"],
      "Every fragile program contains an attention readout (counting via SelectorWidth, or plain numerical averaging). The program without attention (`map`) is robust. Averaging-only programs (frac_prevs, pair_balance) are about as fragile as hist, so the accurate label is 'precise analog readout from attention averages', with counting the extreme case. Correlational evidence; `mean_token` failed to compile.")
entry("A4", "Control: programs with and without counting", ["control.py"], ["control.json"],
      "The per-token lookup program (no attention) has sigma50 around 0.11-0.15, 100-500x higher than the counting programs, and barely decays with N. The attention-without-counting control (shift by one) did not compile in Tracr.")

entry("B1", "Counting robustness vs max sequence length N (length program)", ["scaling.py"], ["scaling.json"],
      "sigma50 falls from 1.4e-3 (N=4) to 4.4e-4 (N=8). For N>=12 the default compiler is not even exact at zero noise (clean accuracy 0.82, 0.60, 0.39 at N=12, 16, 24), caused by the fixed discretizer sharpness (mlp_exactness=100). Attention-scale c=0.25 gives a roughly constant 3x.")
entry("B2", "mlp_exactness (S) needed vs N", ["scaling2.py"], ["scaling2.json"],
      "Exactness is restored by raising S to roughly N^2 (S=300 works to N=16, S=1000 at N=24). Even then sigma50 keeps falling (about N^-1.5); over-sharp S slightly reduces tolerance.")
entry("B3", "Scaling for reverse / hist / sort_unique / frac_prevs", ["scaling3.py"], ["scaling3.json"],
      "Tolerance falls with N for all programs at different rates (log-log slopes about -0.4 to -1.2 at default attention). The attention-scale gain shrinks with N (reverse: 3.4x at N=8, 1.6x at N=16) and frac_prevs loses exactness at c=0.25, N=16.")
entry("B4", "Component diagnosis: which weights limit the length program", ["diagnose.py"], ["diagnose.json"],
      "sigma50 with noise on one tensor group at a time. Embeddings alone nearly equal the all-weights tolerance; attention key and value paths are next; MLP input weights barely matter. Embedding slope about -1.1 and value path about -1.5 match a simple analysis of the count being stored as 1/(n+1).")
entry("B5", "Chunked counting vs plain counting", ["chunk_count.py", "chunk_big.py", "diagnose_chunk.py"], ["chunk_count.json", "chunk_big.json", "diagnose_chunk.json"],
      "Counting inside windows of 4 positions then adding exact integers improves the attention-path components 4-7x but the extra MLP layers become the new bottleneck (mlp_out), so overall tolerance gains only ~1.2x at large N and the trend with N is unchanged. `diagnose_chunk.json` holds the component breakdown (N=16 and 32 complete).")

entry("C1", "Hand-hardening stack on Tracr models", ["hardening.py", "run_hardening2.py"], ["hardening2.json"],
      "Variants applied to compiled weights: exact q/k + MLP rescaling (`balance`), attention scale `c`, snap layers, idealized weight-free clip. Rescaling alone gives only 1.1-1.3x; dense snap layers HURT (0.4-0.6x) because they add noisy parameters; lowering the attention scale gives 3-4x; combined 4-8x. `abs` = dense noise, `nz` = noise on nonzero entries only.")
entry("C2", "Noise-aware fine-tuning from the compiled initialization", ["run_noise_train.py"], ["noise_train.json"],
      "Fine-tuning compiled weights with injected noise. From the raw compiled weights it destroys exactness (e.g. reverse N=8 only 23,185 of 335,922 sequences correct); from the rescaled c=0.25 initialization it keeps exactness (exhaustively verified at N=8) and gives 4-19x. Densifies the weights (84-92% nonzero).")
entry("C3", "Sensitivity-balanced compilation (symmetry rescaling under noise)", ["balance.py", "balance_scaling.py"], ["balance_scaling.json", "balance_32_plain.json"],
      "Optimizes only exact symmetries (residual-dimension scales, q/k, v/out, MLP hidden scale) under injected noise, so the function is unchanged (clean accuracy stays 1.00, max output change ~5e-4). Gain is a constant 2.8-3.8x across N=16-48; the log-log slope moves only from -1.6 to -1.4. The chunked N=48 run did not finish.")
entry("C4", "Balancing generalized to other programs", ["balance2.py"], ["balance2.json"],
      "reverse, hist, sort_unique, frac_prevs at N=8-24. Balancing alone 2.7-6.3x; with attention scale c=0.25 the stack reaches 7-13x over default on five of six settings (frac_prevs 2.7x). The decay with N is unchanged.")
entry("C5", "Certificate-driven training with a general interval bound (negative result)", ["ibp_train.py"], ["ibp_train.json"],
      "Training the digital network's weights to maximize a provable radius under a general (loose) interval bound. The bound certifies radius 0 for the compiled weights and training never reached a certified radius; trained weights lost exactness on long sequences (length-400 accuracy 0.53 and 0.81) and tolerance got worse. NEGATIVE.")

entry("D1", "Same task, different designs: scaling of noise tolerance", ["digital_counter.py", "lsrl_counter.py"], ["digital_counter.json", "lsrl_counter.json"],
      "Counting tokens: hand-built digital one-hot counter with per-step snapping (slope ~-0.36) vs no snapping (~-0.93) vs LSRL linear-state accumulator (~-1.96, 1/N^2) vs the Tracr attention counter (~-1.3 to -1.6). Toy designs; not directly comparable tasks.")
entry("D2", "Prefix-histogram: digital vs Tracr (causal)", ["digital_prefix_hist.py"], ["digital_prefix_hist.json"],
      "Hand-built digital design with snapping is nearly flat in N (about 1.3-1.6e-2 from N=16 to 128); removing snapping gives about N^-1.7; Tracr causal is ~1e-4 at N=32. The digital design is larger (parameters grow as N^2).")
entry("D3", "Second program: majority-so-far", ["second_program.py"], ["second_program.json"],
      "Signed one-hot counter with comparison readout. Digital with snapping decays slowly (6.5e-2 to 2.1e-2, N=8-64) and is exact on all 7,776 length-5 sequences; Tracr causal is 1.0e-3 at N=8 and below the smallest tested noise at N=32. Here digital is also smaller (about 44k vs 505k parameters at N=32).")
entry("D4", "Width-matched redundancy baseline for Tracr", ["width_matched.py"], ["width_matched.json"],
      "r independent compiled Tracr copies with majority-style output summing, sized to match the digital design's parameter count. Redundancy helps (2x at N=16, 6.6x at N=32) but the digital design remains 11-28x better at equal parameters.")
entry("D5", "Digital design under the relative (RMS) noise protocol", ["digital_rel.py"], ["digital_rel.json"],
      "alpha50 of the digital prefix-histogram: 0.107, 0.081, 0.072, 0.061 for N=8, 16, 32, 64, comparable to trained transformers (0.16-0.20 at N=8, 0.034-0.037 at N=32 in the early under-trained runs).")
entry("D6", "Margin tuning of the digital designs (negative)", ["digital_tune.py"], ["digital_tune.json"],
      "Snap thresholds and a global state scale, all exact-preserving. Best gain 1.0x (prefix-hist) and 1.1x (majority): the defaults are already near-optimal for these knobs. NEGATIVE.")
entry("D7", "Weight-free clip activation instead of four dense snap matrices", ["clip_activation.py"], ["clip_activation.json"],
      "Restoring step s' = clip(2z - 0.5, 0, 1) as an activation (no noisy parameters). 20-25% fewer parameters, still exact, and 1.7-1.9x higher sigma50 (1.2-1.5x in relative alpha50).")
entry("D8", "Bilinear (second-order) design and weight-tied composition", ["ideas_round.py", "bilinear_lib.py"], ["ideas_round.json"],
      "Bilinear: z = sum_v x_v W_v s with no hidden gates, biases or snap matrices; relative tolerance alpha50 rises from about 0.1 to 0.6-1.0 and stays nearly flat to N=64. Tied: k template programs share one set of transition matrices plus a tiny role table; 30 exact programs fit in about 4,000 parameters with flat per-head tolerance (joint tolerance levels off at about 5e-2).")
entry("D9", "Tied + bilinear combined", ["tied_bilinear.py"], ["tied_bilinear.json"],
      "Role-gated multiplicative update, no hidden gates and no biases. About 3,400-3,900 parameters for k=1-30 exact programs; per-head sigma50 about 0.12 regardless of k.")
entry("D10", "Matched comparison: bilinear designs vs trained GRUs", ["matched_gru.py"], ["matched_gru.json"],
      "Same programs, token distribution, eval sequences and noise protocols (sigma50 absolute, alpha50 relative); GRUs have 32 or 64 hidden units, 2 seeds each, 2,500 training steps. Majority: bilinear is below the GRUs at N=16 (sigma50 0.12 vs 0.15-0.20) but above them at N=32 (0.115 vs 0.07-0.11, where the GRUs decay and are 99.5% accurate). Dyck: GRUs are 1.2-1.9x higher in sigma50 (N=16: 0.29-0.38 vs 0.24; N=32: 0.23-0.26 vs 0.14) and similar or higher in alpha50. So the bilinear compiled design is competitive with small trained GRUs, not uniformly better. Single token distribution per task; 6 noise draws.")
entry("D11", "Trained GRU baselines", ["trained_gru.py", "gru_long.py"], ["trained_gru_128.json", "trained_gru_long_256.json"],
      "GRU(128) trained 3,000 steps on prefix-hist and majority; prefix-hist is not learned exactly at N=32 (clean 0.58-0.71) in that budget, majority is exact at all N and more tolerant than the early digital designs. `trained_gru_long_256.json`: GRU(256), 7,000 steps, prefix-hist N=16 and 32 reach 0.995 and 0.98 accuracy with alpha50 0.43 and 0.20.")

entry("E1", "Compiler prototype: exactness certificates and empirical tolerance", ["digital_compiler.py", "run_compiler_tests.py", "exhaustive.py"], ["compiler_tests.json"],
      "FSM-bundle specs compiled to the original ReLU-snap digital network. `exact_cert` = verified (state, token) transitions out of total; valid for sequences of any length by induction. `certified_radius` = interval-bound certified worst-case parameter perturbation (any length). Empirical tolerance (rows starting `emp|`) decays about N^-0.45. Prior art: Omlin and Giles (1996) compile automata into recurrent nets with stability proofs, so this is not a new compiler concept.")
entry("E2", "Attacking the noise certificate", ["cert_attack.py", "cert_attack2.py", "cert_attack3.py"], ["cert_attack.json", "cert_attack2.json", "cert_attack3.json"],
      "Gradient (PGD) attack on parameters and tube states against the certified (eps, tube) pairs. `cert_attack3.json` rows: attack violation at multiples of the certified radius (negative = guarantee holds). All programs hold at 1x and break at 1.1x, so the ReLU-snap certificate is tight. (An earlier attack with a mismatched tube width suggested the opposite; superseded.)")
entry("E3", "Certified radius vs number of states (ReLU-snap design)", ["cert_scaling.py"], ["cert_scaling.json"],
      "Refined radius (tube 0.03): majority 3.28e-3 -> 1.47e-3 and dyck 3.46e-3 -> 2.11e-3 for N=8 -> 64; scales roughly D^-0.4 when all entries may be perturbed.")
entry("E4", "Zero-protected certificate (ReLU-snap design)", ["zeros_cert.py"], ["zeros_cert.json"],
      "If exactly-zero weights stay zero, the certified radius is constant in network size (4.03e-3 majority, 3.92e-3 dyck, N=8-64) while the all-entries-noisy radius falls. Attack check: holds at 1x, breaks at 1.1x.")
entry("E5", "Certificates for the bilinear and weight-tied designs", ["cert_bilinear.py", "bilinear_lib.py"], ["cert_bilinear.json"],
      "Interval-bound certificates (exact for any length; the tied certificate is valid for any number of instances k) plus PGD attacks. Radii are 15-40x larger than the ReLU-snap design. Caveats: the zero-protected value 0.1 may be limited by the tube grid and the bilinear-majority certificate could not be broken by the attack even at 1.5x (conservative there); dyck N=16 and tied N=16 are close to tight.")

entry("E6", "Tightened bilinear/tied certificates and function-level adversarial attack", ["cert_bilinear2.py", "cert_bilinear.py", "bilinear_lib.py"], ["cert_bilinear2.json"],
      "`fine_eps` = largest certified per-parameter radius when the tube width is searched finely (supersedes the coarse values in E5). Bilinear dense: majority 0.068/0.057/0.046 and dyck 0.077/0.067/0.057 for N=8/16/32; zeros-protected: exactly 0.100 for every size (an analytic limit from the saturating rows of the shift matrices, not a grid artifact); tied: dense 0.043/0.036 and protected 0.076 (valid for any k). `funcattack` rows: gradient attack on a single fixed perturbation (same at every time step) over length-4N sequences; `min_margin_found` is the smallest correct-minus-best-other output margin found (about 0.86-0.90 at the certified radius means the network is still correct; 0 or negative means a real wrong output). A real wrong output is found at 2x the certified radius for all dense cases and at 2x-3x for zeros-protected ones, so the certificate is within about 2-3x of a demonstrated counterexample (the true radius lies between the certificate and that point). Attack search is limited, so these are upper bounds on the true radius.")

entry("E7", "Function-level attack: bilinear N=32 and weight-tied design with k = 1, 4, 15", ["func_attack_round2.py", "cert_bilinear2.py"], ["func_attack_round2.json"],
      "Same single-fixed-perturbation attack as E6 on length-4N sequences, extended to N=32 and to the tied design (one shared perturbation of the shared matrices and role tables, k instances at once). `min_margin_found` at 1x the certified radius is 0.89-0.92 in all ten cases (network still correct). Real wrong outputs are found at 2x the certified radius for every dense case (margins -19 to -40) and at 2x (tie, margin 0.000) or 3x for the zeros-protected cases; results for k=1, 4 and 15 are identical, so the tied certificate and its tightness do not depend on k. Upper bounds only: the attack search is limited, ties count as failures. (The first run of the k=15 cases died and was resumed with the same script.)")

entry("F1", "How much fragility comes from the zero entries?", ["zeros_empirical.py"], ["zeros_empirical.json"],
      "sigma50 with noise on all / only zero / only nonzero entries for digital designs, composites, and Tracr hist. Protecting zeros gives about 1.5-2.9x (digital) and 1.9-2.3x (Tracr, roughly constant in N): zeros are not what drives Tracr's decay with length.")
entry("F2", "Parallel composition of k compiled programs (ReLU-snap design)", ["composition_exp.py"], ["composition_exp.json"],
      "k programs in one dense network: all composites exact (certificate 1224/1224 at k=8), parameters grow roughly as k^2, joint tolerance falls about k^-0.65.")
entry("F3", "Composition: per-head tolerance and composite certificates", ["comp_perhead.py", "comp_cert.py"], ["comp_perhead.json", "comp_cert.json"],
      "With zeros protected, per-head sigma50 is flat in k (7.6e-2 to 8.7e-2) and the certified radius is the same for k=1, 2, 4 (about 3.9e-3 to 4.0e-3); with dense noise both fall. The decay of the joint (all-heads-exact) number is largely the statistical cost of requiring every head to be right.")
entry("F4", "Composing trained GRUs: merging vs stacking", ["gru_merge.py"], ["gru_merge.json"],
      "Weight averaging destroys both trained programs (accuracy 0.04 and 0.15, printed in the log). Block-diagonal stacking is exact with no tolerance cost (alpha50 about 1). Block-diagonal composition is easy in any architecture; the open problem is shared-capacity composition.")

entry("G1", "Noise-aware training of a trained transformer (hist)", ["noise_aware_hist.py"], ["noise_aware_hist.json"],
      "N=16: alpha50 0.196 at clean accuracy 0.967 (baseline about 0.075 at 0.993); N=32: clean accuracy only 0.68. It trades exactness for tolerance and does not fix the scaling. One seed.")
entry("G2", "Hybrid transformer + frozen digital counters", ["hybrid_transformer.py", "hybrid2.py", "retrain_baselines.py"], ["hybrid_transformer.json", "hybrid2.json", "retrain_baselines.json"],
      "hist task. Hybrid reaches clean accuracy 1.000 at N=16, 32, 64 with alpha50 0.068, 0.046, 0.034; the transformer alone, properly trained (8,000 steps), reaches 0.995 / 0.993 / 0.875 with alpha50 0.08-0.086 / 0.046-0.048 / 0.036. So the hybrid's lasting benefit is exactness at long lengths, not tolerance. `hybrid2.json`: reverse at N=32 not learned by either model in 3,000 steps (inconclusive).")

entry("H1", "Wild ideas: replication, superposition, noise-trained bilinear", ["wild_round.py"], ["wild_round.json"],
      "Replication of every state unit with unscaled weights gives about sqrt(r) tolerance gain at r^2 parameters (inefficient). Superposition codes need d of about 96-192 dimensions to decode 51 one-hot states exactly: no compression (negative). Noise-injection training of the bilinear weights broke exactness (65% of one-step transitions correct, length-400 accuracy 0.23) and did not raise tolerance (negative).")


entry("H2", "Error-correcting sparse state codes (Willshaw transitions) and a shared-memory universal interpreter", ["willshaw_fsm.py"], ["willshaw_fsm.json"],
      "(A) Each state is a sparse constant-weight codeword (pairwise overlap <= 1), transitions are binary Willshaw associations recalled by thresholding; every transition was verified exactly. It needs MORE parameters than the one-hot bilinear design (2-15x more: e.g. dyck N=64, n=66 states: 55.6k vs 26.3k), found no exact code for majority at N=32 and 64 within code length 128, and gives similar sigma50 (0.08-0.11 vs 0.11-0.20) with lower relative alpha50. (B) Several DIFFERENT programs (majority, dyck, mod counters, random FSMs with 48 states) stored in ONE associative memory with a program code in the key: no exact shared memory was found up to 256 state units for P = 2, 4, 8 programs, and a memory of that size would exceed the separate one-hot tables (27k, 42k, 83k parameters). NEGATIVE at these sizes. Caveats: random codes, weight 3, 4-5 seeds, strict exactness requirement; theory suggests some compression is possible only for much larger state spaces or with optimized codes.")

OTHER_NOTES = {
    "noise_aware_transformer.json": "earlier noise-aware transformer run (superseded by G1)",
    "composition_tests.json": "earlier composition tests (superseded by F2/F3)",
    "balance_32.json": "early balancing run at N=32 (chunk design), see C3",
    "balance_2.json": "",
}


# ------------------------------------------------------------------ build
def main():
    os.makedirs(OUT + "/code", exist_ok=True); os.makedirs(OUT + "/results", exist_ok=True); os.makedirs(OUT + "/figures", exist_ok=True)
    for f in glob.glob(os.path.join(SRC, "*.py")) + glob.glob(os.path.join(SRC, "*.log")):
        if os.path.basename(f) in ("build_results_md.py",) or f.endswith(".log"): continue
        shutil.copy(f, OUT + "/code/")
    shutil.copy(os.path.join(SRC, "build_results_md.py"), OUT + "/code/")
    for f in glob.glob(os.path.join(SRC, "*.json")) + glob.glob(os.path.join(SRC, "*.csv")): shutil.copy(f, OUT + "/results/")
    for f in glob.glob(os.path.join(FIG_SRC, "*.png")): shutil.copy(f, OUT + "/figures/")
    used = {r for e in E for r in e["results"]}; lines = []
    lines += ["# Results consolidation", "",
              "Each entry lists the **code file(s)** that produced it, the **result file(s)**, a **table rendered directly from the result file**, and an explanation.",
              "Tables are regenerated by `code/build_results_md.py`; metric definitions: *sigma50* = absolute per-parameter Gaussian noise at which exact-sequence accuracy falls below 50%; *alpha50* = same with noise proportional to each tensor's RMS; *s50(curve)* = sigma50 computed from a stored accuracy-vs-noise curve. Folder layout: `code/`, `results/`, `figures/`.", "",
              "## Status of the main claims (updated as results change)", "",
              "| Claim | Status |", "|---|---|",
              "| Compiled Tracr programs are 15-170x less noise-tolerant than trained transformers | supported (A2) |",
              "| Fragility comes from analog readout from attention averages (counting is the extreme case) | supported, correlational across programs (A3, A4, B4) |",
              "| Default Tracr silently loses exactness for long counting; sharpness >= ~N^2 fixes it | supported on `length` only (B1, B2) |",
              "| Trained transformers also lose tolerance with length | weaker than first claimed: about N^-0.6 to -0.8 when properly trained (G2) |",
              "| Digital restoring designs flatten the decay and come with exactness + noise certificates | supported for finite-state programs (D1-D3, E1-E5) |",
              "| Bilinear digital design reaches trained-GRU-level relative tolerance | provisional; matched comparison in progress (D8, D10) |",
              "| Weight-tied composition: 30 exact programs in ~4k parameters | supported for template instances only (D8, D9) |",
              "| Bilinear / tied designs carry certified radii 0.036-0.077 (dense) and 0.076-0.1 (zeros protected), within ~2-3x of a real counterexample | supported (E5, E6) |",
              "| Bilinear compiled design is competitive with small trained GRUs | supported on majority and dyck only; GRUs are higher on dyck (D10) |",
              "| Hybrid transformer + digital counter improves tolerance | NOT supported; improves exactness at long N only (G2) |",
              "| Certificate-driven training, noise-trained bilinear weights, superposition state sharing | negative results (C5, H1) |",
              "| Error-correcting sparse codes and a shared associative memory do not compress below one-hot tables at tested sizes | negative (H2) |", ""]
    for sec, name in SECTIONS.items():
        lines += [f"## {sec}. {name}", ""]
        for e in [x for x in E if x["id"].startswith(sec)]:
            lines += [f"### {e['id']}. {e['title']}", "", "**Code:** " + ", ".join(f"`code/{c}`" for c in e["code"]) + "  \n**Results:** " + ", ".join(f"`results/{r}`" for r in e["results"]), ""]
            for r in e["results"]:
                p = os.path.join(SRC, r)
                if not os.path.exists(p): lines += [f"*(missing result file {r})*", ""]; continue
                try: obj = json.load(open(p))
                except Exception as ex: lines += [f"*(could not read {r}: {ex})*", ""]; continue
                if len(e["results"]) > 1: lines += [f"`{r}`", ""]
                lines += [table(obj, **e["kw"]), ""]
            lines += [e["text"], ""]
    rest = sorted(os.path.basename(f) for f in glob.glob(os.path.join(SRC, "*.json")) if os.path.basename(f) not in used)
    lines += ["## Other result files (no consolidated entry yet)", "", "| file | note |", "|---|---|"] + [f"| `results/{f}` | {OTHER_NOTES.get(f, '')} |" for f in rest] + [""]
    figs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(OUT, "figures", "*.png")))
    lines += ["## Figures", ""] + [f"- `figures/{f}`" for f in figs] + [""]
    open(OUT + "/RESULTS.md", "w").write("\n".join(lines))
    open(OUT + "/README.md", "w").write("# Project archive\n\n- `RESULTS.md` consolidated results (code file, result file, table, explanation per experiment)\n- `code/` every script (plus `build_results_md.py`, which regenerates RESULTS.md)\n- `results/` every JSON result file\n- `figures/` PNG figures\n\nEnvironment: Python 3.12, `jax`, `optax`, `dm-haiku`, `tracr` (pip install from GitHub google-deepmind/tracr), `numpy`, `matplotlib`; LSRL (github.com/AleksandarPetrov/LSRL) is cloned into `LSRL/` for `lsrl_counter.py`.\n\nTo add an experiment: append an `entry(...)` in `code/build_results_md.py` and re-run it from the directory holding the scripts and JSONs.\n")
    print("wrote", OUT + "/RESULTS.md", "|", len(E), "entries |", len(rest), "unlisted result files")


if __name__ == "__main__":
    main()
