# Pre-registration: first-order (linear) predictor of parameter-noise failure levels
Written before any predictor value was computed. Date of writing: 2026-11-13.

**Data split.** Development set = contaminated run (J, 16 networks). Held-out set = fresh run (K, correct attempts of 3b/no-feedback and category 4). The fresh run is not looked at until the predictor and its constants are frozen.

**Predictor** (for a network with clean margins m_i > 0 on the evaluation samples i = (sequence, position), margin = logit gap for categorical outputs, atol - |error| for numeric outputs; g_i = gradient of m_i w.r.t. all parameters; w = parameter vector):
* absolute Gaussian noise: sigma_hat = min_i m_i / ||g_i||_2
* relative noise (delta_j = sigma w_j z_j): sigma_hat_rel = min_i m_i / ||g_i * w||_2
* non-zero-only / zero-only noise: restrict g_i to non-zero / zero entries
* parameter group G: restrict g_i to the group
* L-inf box: rho_hat = min_i m_i / ||g_i||_1   (to be compared with the certified radius and the attack-found radius)
The measured quantities are sigma50 (project definition), certified radius and attack first-failure from the harness.

**Fit (development only).** log10 sigma50 = a + b log10 sigma_hat, ordinary least squares over networks, separately per noise mode; also report b = 1 with a free offset only. The constants (a, b) are then frozen.

**Held-out test.** Apply the frozen constants to the fresh run. Report: Spearman rho, median absolute error in log10, fraction within factor 2 and 3, and every network that is off by more than a factor 5. Success criterion decided now: median absolute log10 error <= 0.3 (factor 2) and Spearman >= 0.8 on the absolute-noise mode. If the criterion is not met the predictor is reported as failed.

**Counterexample search (after the test).** Look for pairs of networks with similar sigma_hat but sigma50 differing by more than 5x.
