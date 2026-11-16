# Pre-registration: predictor v4 = v3 with saturation margins added to the failure set (written before any v4 value was computed; this is the LAST predictor experiment)
v3 (tangent-cone Monte-Carlo, s = 1e-5, 150 draws, absolute Gaussian noise, project sigma grid) is unchanged except that a sequence now counts as failed if EITHER an output margin OR a saturation margin becomes non-positive:
* softmax attention (family and Tracr): for every (layer, head, query position >= 1) the clean top set S = {keys with score >= max - 0.01}; margin = min_{j in S} s_j - max_{j not in S} s_j (S kept fixed); if S is the whole allowed set the margin is +inf.
* clip units (bilinear): for every (step, state unit) with clean pre-activation z: on-units margin z - 0.75, off-units margin 0.25 - z.
No constants. Absolute noise only.
Success criterion (decided now), evaluated separately on: (a) LLM-built family, fresh run (35 attempts); (b) Tracr (6 tasks); (c) bilinear (12 programs): median |log10(pred/measured)| <= 0.3 and >= 2/3 of cases within 3x. v4 is declared successful only if (b) and (c) both meet it and (a) does not get worse than v3 (median error <= 0.3). Otherwise it is reported as failed and the predictor line of work stops.
Known risk stated in advance: set flips can be harmless, so v4 may under-predict robustness (too pessimistic).
