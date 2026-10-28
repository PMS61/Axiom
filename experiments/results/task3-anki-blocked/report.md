# Bayesian dynamic latent-state model: diagnostics and ablations

Real Anki observations: fixed disjoint user cohorts from local Parquet files. The provided train/test partition is preserved (holdout setting is ignored). Exact recorded elapsed_seconds / 86400 supplies gaps, with explicit bounds [1 second, 3650 days] rather than clipping. Item IDs are user-nested. No known parameter truth or recovery claims. See selection.json artifacts for selection/exclusion counts and source hashes.

Run `task3-anki-blocked`. Source: **anki**. 4 independent chains, 8000 iterations each; 4000 discarded warmup; no thinning.

Sampler: `blocked`; 5 MH proposals per decay sweep. Joint covariance and scale adapt only during warmup. Blocked effects use the exact Gaussian conditional under unchanged priors.

## Scope and model

This study targets `AnkiFirstRun/dynamic_retention_model_optimised.py`, the pure NumPy time-aware OU model, not the event-indexed AR(1) model used by the original recall demo. The optimized-named file actually contains the pre-Numba implementation; it is reused directly for FFBS, priors and collapsed MH.

`mu = b0 + alpha_user + beta_item + gamma_user * previous_outcome`; `rate = lambda * exp(-kappa * log(1 + previous_successes))`; `phi = exp(-rate * gap_days)`; `q = sigma2 * (1 - phi²)/(2 * rate)`. The first state has variance 1. A Bernoulli-logistic observation measures the latent state. This is mean reversion on the logit scale, not a guarantee of monotonically decreasing recall with time.

Full has all components. No-practice fixes kappa=0; no-hierarchy fixes ability/item effects to zero and uses one population practice slope; no-dynamics is hierarchical lagged logistic regression with the same effect priors, removing state persistence and diffusion (and hence decay parameters). The last contrast removes the entire dynamics block; it cannot isolate persistence from diffusion.

Priors are inherited: b0 and g0 N(0,100); group effects Gaussian with inverse-gamma(2,1) variances; diffusion inverse-gamma(2,0.5); log lambda N(log(0.05),1.5²), restricted to [1e-4,20]; kappa N(0,1), restricted to [-3,3]. The harness uses the reference `polyagamma` package instead of the repository's truncated-series approximation. This is a deliberate sampler change, so these timings and diagnostics are not a reproduction of an earlier unrecorded run.

## Protocol and diagnostic gate

Each chain has its own seed and dispersed initial lambda (0.005, 0.05, 0.5, 2), diffusion (0.05, 0.2, 1, 3), kappa (-1, 0, 0.5, 1.5) and effect values. MH proposal adaptation stops at warmup. All retained draws enter diagnostics; prediction uses at most 200 evenly spaced draws per chain, with equal weight across chains. Warmup traces are retained separately. Lambda proposal acceptance and kappa sweep-move fractions are in chain-metadata.json; the latter is not an acceptance rate.

Rank-normalized folded split R̂ < 1.01 and both bulk/tail ESS ≥ 100 × chain count (400 here) are required for every monitored scalar. Nonfinite diagnostics fail the gate. Diagnostics include all effects, learned scales, terminal latent states, and centered effect contrasts, not just decay parameters. Fixed ablated parameters are excluded. Passing does not prove convergence; failing blocks trustworthy posterior interpretation. The old within-single-chain split R̂ is not used. See [Stan diagnostics](https://mc-stan.org/learn-stan/diagnostics-warnings.html) and [ArviZ rank R̂](https://python.arviz.org/en/v0.23.4/api/generated/arviz.rhat.html).

Each user/item sequence has a chronological held-out suffix. Every variant sees identical training and test events. Prediction is prequential: an outcome is used only after its own prediction, to update the state and practice count before later predictions. Dynamic prediction inherits the branch's logistic-normal approximation and PG-mean filtering; it is approximate rather than an exact held-out Bayesian filter. Log loss, Brier score, MAE, AUC and calibration bias are reported. No held-out score selects hyperparameters.

Fit cost is summed process CPU across chains (ESS/sec uses this denominator), with summed per-chain elapsed time also reported. Parallel job wall time is recorded separately and includes analysis/plotting overhead. Prediction CPU is separate. These costs are machine-specific.

## Results

| Scenario / replicate | Variant | Log loss | Brier | AUC | Max R̂ | Min bulk ESS | Min tail ESS | CPU s | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| anki / 0 | full | 0.4348 | 0.1348 | 0.786 | 1.283 | 11.7 | 26.8 | 136.3 | FAIL |
| anki / 0 | no_practice | 0.4355 | 0.1354 | 0.781 | 1.018 | 95.5 | 284.1 | 119.5 | FAIL |
| anki / 0 | no_hierarchy | 0.5008 | 0.1597 | 0.690 | 1.033 | 109.0 | 153.8 | 117.0 | FAIL |
| anki / 0 | no_dynamics | 0.4532 | 0.1409 | 0.776 | 1.001 | 2284.9 | 3803.3 | 30.3 | PASS |
| anki / 1 | full | 0.5824 | 0.1634 | 0.345 | 1.069 | 71.3 | 196.5 | 66.0 | FAIL |
| anki / 1 | no_practice | 0.5881 | 0.1614 | 0.333 | 1.043 | 121.6 | 244.1 | 53.0 | FAIL |
| anki / 1 | no_hierarchy | 0.5252 | 0.1494 | 0.379 | 1.066 | 66.7 | 107.5 | 42.8 | FAIL |
| anki / 1 | no_dynamics | 0.6160 | 0.1820 | 0.518 | 1.004 | 1452.3 | 2676.3 | 21.0 | PASS |

## Controlled component contrasts

Delta = ablated log loss − full log loss: positive favors the full model. Intervals are paired user-cluster bootstrap percentiles (1,000 resamples). With few users they are unstable and do not include MCMC uncertainty or correct for multiple comparisons. Results remain exploratory when either model fails diagnostics.

| Scenario / replicate | Ablation | Delta log loss | 95% cluster interval | Interpretation |
|---|---|---:|---|---|
| anki / 0 | no_practice | 0.0007 | [-0.0057, 0.0072] | Convergence unresolved |
| anki / 0 | no_hierarchy | 0.0660 | [-0.0078, 0.1893] | Convergence unresolved |
| anki / 0 | no_dynamics | 0.0184 | [0.0010, 0.0392] | Convergence unresolved |
| anki / 1 | no_practice | 0.0057 | [-0.0431, 0.0534] | Convergence unresolved |
| anki / 1 | no_hierarchy | -0.0572 | [-0.2617, 0.0100] | Convergence unresolved |
| anki / 1 | no_dynamics | 0.0336 | [-0.0543, 0.2243] | Convergence unresolved |

## Recovery and identifiability

Real Anki observations have no known parameter truth. Recovery is not computed or inferred from fit quality.

The likelihood depends on b0+alpha+beta, so additive shifts in b0 and group effects are not separately likelihood-identified; proper priors regularize these directions without creating data information. Centered alpha/beta contrasts and population_level are reported alongside raw coefficients. Centering is post-processing, not a sampler reparameterization. User-nested item designs worsen ability/item separation; this cannot be fixed just by longer chains.

Lambda and kappa share log(rate)=log(lambda)−kappa×log(1+past_successes). A narrow practice-count range makes them hard to separate. Diffusion and decay both control transition variance, giving additional posterior dependence. Pair plots and correlations quantify these sample-specific ridges; they do not prove structural identifiability.

## Conclusions and next investigations

2 of 8 fitted variants pass the full diagnostic gate. No component is declared empirically justified from predictive point estimates alone. A component requires a converged controlled contrast, a credible predictive improvement, and replication on representative learner data. Synthetic matched recovery tests the implementation under assumed ground truth; it does not establish real-world usefulness.

Inspect chains for drift, poor rank overlap and long autocorrelation first. Extend warmup and retained draws using saved settings. If diagnostics remain poor, investigate blocked population/effect updates and joint decay proposals using the sampler selector. Non-centered random effects remain a separate unimplemented hypothesis. Use short-gap scenarios and multiple practice ranges to probe weak identification. Run more independent synthetic datasets and a prior-drawn SBC study before claiming calibrated inference.

No previously observed ESS/R̂ values are reconstructed without their original chain files and exact configuration. This run is new evidence. See provenance.json for revision, source hashes, dependency versions, seed and dataset hash; config.json and dataset files allow reruns. NPZ files preserve chain/draw axes and all retained parameters.

## Artifacts

![Comparison](comparison.png)



Per-fit trace/ACF, pooled rank and posterior-pair plots are alongside CSV diagnostics, predictions, draws, recovery summaries and chain metadata. Report status is never changed to converged just because computation finished.
