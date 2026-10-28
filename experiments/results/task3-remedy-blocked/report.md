# Bayesian dynamic latent-state model: diagnostics and ablations

Run `task3-remedy-blocked`. Source: **synthetic**. 4 independent chains, 8000 iterations each; 4000 discarded warmup; no thinning.

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
| full / 0 | full | 0.6145 | 0.2130 | 0.641 | 1.058 | 80.3 | 335.9 | 108.6 | FAIL |
| full / 0 | no_dynamics | 0.6043 | 0.2075 | 0.660 | 1.001 | 11028.6 | 11866.9 | 24.2 | PASS |
| full / 1 | full | 0.6364 | 0.2230 | 0.710 | 1.021 | 180.8 | 372.6 | 106.5 | FAIL |
| full / 1 | no_dynamics | 0.6438 | 0.2267 | 0.671 | 1.001 | 10353.8 | 12217.1 | 20.4 | PASS |

## Controlled component contrasts

Delta = ablated log loss − full log loss: positive favors the full model. Intervals are paired user-cluster bootstrap percentiles (1,000 resamples). With few users they are unstable and do not include MCMC uncertainty or correct for multiple comparisons. Results remain exploratory when either model fails diagnostics.

| Scenario / replicate | Ablation | Delta log loss | 95% cluster interval | Interpretation |
|---|---|---:|---|---|
| full / 0 | no_dynamics | -0.0102 | [-0.0362, 0.0122] | Convergence unresolved |
| full / 1 | no_dynamics | 0.0074 | [-0.0163, 0.0312] | Convergence unresolved |

## Recovery and identifiability

Known truth is available. Matched generating/fitted variants yield 58 parameter intervals; 52 pass their parameter diagnostic gate; 52 of those contain the truth. These are dependent, few-replicate descriptive counts, not calibrated coverage or simulation-based calibration.

Recovery.csv contains truth, posterior mean, signed error, squared error, 95% interval, coverage and a parameter-level gate for all recoverable parameters. Recovery-summary.csv gives replicate bias, RMSE and coverage with separate convergence counts. Misspecified variants are flagged and must not be interpreted as parameter recovery under their own DGP.

The likelihood depends on b0+alpha+beta, so additive shifts in b0 and group effects are not separately likelihood-identified; proper priors regularize these directions without creating data information. Centered alpha/beta contrasts and population_level are reported alongside raw coefficients. Centering is post-processing, not a sampler reparameterization. User-nested item designs worsen ability/item separation; this cannot be fixed just by longer chains.

Lambda and kappa share log(rate)=log(lambda)−kappa×log(1+past_successes). A narrow practice-count range makes them hard to separate. Diffusion and decay both control transition variance, giving additional posterior dependence. Pair plots and correlations quantify these sample-specific ridges; they do not prove structural identifiability.

## Conclusions and next investigations

2 of 4 fitted variants pass the full diagnostic gate. No component is declared empirically justified from predictive point estimates alone. A component requires a converged controlled contrast, a credible predictive improvement, and replication on representative learner data. Synthetic matched recovery tests the implementation under assumed ground truth; it does not establish real-world usefulness.

Inspect chains for drift, poor rank overlap and long autocorrelation first. Extend warmup and retained draws using saved settings. If diagnostics remain poor, investigate blocked population/effect updates and joint decay proposals using the sampler selector. Non-centered random effects remain a separate unimplemented hypothesis. Use short-gap scenarios and multiple practice ranges to probe weak identification. Run more independent synthetic datasets and a prior-drawn SBC study before claiming calibrated inference.

No previously observed ESS/R̂ values are reconstructed without their original chain files and exact configuration. This run is new evidence. See provenance.json for revision, source hashes, dependency versions, seed and dataset hash; config.json and dataset files allow reruns. NPZ files preserve chain/draw axes and all retained parameters.

## Artifacts

![Comparison](comparison.png)

![Recovery](recovery.png)

Per-fit trace/ACF, pooled rank and posterior-pair plots are alongside CSV diagnostics, predictions, draws, recovery summaries and chain metadata. Report status is never changed to converged just because computation finished.
