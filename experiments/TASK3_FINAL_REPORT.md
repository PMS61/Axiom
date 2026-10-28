# Task 3 final research report — 28 October 2026

Task 3 evaluates computation and inference for the Bayesian OU retention model.
It does **not** establish learner efficacy or validate every full-model posterior.
The bounded investigation is complete: all 12 prespecified sensitivity fits and
48 chains finished. Scientific success does not require every fit to converge,
every fixed truth to be covered, or a predictive advantage.

Tasks 1 (FSRS) and 2 (other baselines/statistical validation) are outside this
report. Their code, results and claims are not changed. This report consolidates,
rather than replaces, the preserved stage reports and raw experiments.

## Model and invariant decision rule

For a review history, x is the previous binary recall and C is
`log(1 + successes strictly before this review)`. With
`mu = b0 + alpha_user + beta_card + gamma_user x`, the latent model is

```
rate_t = lambda exp(-kappa C_t)
a_t = rate_t dt_t
phi_t = exp(-a_t)
q_t = sigma2 [1-exp(-2 a_t)] / (2 rate_t)
psi_t ~ Normal(phi_t psi_(t-1) + (1-phi_t) mu_t, q_t)
r_t ~ Bernoulli(logistic(psi_t)).
```

The initial latent distribution is `Normal(mu_0, 1)`. Its variance is **one**,
not the stationary variance, and it is independent of sigma2. Time is days.
The owning implementation is `AnkiFirstRun/dynamic_retention_model_optimised.py`;
the diagnostic harness uses reference `polyagamma` draws. Its experimental
innovation and coefficient/path transitions remain CLI-only.

Every monitored scalar must have **finite folded rank-normalized split Rhat
<1.01 and bulk AND tail ESS >=400**, using four independent chains. Undefined
statistics fail. No rounding, removed chains, selected parameters, thinning,
threshold adjustment or outcome-dependent extension qualifies a fit. The nested
full fits retain all 142 raw and derived monitors: global parameters, all group
effects and variances, terminal training states, centered effects and nested
likelihood combinations. Fixed parameters in labelled controls are constants,
not evidence for successful inference of the free full model.

Original priors and parameter support are unchanged in sampler comparisons:
log-lambda has its original normal prior and bounded support; kappa has its
original symmetric truncated normal support (-3,3); diffusion retains its
inverse-gamma prior; b0/g0 have N(0,100) priors; the three hierarchy variances
have IG(2,1) priors. The separately labelled MODEL analysis below changes only
the shared hierarchy prior scale and makes no sampler-improvement claim.

## Completed convergence and computational investigation

| Stage and design | Observed outcome | Interpretation |
|---|---|---|
| Initial crossed synthetic study: 5 conditions × 2 datasets × 4 model variants, 2400/1200 iterations/warmup | 0/40 whole-fit passes | Predictive/ablation results are exploratory |
| Longer matched first dataset, full and static, 8000/4000 | 0/2 passes | More draws alone did not establish convergence |
| Four sampler choices on two ordinary and one short-gap synthetic datasets, 8000/4000 | 0/12 full passes; blocked static passes on both ordinary datasets | Joint decay and coefficient blocking resolve some, not all, bottlenecks |
| Actual-data sampler pilot, two fixed disjoint cohorts | Full fits fail with the four earlier kernels | Real observations do not by themselves fix mixing |
| Dynamics ladder, four real/synthetic datasets, 32 fits | Static 4/4; fixed all dynamics 4/4; interwoven diffusion-only 4/4; interwoven free decay/diffusion 2/4; full 0/8 across centered/interwoven | These fixed controls change the model; 14/32 overall is not a full-model success rate |
| Coefficient/path interweaving, 12 fits/48 chains on the same four datasets | New kernel 3/4 full passes, centered 0/4, innovation-only 0/4 | Substantial full-model mixing improvement, with one unresolved full fit |

All later sampler studies use four chains, 8000 total iterations, 4000 warmup,
4000 retained draws per chain, and the same datasets/splits/seeds within each
matched comparison. Adaptation freezes after warmup. The coefficient/path phase
replayed **128 baseline parameter arrays exactly**, including available warmup.
The present sensitivity study replays the four coefficient references again;
its exact-replay evidence is published with the sensitivity results.

| Original-order / original-prior dataset | Coefficient/path max Rhat | Minimum bulk / tail ESS | Whole-fit gate |
|---|---:|---:|---|
| Real cohort 1 | 1.003671 | 793.5 / 1849.5 | PASS, chronology provisional |
| Real cohort 2 | 1.007600 | 600.8 / 637.9 | PASS, chronology provisional |
| Nested synthetic 1 | 1.005704 | 333.446827 / 1119.9 | FAIL: kappa bulk ESS |
| Nested synthetic 2 | 1.007968 | 641.5 / 1502.5 | PASS |

Coefficient/path interweaving is an exact Gaussian PG coefficient conditional
at fixed latent innovations and OU parameters, followed by path reconstruction
and the original hierarchy-scale conditionals. It retains gamma/g0 cross-prior
precision and the zero g0 likelihood column. Density, gradient, Gaussian-draw,
Jacobian and innovation-preservation checks support its target. See the full
[derivation and matched results](TASK3_COEFFICIENT_REPORT.md).

Minimum bulk ESS/CPU ratios relative to innovation-only are 4.37, 1.34, 0.85 and
0.97 respectively. Sampling costs increase about 1.28–1.37 times. These are
contemporary summed process-CPU comparisons, **not a universal efficiency win**.
Minimum ESS may correspond to different limiting parameters. Earlier timing
segments affected by interruptions or concurrent builds are explicitly qualified
in the stage reports. Neither wall time nor cost differences between changed
priors/orders establish a sampler advantage.

## Remaining decay/practice limitation: completed characterization

No additional unvalidated sampler was introduced, and the failed chain budget was
not extended. This investigation takes the explicitly allowed characterization
route. All original chains and failure diagnostics remain available.

Let eta=log(rate)=log(lambda)-kappa C. At two distinct C values the noiseless
rate map is injective in log(lambda),kappa. Thus a large sample correlation is
not proof of an exact structural symmetry of those parameters. The training
rate-design rank and practice ranges are quantified for all four datasets in
`decay-design-characterization.csv`. This design calculation does not establish
identifiability from binary outcomes with unobserved states.

The transition derivatives explain weak practical separation:
`d phi / d eta = -a exp(-a)` and
`d log q / d eta = 2a / (exp(2a)-1) - 1`. For small a, phi≈1-a and
q≈sigma2 dt: the path reveals little decay information and the mean's multiplier
1-phi is small. For large a, phi≈0 and q≈sigma2/(2 rate): diffusion and rate
can compensate. The initial state contributes no decay/diffusion information.
These asymptotics explain potential weak information; they are not a proof that
all finite datasets are non-identifiable.

For the failing synthetic 1 coefficient/path fit, each chain's kappa lag-1
correlation is 0.914–0.920 and lag-10 correlation 0.521–0.584. Within-chain
corr(log lambda,kappa) is 0.357–0.432. Chain means are approximately
0.116, 0.110, 0.190 and 0.189; the first chain's half means shift from -0.018 to
0.250. Kappa's pooled mean is 0.151, interval [-1.533,1.382], MCSE 0.0435,
bulk ESS 333.446827 and Rhat 1.005387. The low Rhat and adequate tail ESS do not
override the bulk failure. The interval includes both zero and fixed truth 0.6;
that does not validate inference. These summaries are descriptive because the
whole fit fails. All four chains, three autocorrelation lags, half means and
correlations with diffusion and mean slope are retained in
`decay-chain-characterization.csv`.

The weighted C spread uses weights `(a exp(-a))²` at marginal posterior medians
and is explicitly a **phi-Jacobian design proxy**, not observed/full Fisher
information. It is accompanied by unweighted rank/range and time-scale regime
fractions, so it cannot be used as a likelihood-identification certificate.
All four transition rate-design matrices have rank 2. At marginal posterior
medians, fractions with a<0.01 / a>10 are 22.2% / 24.5% (real 1),
22.2% / 3.8% (real 2), 22.5% / 16.7% (synthetic 1), and 22.6% / 3.8%
(synthetic 2). Synthetic 1 spans C=0–3.526, with phi-sensitivity-weighted C
variance 0.875; the failure cannot be attributed to literally constant practice.
These median-based regime counts characterize the tested design, not full
posterior information or the cause of each failed parameter.
The available evidence establishes residual autocorrelation and weak practical
information, not a proven sampling bug or a uniquely identified best remedy.
The negative result is final at this bounded budget: synthetic 1 full-model
inference remains unvalidated.

## Chronology: feasible-order posterior sensitivity

No authenticated producer ordering was recovered. The extract entered Git already
processed; the available Task 1 loader uses another source route and cannot
authenticate it. Local raw/cache paths checked in the previous audit remain
unavailable. Public schema resemblance cannot establish row correspondence.

The previous longest-gap-first audit eliminates timing residual discrepancies,
but does not select a unique chronology. A two-permutation audit found changed
outcome/lag/practice positions in 9/30 and 6/30 selected histories. This report
completes the missing **posterior** comparison on these exact cohorts:

1. Recorded order is retained as the reference.
2. Stable longest elapsed gap first within each card/day is a feasible candidate.
3. Keep that first event and reverse the remaining events on the same day for a
   second feasible candidate.

Both candidates satisfy the strict whole-history day-floor consistency check.
Every raw row is preserved with partition and original Parquet row index, source
ID mapping, dense ID, candidate permutation and original cut in the local raw
artifacts. Gaps stay attached to their rows; x and cumulative practice are rebuilt.
No candidate exchanges train/test rows or changes split counts. Recorded datasets
are independently reconstructed from the Parquet rows and compared exactly before
sampling. No cohort was widened, no observations clipped and no earlier result
replaced. Test predictions are realigned to original review identities for
probability comparisons; a changed final event changes psi_last's estimand.

The study is bounded to two candidates per cohort, each fitted with four chains
at 8000/4000 and all 142 diagnostics. It is **not** exhaustive ordering uncertainty,
model averaging, recovered ground truth, or evidence that the original timing
exclusions identified a uniquely clean sample. The recorded reference can itself
be infeasible under a day-floor timing interpretation; it remains a reproducible
reference, not an authenticated alternative.

### Measured order results

All six order fits pass every scalar gate. This establishes Monte Carlo
agreement at this budget for these specified histories, while authentic order
remains unknown. The four reference fits in this study (two real, two synthetic)
reproduce **all 64 saved retained/available-warmup parameter arrays exactly**.
All CPU comparisons below use the new common execution segment.

| Cohort | Order | Max Rhat | Minimum bulk / tail ESS | Held-out log loss | Gate |
|---|---|---:|---:|---:|---|
| 1 | Recorded | 1.003671 | 793.51 / 1849.51 | 0.433554 | PASS |
| 1 | Longest first | 1.004065 | 818.02 / 1553.74 | 0.462435 | PASS |
| 1 | Reverse remainder | 1.002928 | 845.96 / 1649.99 | 0.430961 | PASS |
| 2 | Recorded | 1.007600 | 600.78 / 637.90 | 0.576472 | PASS |
| 2 | Longest first | 1.007159 | 759.30 / 688.99 | 0.581020 | PASS |
| 2 | Reverse remainder | 1.003601 | 775.08 / 800.53 | 0.581317 | PASS |

Feasible-pair log-loss differences (reverse minus longest) are -0.031475 and
+0.000297. Thus the first cohort's prediction sensitivity is appreciable compared
with the preceding same-target sampler score differences; the second cohort's
aggregate score is much more stable. This is sensitivity, not a ranking of
chronologies or a component benefit. Longest-first changes event-aligned
probabilities from the recorded reference by mean absolute 0.02474 / 0.01060
and maximum absolute 0.38010 / 0.10835 in cohorts 1 / 2 respectively.

Posterior impact is also measured rather than inferred from predictor changes.
For cohort 1, reverse minus longest shifts mean gamma_mean by +0.12073,
population_level by -0.09148 and sigma2 by +0.02643. These are about
+0.187, -0.152 and +0.182 of the longest-order marginal posterior SDs: modest
relative to posterior uncertainty, but not explained solely by their reported
mean MCSEs (quadrature 0.01573, 0.01577, 0.00558 respectively). Shared seeds can
correlate estimates; conservative sums of marginal MCSEs are also retained
(0.02224, 0.02230, 0.00778). These are descriptive screens, not significance
tests, multiplicity-adjusted discoveries or a guarantee of MCSE accuracy.
Kappa's feasible-pair shift is -0.02119 / +0.04192 across the two cohorts;
its broad uncertainty and small shift do not validate a practice benefit.

The [complete measured report](results/task3-completion/report.md) retains every
parameter comparison, not just these examples, as well as all six whole-fit
gates, recorded-reference score deltas and event-aligned probabilities. Traces
for every chain and condition are saved for
[cohort 1](results/task3-completion/cohort-0-retained-traces.png) and
[cohort 2](results/task3-completion/cohort-1-retained-traces.png).


## Hierarchy recovery and prior sensitivity: MODEL analysis

The known truths are two fixed-parameter conditional-design datasets: the real
nested user/card structure, lengths and gaps are held fixed and new outcomes and
latent trajectories generated. They are not prior-drawn SBC. With only three
users, the population lag-variance information is intrinsically limited even if
many reviews identify the realized user slopes. Endogenous real scheduling is
not simulated by this conditional-design generator.

The original coefficient/path fits produce 135/142 and 141/142 truth-containing
intervals. These are dependent summaries, not calibrated coverage trials. Even
the passing synthetic 2 fit estimates lag variance at 1.005 with interval
[0.183,4.147], excluding its generating value 0.04. Its variance diagnostic
passes (Rhat 1.000508, bulk ESS 7018); convergence and truth recovery are distinct.

The original IG(2,1) assigns only **3.61e-10** probability below 0.04. An oracle
control supplies the exact three realized slopes and exact population mean:

```
tau_gamma² | gamma_true, g0_true ~
  IG(2 + 3/2, b + 0.5 sum((gamma_true - g0_true)²)).
```

For b=1 its 95% intervals are [0.128,1.212] and [0.137,1.297], still excluding
0.04. This is an exact conditional control with additional observations, not
full-model recovery, a sampler intervention, or a monotonic information bound
on the original posterior. It isolates the combination of few groups and the
prior's near-zero suppression without invoking any MCMC defect.

The prespecified full MODEL sensitivity holds IG shape at 2 and compares shared
scales b=0.1, 1 and 4 for **all three hierarchy variances**, not just whichever
parameter misses truth. Prior means are 0.1, 1 and 4; the direction and magnitudes
were frozen before new fits. No other prior/support, data, kernel, seed or budget
changes. The alternatives are sensitivity probes, not recommended or empirically
selected priors. Their posterior diagnostics and all known-truth rows, including
failures, are retained separately. A truth-containing interval under a changed
prior cannot be called a sampler improvement. The oracle intervals and prior
mass are published in `hierarchy-oracle.csv`.

### Measured prior sensitivity

| Synthetic dataset | Shared IG scale | Lag-variance mean | 95% interval | Whole-fit gate |
|---|---:|---:|---|---|
| 1 | 0.1 | 0.26163 | [0.02581, 1.13909] | FAIL: kappa Rhat 1.011436, bulk ESS 382.70 |
| 1 | 1 (original) | 0.96938 | [0.21771, 3.36242] | FAIL: kappa bulk ESS 333.446827 |
| 1 | 4 | 2.67422 | [0.70197, 8.67534] | FAIL: eight monitors |
| 2 | 0.1 | 0.12948 | [0.01841, 0.61259] | PASS, minimum bulk ESS 454.06 |
| 2 | 1 (original) | 1.00515 | [0.18314, 4.14720] | PASS |
| 2 | 4 | 3.72705 | [0.73995, 14.68172] | PASS |

The passing synthetic 2 fits demonstrate substantial hierarchy-prior dependence
even with acceptable Monte Carlo diagnostics. The lower-scale interval includes
0.04, but dependent truth-containing intervals across all 142 summaries fall
from 141 to **111**; on failing synthetic 1 they fall from 135 to **117**.
Thus recovering the singled-out variance does not establish better overall
recovery. No prior is selected or recommended from this exercise. Synthetic 1
fails at every tested prior, so its posterior sensitivity is provisional.

For passing synthetic 2, lower/higher-scale log-loss changes versus the original
are -0.018318 / +0.010979 and Brier changes -0.008069 / +0.004775.
Sampling CPU ratios are 1.019 / 1.003; minimum ESS/CPU ratios are 0.695 / 1.005.
These compare different statistical targets and one fixed dataset, not sampler
improvements or evidence for a universally better prior. All six MODEL fits,
all failures, all recovery rows and CPU/score deltas are in the measured report.

![Order and hierarchy sensitivity](results/task3-completion/completion-sensitivity.png)

## Identifiability and component efficacy

The likelihood depends on b0+alpha_user+beta_card. Because cards are nested
within user, for each user delta can be added to alpha and subtracted from every
one of that user's card effects without changing mu. A global b0/alpha shift is
also invariant. Proper priors allocate these raw effects; Gaussian blocking
improves sampling along their directions without supplying likelihood information.
User-level sums and within-user card contrasts are monitored alongside **all**
raw effects; none are dropped to manufacture convergence. Centering is reporting,
not an imposed constraint or a changed prior.

Controlled component ablations are complete at the original protocol's scope:
no_practice fixes kappa=0; no_hierarchy removes intercept and slope heterogeneity;
no_dynamics removes the whole persistence/diffusion/decay block. These comparisons
change the statistical model. No_dynamics is not isolated persistence removal,
and no_hierarchy does not isolate intercepts from random slopes. The later fixed
OU ladder locates computational difficulties; fixing parameters at known or
selected values is not a free-parameter efficacy test.

Initial mean held-out log losses are 0.5910 (full), 0.5915 (no practice), 0.6031
(no hierarchy), 0.5996 (no dynamics), across heterogeneous synthetic scenarios.
All associated whole fits fail. The two full-generating datasets give opposite
static-versus-full log-loss directions (-0.0081,+0.0086), with user-cluster
intervals spanning zero. These few-cluster intervals omit MCMC uncertainty.
They neither select a component nor establish equivalence.

The inherited logistic-normal/PG-mean prequential filter is approximate and uses
at most 200 posterior draws per chain. It predicts before incorporating each test
outcome. Proper log loss/Brier scores, AUC and calibration bias are retained, but
kernel comparisons target the same posterior and are prediction-stability
screens. An order/prior comparison changes the target/history and measures
sensitivity, not a learner benefit. No universal gain, statistically established
component benefit, calibrated recovery, or deployed scheduling effectiveness
follows from Task 3. Isolated random-slope/intercept or persistence/diffusion
benefit would require a separately designed study with adequate inference and
chronology; it is outside this bounded deliverable.

## Reproduction and preservation

Run from the repository root; choose **new output paths** on every repetition:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-diagnostics/bin/python \
  -m model_diagnostics.completion_study --out .task3-runs/NEW-completion
.venv-diagnostics/bin/python -m model_diagnostics.completion_report \
  --source .task3-runs/NEW-completion \
  --out experiments/results/NEW-completion
```

The default source is the preserved `.task3-runs/task3-coefficient-investigation`.
The new run is `.task3-runs/task3-completion-sensitivity`; aggregate reproducible
results are in [task3-completion](results/task3-completion/). Raw posterior and
available warmup NPZs preserve chain/draw dimensions, exact datasets/truths/cuts,
row mappings and predictions. Source snapshots (including the owning model),
package/data/source hashes, seeds, adaptation counts, protocol, per-chain CPU and
execution segment are saved. Raw private source rows/IDs remain local; curated
event comparisons use dense item and recorded history index. Builds/tests run
after timed fits finish. Ignored `.task3-runs/` must be backed up for raw replay;
Git alone does not preserve these arrays and real row mappings.

The earlier `.task3-runs/task3-dynamics-*`, `.task3-runs/task3-anki-*`, coefficient
study and order-audit directories are unchanged. The pre-existing untracked
`research/M4_approach_report.tex` is untouched. No production sampler default,
UI control, source Parquet, Task 1/2 file, or existing curated result is modified.
No further push is authorized or performed.

Stage evidence:

- [Initial study and ablations](TASK3_FINDINGS.md)
- [Sampler audit/remedies](TASK3_SAMPLER_REPORT.md)
- [Actual-data pilot and selection limits](TASK3_ANKI_REPORT.md)
- [Dynamics ladder and interrupted-run preservation](TASK3_DYNAMICS_REPORT.md)
- [Coefficient/path derivation and measured comparisons](TASK3_COEFFICIENT_REPORT.md)
- [Completion protocol, scalar diagnostics and comparisons](results/task3-completion/)

## Final validation and decision

After all timed fits finished:

- `npm run test:diagnostics`: **32 tests passed**, including the five new
  permutation/split/prior/oracle/output-preservation checks.
- `npm run test:bayesian`: **2 Node and 3 Python tests passed**.
- `npm run typecheck`, `npx biome check package.json`, `git diff --check` and
  `npm run build`: passed.
- Repository-wide `npm run lint`: retains **280 errors and 240 warnings** across
  150 files; no broad unrelated repairs were made. Build warnings retain the
  existing workspace-root/multiple-lockfile, dynamic tracing and Node deprecation
  messages. The Bayesian Node tests also retain the module-type warning.
- `/bayesian-demo` returned HTTP **200** after the build. No UI behavior changed.
- All **88** new raw/source snapshot artifact hashes in the manifest verify;
  every local final-report link resolves. All 12 fits / 48 chains are complete;
  no investigation worker remains. The development server remains available.

Task 3 is complete as a bounded research deliverable. The coefficient/path
sampler materially improves exploration under unchanged priors, but synthetic 1
still fails full inference validation. Feasible order affects posterior and
predictive results without recovering authentic chronology. Hierarchy recovery
is strongly prior-sensitive with only three groups; better recovery of one
variance can coincide with worse recovery elsewhere. Component efficacy,
calibrated coverage/SBC and learner benefit remain unestablished. Those are
scientific limits of the completed study, not outcomes to manufacture by changing
thresholds, selecting chains, extending runs or relabelling changed priors.
