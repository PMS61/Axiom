# Task 3 — coefficient/path interweaving

28 October 2026. Experimental CLI-only sampler intervention; Tasks 1–2 and the
original owning model remain unchanged. No prior, support, gate or data order is
changed. The protocol below was recorded before the timed fits started.

**Measured outcome:** the new kernel passes 3/4 full-model fits; both baselines
pass 0/4. Synthetic 1 still fails kappa bulk ESS (333.4). Real chronology and
weak hierarchy-scale recovery remain unresolved. Complete measured results follow
the derivation and protocol below.

## Derivation

At fixed OU parameters and standard-normal state innovations z, write the latent
path as psi = a(z) + H theta, with theta = [b0, alpha, beta, gamma, g0]. The noise
path a starts at z0 (initial variance one), then obeys a_t = phi_t a_(t-1) +
sqrt(q_t) z_t. Constant b0/user/card effects have response one, because the initial
phi is zero and phi + (1-phi) = 1. The lag response starts at h0=x0 and obeys
h_t=phi_t h_(t-1)+(1-phi_t)x_t. g0 has zero likelihood column; the unchanged
conditional prior couples gamma to g0.

Holding the current Polya–Gamma variables omega fixed gives precision
P = P_prior + H' diag(omega) H and information
v = H' (r - 1/2 - omega*a). The exact draw is N(P^-1 v, P^-1), obtained by
Cholesky solves. P_prior retains b0,g0 variances 100 and all original effect
variances, including gamma/g0 off-diagonal entries -1/tau2_gamma and the
user-count contribution to g0 precision. No centering constraint is imposed.

The triangular state Jacobian prod sqrt(q_t) cancels the transition determinants;
it depends on OU parameters, not theta or hierarchy scales. The standard-normal
innovation density is constant during the coefficient update. The resulting path
must be reconstructed with the sampled coefficients. Original inverse-gamma
hierarchy-scale conditionals are then valid at fixed coefficients/z. The complete
extra Gibbs transition follows the existing centered and innovation parameter
updates, after the collapsed decay step has immediately restored sigma2. The same
PG variables are used throughout this sweep; the next sweep refreshes them from
the resulting path. The initial-state variance is one, not stationary variance.

Independent tests compare an explicit centered transition/PG/prior density plus
state Jacobian with the Gaussian precision and gradient, check every coefficient
path derivative with ragged histories and extreme gaps, compare simulated
conditional moments, and verify innovation preservation. Disabled extra moves
reproduce the preceding implementation exactly. These are conditional-kernel
correctness checks; matching posterior estimates alone would not prove correctness.

## Prespecified bounded comparison

The completed dynamics investigation supplies exact saved datasets, truths and
split points. They are copied byte-for-byte into a NEW output directory; original
run artifacts are never modified. Real data retain recorded order and remain
provisional because source chronology has not been authenticated. Local owning
sources still only consume the already-preprocessed AnkiFirstRun Parquet files;
no original per-user cache was found at the Task 1 or Hugging Face cache locations.
No data were downloaded and no gated access was attempted.

Four datasets (two real, two fixed-truth nested synthetic) × three full-model
kernels (centered, existing innovation, innovation plus one exact coefficient/path
move per sweep) × four chains = 12 fits / 48 chains. Each uses 8,000 iterations,
4,000 warmup, no thinning, five decay MH proposals and, for both interwoven
kernels, five innovation parameter proposals per sweep. Warmup-only adaptation
is frozen during retained draws. Seeds are exactly the preceding study's seeds.
All three kernels are rerun in the same execution segment. Saved baseline retained
and warmup arrays must match exactly; timing is newly measured. No blind chain
extensions, parameter selection or adaptation changes follow observed results.

Every original scalar, terminal state, centered contrast, combined user level and
within-user card contrast remains monitored. Whole-fit pass requires finite rank
normalized folded split Rhat <1.01 AND bulk/tail ESS >=400 for EVERY scalar.
Predictions use the inherited approximate prequential filter and at most 200 draws
per chain. Scores compare stability under the same model; they cannot rank model
components or establish efficacy. Recovery retains all 142 dependent truths per
synthetic fit and is not SBC or calibrated coverage. Hierarchy non-centering and
prior sensitivity remain separate hypotheses, not part of this intervention.

Reproduce from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-diagnostics/bin/python \
  -m model_diagnostics.coefficient_study --out .task3-runs/NEW-COEFFICIENT-STUDY
```

The runner saves exact data/split/truth fingerprints, a source snapshot, versions,
seeds, metadata, raw chain/draw arrays, warmup, per-chain predictions, all scalar
diagnostics, recovery and plots. No output directory is overwritten. Reported
ESS/CPU ratios will use contemporary baselines; timing is one realization on this
machine, not a repeated hardware benchmark. No build or test is run during fits.

## Measured results

All 12 fits / 48 chains finished in the prespecified budget. **The new kernel
passes 3/4 full-model fits, versus 0/4 for each baseline.** Every fit retains all
142 monitors. All 128 replayed baseline arrays (retained and available warmup)
match the preceding saved study exactly. The [complete results](results/task3-coefficient-investigation/report.md)
include every diagnostic, matched per-parameter comparison, failure and truth.

| Dataset | New kernel max Rhat | Min bulk / tail ESS | Whole-fit gate | Minimum ESS/CPU ratio vs innovation |
|---|---:|---:|---|---:|
| Real cohort 1 | 1.003671 | 793.5 / 1849.5 | PASS | 4.37 |
| Real cohort 2 | 1.007600 | 600.8 / 637.9 | PASS | 1.34 |
| Nested synthetic 1 | 1.005704 | 333.4 / 1119.9 | FAIL | 0.85 |
| Nested synthetic 2 | 1.007968 | 641.5 / 1502.5 | PASS | 0.97 |

The first real cohort's 33 preceding innovation failures and the second's 17
are resolved at this budget, including combined user levels, card contrasts,
lag slopes and terminal states. Synthetic 1 goes from seven failures to one:
kappa has Rhat 1.0053865 and bulk ESS 333.4468 (tail ESS 1350.2). This is a failure,
not a near-pass. Synthetic 2's preceding lambda Rhat failure is resolved.
This supports the state/location coupling hypothesis without establishing a
universal remedy: decay/practice mixing still limits one dataset. No chains were
extended, removed, or selected after observing diagnostics.

Added CPU is 1.28–1.37 times the innovation-only baseline. Although every dataset
improves minimum bulk ESS, ESS/CPU improves on the real cohorts and slightly
worsens on both synthetic datasets. The 0.97 ratio is close enough to one that a
single timing realization cannot establish an efficiency difference. Relative to
the newly rerun centered baseline, minimum ESS/CPU ratios are 3.72, 2.81, 5.28 and
4.05. All comparisons share an execution segment and exclude prediction CPU.
The whole-fit gate and computational efficiency answer different questions.

New-kernel minus innovation held-out log-loss deltas are -0.00059, +0.00007,
+0.00149 and -0.00160. Both target the same model, so these are approximate
prequential prediction-stability screens. Neither these differences nor the real
sampling passes establish predictive superiority, component benefit or efficacy.

## Recovery and identification remain separate

New synthetic fits contain truth in 135/142 and 141/142 dependent intervals;
synthetic 1 still fails the whole gate. The passing synthetic 2 fit nevertheless
misses the generating lag-slope variance .04: its posterior mean is 1.005 and
95% interval [.183,4.147], while that variance's Rhat is 1.00051 and bulk ESS
7018.3. Synthetic 1 similarly estimates .969 with interval [.218,3.362]. This
confirms that better mixing does not remove the prior/limited-data recovery
problem. Three users and the unchanged IG(2,1) scale prior leave weak information;
these two fixed truths cannot establish calibrated coverage or SBC. No prior
was changed to make recovery appear better.

The exact nested user/card likelihood symmetries also remain. Passing diagnostics
support exploration of the proper posterior, not separate likelihood
identification of b0/alpha/beta. Combined user levels and within-user card contrasts
remain useful alongside every raw monitor. Hierarchy non-centering was not needed
to obtain the three measured passes, but that does not rule out its benefit in
other designs. Prior sensitivity must remain a separately labelled model analysis.

## Selected-cohort order ambiguity

The [read-only audit](results/task3-coefficient-investigation/selected-order-audit.json)
compares longest-gap-first order with the same first event and reversed remaining
events within each card/day. Both permutations are strictly day-floor feasible in
all 60 selected histories. They give different outcome order and different lag
and cumulative-practice predictors in **9/30 histories in cohort 1 and 6/30 in
cohort 2**. Across the cohorts, 36 outcome positions, 34 lag positions and 52
practice-count positions differ. This is a lower bound from two diagnostic
permutations, not an exhaustive permutation audit or a posterior sensitivity fit.

Both cohort maximum residual ranges are unchanged between these permutations
(.80784 and .78208 days). Thus even whole-history feasibility cannot authenticate
outcome chronology in the exact current cohorts. No model data were reordered,
no original per-user files were recovered, and no gated source was accessed.
Sampling passes cannot turn the recorded real order into authenticated history.

## Next bounded decisions

Keep the kernel CLI-only. The next sampler hypothesis can target the remaining
lambda/kappa ridge with a separately derived and tested proposal at the same
posterior target, using a new matched protocol rather than extending the failed
chains until they pass. Location/state coupling is no longer the dominant
observed failure in this pilot. Any hierarchy-scale prior sensitivity study must
be separate from sampler comparison. Prior-drawn SBC over more datasets follows
correctness checks and adequate per-fit diagnostics; these fixed-truth checks
cannot substitute for it.

Before broader real inference, obtain producer/original sequence provenance or
explicitly fit a prespecified feasible-order sensitivity study, preserving
sequence indices and provided split points. The current audit quantifies predictor
ambiguity only; it does not measure its posterior impact. Keep original real
cohorts provisional and avoid widening them based on the old timing exclusions.

## Preservation and validation

Raw datasets, truths, posterior/warmup NPZs and per-chain predictions are in ignored
`.task3-runs/task3-coefficient-investigation`; the read-only audit is in
`.task3-runs/task3-coefficient-order-audit`. All preceding directories and curated
results remain untouched. Aggregate results and selected figures are under
`experiments/results/task3-coefficient-investigation`. No Task 1/2 sources/results,
original owning model, source Parquet data or unrelated research files changed.
The completed phase is published on `BayesianScheduler` at the user’s request;
ignored raw run artifacts remain local.

Diagnostics tests pass 27 checks, including six new coefficient/path, order-audit
and output-preservation tests. Bayesian demo tests pass two Node and three Python
checks. TypeScript checking, focused Biome on package.json and git diff whitespace
checks pass. Production build passes with the existing workspace-root/multiple-
lockfile, dynamic tracing and Node deprecation warnings. Timed fits finished before
any build or final validation tests began. The existing `/bayesian-demo` route
returns HTTP 200 after the build. No UI controls were added.

![Full-model sampling and CPU efficiency](results/task3-coefficient-investigation/coefficient-comparison.png)
