# Task 3 — dynamics isolation and innovation interweaving

28 October 2026. This follow-up preserves every preceding run. The experiment
is a bounded diagnostic pilot, not a new efficacy benchmark or Tasks 1–2 work.

## Extract provenance and stronger integrity audit

The only local preparation evidence found is commit `7281f7f` (12 October 2026),
which introduced both Parquet files as already preprocessed data. The tracked
Python loader consumes those files; it does not create them. It concatenates
and resplits rows and uses integer `delta_t`, so its behavior cannot establish
upstream ordering or authenticate the input fields. No local preparation script,
source download fingerprint or original intra-day timestamps was located.
The audit records SHA-256, Git blob identity and Parquet writer metadata.
Writer package versions identify the writer environment, not dataset provenance.

Across 2,664,994 rows / 316,334 user/card histories, stored card-day order never
goes backwards. The combined files are globally day-sorted, and the train file
contains essentially exactly 80% of all rows. This is consistent with a global
row split after sorting by day; it does not prove the preparation algorithm.
Users have their own time origins if `timestamp` is a renamed `day_offset`,
so such a global day split would not necessarily be a shared calendar cutoff.
That renaming remains an inference, not an authenticated fact.

There are 2,348,660 adjacent card intervals, including 945,560 same-day ties.
177,986 intervals in 59,271 histories differ by more than one day between the
calendar gap and recorded elapsed seconds. Of these, calendar gaps exceed
elapsed gaps 83,238 times; the reverse occurs 94,748 times. Ordering rows by day
alone cannot explain both kinds. Plausible causes include omitted reviews,
within-day order loss and field transformations; the local files do not identify
which cause applies to a particular history. `delta_t` agrees with rounded
elapsed days for 2,592,475 rows, but is not universally that transformation.

The public [Anki Revlogs 10K schema](https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k)
uses `card_id`, per-user `day_offset`, `elapsed_days`, and `elapsed_seconds` and
describes chronological input order. The resemblance supports a provenance
hypothesis only. Its gated files were not accessed, licenses were not accepted,
and no claim of authenticated correspondence is made. Local state codes also
need a preparation explanation before assuming that every public-schema detail
applies unchanged.

**Correction to the earlier integrity description:** checking each adjacent gap
within one day is necessary but insufficient for a globally coherent history.
Under a day-floor interpretation, `timestamp - cumulative_elapsed_days` must
have range less than one day for some unknown initial time of day. One of the
60 selected histories fails even the tolerance-adjusted necessary check (range
1.24465 days). The existing cohort selections are preserved for matched sampler
comparisons and explicitly remain provisional. This is not a newly validated
clean dataset. Do not extend these real-data results to mechanism or efficacy
claims. The audit does not replace missing outcomes or modify the source files.

## Prespecified experiment

Both saved real cohorts are reused exactly, including their original provided
split points. Two new synthetic datasets reuse their nested user/card identities,
lengths and exact gap schedules while generating every outcome and latent state
anew. Practice counts and lagged outcomes are recomputed from those synthetic
outcomes. Timing is treated as fixed, exogenous design in these simulations;
actual learner schedules can be endogenous. This tests conditional-design
recovery, not an Anki scheduling simulator or prior-drawn SBC.

The ladder is static regression, latent dynamics with all OU parameters fixed,
free diffusion only, free decay plus diffusion, and the full model with free
practice dependence. Fixed real-data controls use lambda=.05/day, sigma2=.5/day,
kappa=0, chosen before observing the new fits. Synthetic controls fix the
corresponding generating values (.35, .3, .6). **Fixing parameters changes the
fitted model.** A passing control is not a sampler-only convergence win for the
full model. Full models retain every original parameter prior and support.

Every fit has four chains, 8,000 iterations / 4,000 warmup, no thinning. Both
centered and interwoven kernels use joint decay and exact coefficient blocking;
interweaving adds five joint innovation proposals per sweep. Static and fully
fixed controls are not duplicated: 32 fits / 128 computed chains altogether.
CPU-normalized comparisons account for the added work. This is a single timing
realization per fit on this machine, not a repeated hardware benchmark.

The unchanged gate is folded rank-normalized split R-hat <1.01, bulk ESS >=400
and tail ESS >=400 for **every monitored scalar**. Undefined fails. All original
raw effects, scales, centered effects and terminal states remain monitored;
additional combined user levels and within-user card contrasts address the
nested likelihood symmetries. The training mean-design matrices each have
37 columns but rank 33: four exact likelihood-null directions, consistent with
three nested user/card shifts plus the global offset shift. Proper priors keep
the Gaussian conditional precision full rank; they do not create likelihood
information along those directions. Fixed constants are excluded only because they
are not sampled. They remain in posterior files to support predictions.

Synthetic truth includes generating hierarchy variances and terminal training
states. The recovery table retains every available truth, including failed
parameters; no calibrated coverage claim follows from two datasets. Real data
never receive a recovery calculation. Raw offsets remain prior-allocated along
nested likelihood symmetries even if their Monte Carlo diagnostics pass.

## Innovation target and update order

After the ordinary PG -> FFBS -> coefficients/scales -> collapsed decay -> exact
diffusion update, form

`z_t = [psi_t - phi_t psi_(t-1) - (1-phi_t) mu_t] / sqrt(q_t)`.

At the first event phi=0 and q=1, preserving the original initial-state prior.
Hold z and the current PG variables fixed, propose symmetric increments in
`(log lambda, kappa, log sigma2)`, and reconstruct the entire path with the new
OU parameters. The transformed transition Gaussian determinants cancel the
triangular state Jacobian `prod sqrt(q_t)`. The standard-normal z density is
constant in this parameter-only move. The remaining likelihood is the exact PG
quadratic `sum[(r-.5) psi - omega psi²/2]`. Parameter prior terms are the original
normal priors in log lambda and kappa and `-a log sigma2 - b/sigma2` for diffusion:
the last expression includes the log-diffusion Jacobian. Bounds are unchanged.

The state and parameters are accepted together; rejected parameters and path
remain unchanged. No partially collapsed sigma2 move is added after this
transformation; the centered step already restored sigma2 before interweaving.
The next sweep refreshes PG variables from the resulting path. Proposal scale
adapts only in warmup and is frozen for retained draws. Priors, likelihood and
initial variance are unchanged by this extra transition.

[Stan's reparameterization guidance](https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html)
and [Kastner & Frühwirth-Schnatter's ASIS work](https://arxiv.org/abs/1706.05280)
motivate the hypothesis; the stochastic-volatility results do not prove this
OU logistic implementation will mix well. The implementation is experimental
and CLI-only until the matched pilot is evaluated. Hierarchy non-centering is
still a separate hypothesis; a latent-path intervention does not automatically
resolve weakly informed hierarchy scales.

## Reproduction and artifacts

From the repository root:

```sh
.venv-diagnostics/bin/python -m model_diagnostics.provenance_audit \
  --out .task3-runs/NEW-AUDIT/audit.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-diagnostics/bin/python \
  -m model_diagnostics.investigate --out .task3-runs/NEW-INVESTIGATION
```

The runner requires the preceding saved `task3-anki-joint_blocked` selection
and dataset files. It refuses an existing output directory. Exact datasets,
truths, split points, draw/warmup NPZs, predictions, source hashes, chain settings,
all scalar diagnostics and trace/rank/dependence figures are saved locally.
Original-ID mappings stay in the preceding ignored run; new aggregate artifacts
use dense IDs. Existing reports and run directories are not rewritten.

After computation finishes, curate and compare without changing the raw run:

```sh
.venv-diagnostics/bin/python -m model_diagnostics.investigation_report \
  --source .task3-runs/task3-dynamics-investigation \
  --out experiments/results/task3-dynamics-investigation
```

Both source Parquet blobs match their committed versions. No input files or
unrelated research were changed.
