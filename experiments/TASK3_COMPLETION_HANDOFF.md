# Task 3 completion handoff

User request: push the completed coefficient/path phase and start another session
that finishes the remaining Task 3 work. Continue on `BayesianScheduler` in this
checkout. Tasks 1 (FSRS) and 2 (baselines/statistical validation) belong to teammates.
Do not modify their implementation/results or message other chats.

## Read first

- `AGENTS.md`, `RESEARCH_WORKSPACE.md`.
- `experiments/TASK3_COEFFICIENT_REPORT.md` and its curated results under
  `experiments/results/task3-coefficient-investigation/`.
- `experiments/TASK3_DYNAMICS_REPORT.md`, `TASK3_ANKI_REPORT.md`,
  `TASK3_SAMPLER_REPORT.md` and `TASK3_FINDINGS.md`.
- `model_diagnostics/{coefficient_innovation,coefficient_study,coefficient_report,
  selected_order_audit,innovation,investigate,model,analysis,anki}.py` and tests.

The owning OU model is `AnkiFirstRun/dynamic_retention_model_optimised.py` (pure
NumPy despite its name). Keep it unchanged unless an explicitly justified Task 3
requirement calls for it. Experimental interweaving stays CLI-only.

## Current measured outcome

The new exact PG coefficient/path Gibbs move at fixed innovations and OU
parameters preserves the same likelihood, priors and initial-state variance one.
It follows centered updates and the existing innovation parameter move, jointly
reconstructs psi, then redraws original hierarchy-scale conditionals. g0 has no
likelihood column; gamma/g0 prior precision retains off-diagonal terms.
Independent density, gradient, Jacobian, Gaussian-draw and innovation-preservation
tests pass. Disabled extra moves reproduce the prior kernel exactly.

12 fits / 48 chains, 8000 iterations / 4000 warmup, no thinning, four chains,
identical datasets/splits/seeds and full-model priors. Both baselines were rerun in
one execution segment: all 128 retained/available-warmup parameter arrays exactly
match saved prior baselines. New kernel passes 3/4 fits; each baseline passes 0/4.
All 142 monitors are retained. New fit results:

- Real 1: Rhat max 1.003671, bulk/tail minimum 793.5/1849.5, PASS.
- Real 2: 1.007600, 600.8/637.9, PASS.
- Synthetic 1: 1.005704, 333.4/1119.9, FAIL solely kappa bulk ESS 333.4468.
- Synthetic 2: 1.007968, 641.5/1502.5, PASS.

Minimum ESS/CPU ratios vs innovation-only are 4.37, 1.34, .85, .97 respectively;
extra CPU is 1.28–1.37x. This is not a universal efficiency win. Predictions use
an inherited approximate prequential filter; scores are stability checks of the
same model, not efficacy or component-benefit evidence.

Known-truth interval counts are 135/142 and 141/142, dependent and not calibrated
coverage/SBC. Even the passing synthetic 2 fit misses gamma variance truth .04:
mean 1.005, interval [.183,4.147], Rhat 1.00051, bulk ESS 7018. Three users plus
unchanged IG(2,1) prior leave weak hierarchy information. Do not fix this by
changing priors under a sampler-only label. Nested user/card likelihood symmetries
persist despite sampling passes; raw effects remain prior-allocated.

## Data chronology

AnkiFirstRun extract entered Git preprocessed; no producer or original timestamps
were found. Public schema resemblance does not authenticate correspondence.
The global longest-gap-first within-card/day diagnostic eliminates ALL adjacent
and cumulative day-floor timing discrepancies, superseding the earlier missing-
review interpretation. It is not uniquely authenticated order. No model dataset
has been reordered. Old timing exclusions do not identify a uniquely clean subset.

The new read-only selected-cohort audit compares longest-gap-first with the first
event retained and remaining same-day events reversed. Both orders are feasible
in all 60 histories, yet outcome/lag/practice order differs in 9/30 and 6/30
histories (36 outcome positions, 34 lag positions, 52 practice positions overall).
This is a two-permutation lower bound, not posterior sensitivity. Local Task 1
raw cache and expected Hugging Face cache were absent. No gated data access or
license acceptance was attempted. Do not widen real cohorts before resolving or
explicitly characterizing order sensitivity.

## Finish Task 3

Own the remaining investigation and final deliverable; do not stop at another
plan or routine handoff. A scientifically complete study may conclude that a
mechanism is weakly identified or a kernel fails. Completion must not depend on
manufacturing passes or statistically significant benefit.

1. Address the remaining decay/practice bottleneck with a derived/tested sampler
   intervention and a prespecified bounded matched comparison, or characterize
   the limitation rigorously if no tested intervention resolves it. No blind chain
   extension, adaptive stopping or threshold tuning. Preserve all baselines.
2. Recover authenticated ordering if available. Otherwise run a bounded, explicitly
   labelled feasible-order POSTERIOR sensitivity comparison on the same cohorts,
   preserving split boundaries, original row indices/mappings, outcomes and gaps.
   Keep recorded order as a reference. Do not call permutations original truth;
   include within-chain diagnostics before interpreting posterior changes.
3. Quantify weak hierarchy recovery/prior sensitivity as a separate MODEL analysis
   with labelled priors/controls and justified scope. Monte Carlo convergence is
   not truth recovery. Do not require fixed truth to lie inside every interval.
   Prior-drawn SBC, if used, requires actual prior draws; the two fixed truths are
   not SBC. Avoid expanding into an unbounded simulation program.
4. Consolidate Task 3 into a clear final report with completed convergence,
   component-ablation, identification, recovery, predictive and computational
   findings, unresolved scientific limitations and reproducible commands. State
   where component efficacy remains unestablished; Tasks 1–2 stay out of scope.
   Keep product controls simple and experimental kernels CLI-only unless validated.
5. Run relevant tests/build, preserve artifacts, and deliver the final Task 3
   conclusion. Do not claim complete validated full-model inference if a fit fails.

NONNEGOTIABLE gate: finite rank-normalized folded split Rhat <1.01 AND bulk/tail
ESS >=400 for EVERY monitored scalar. Undefined fails. Keep all raw and derived
monitors, chains and truths. No rounded near-passes, altered priors/support under
sampler labels, or fixed dynamics labelled full-model sampling wins.

## Preservation and execution

Never delete/overwrite/reset `.task3-runs/`: it contains irreplaceable posterior,
warmup, truths, exact datasets, mappings and split points. Use NEW directories.
Latest raw run: `.task3-runs/task3-coefficient-investigation` (complete, 12 fits).
Order audit: `.task3-runs/task3-coefficient-order-audit/audit.json`.
Preceding full dynamics: `.task3-runs/task3-dynamics-completed` (32 fits).
Earlier interrupted directories remain preserved with documented exact replay.
Original real selections: `.task3-runs/task3-anki-joint_blocked`.
All source snapshots, hashes, packages, seeds and CPU metadata must remain usable.

Leave the pre-existing untracked `research/M4_approach_report.tex` untouched.
Reverify branch, worktree and processes before acting. `.venv-diagnostics` is
configured; use OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 for matched timing and
avoid concurrent builds/tests during timed fits. No investigation workers remained
after this phase. Dev serves port 3000; reverify current state.

Last validation: diagnostics 27 tests, Bayesian 2 Node + 3 Python tests, typecheck,
Biome package.json, git diff --check and production build pass. Existing workspace
root/multiple-lockfile, dynamic tracing and Node deprecation warnings remain.
`/bayesian-demo` returned HTTP 200. No UI changes were made this phase. Root lint
has previously documented unrelated failures and was not broadly repaired.

The current session is authorized to commit/push this completed phase. The new
session should finish Task 3 locally; do not make a further push unless requested.
