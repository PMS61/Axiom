# Bayesian research workspace

Open [http://localhost:3000](http://localhost:3000) or `/bayesian-demo`.
The old six-topic recall illustration is at `/bayesian-demo/recall`.

The workspace is for local research: configure datasets and MCMC, run independent
chains, inspect diagnostics and compare nested model variants. It requires no
account, database or API keys. Computation runs in local Python worker processes;
closing/reloading a browser tab does not stop a run.

## Environment

From the repository root:

```sh
python3 -m venv .venv-diagnostics
.venv-diagnostics/bin/pip install -r requirements-diagnostics.txt
cd client
npm install
AXIOM_DIAGNOSTICS_PYTHON="$PWD/../.venv-diagnostics/bin/python" npm run dev
```

Node 22.18+ is needed for the existing built-in TypeScript tests. Python 3.10+
is supported by the harness. Use the same interpreter for CLI runs and tests.
Alternatively set the absolute `AXIOM_DIAGNOSTICS_PYTHON` path in ignored
`client/.env.local`. This machine is already configured. The environment indicator
verifies NumPy, SciPy, Matplotlib, ArviZ and polyagamma. Full repository sources
must be present beside `client/`; the Python bridge is not packaged for serverless
hosting. To serve a production build locally, use `npm run build` followed by
`npm run start`, with the same environment setting.

## Workspace

- **Configuration:** synthetic or uploaded data, generating conditions, dataset
  repetitions, fitted variants, total iterations, warmup, seed, worker count and
  chronological holdout fraction. Four chains are mandatory; draws are not thinned.
- **Runs:** saved local studies, progress, cancellation and reloadable settings.
  One browser-launched study at a time avoids overlapping benchmarks. Cancel is
  checked every 25 iterations. Completion means computation finished, not convergence.
- **Results:** held-out log loss, Brier, AUC, rank R-hat, minimum bulk/tail ESS,
  CPU costs, plus paired component-ablation intervals.
- **Diagnostics:** filter every monitored parameter, group effect and terminal
  state by model/name or show failed gates. Includes MCSE and ESS per CPU second.
- **Recovery:** known truth, error, 95% intervals, coverage and diagnostic status.
  Matched generating/fitted models are selected by default. Uploaded data never
  produce recovery claims.
- **Plots:** trace/warmup and ACF, rank histograms, posterior dependence, comparative
  performance/cost and parameter recovery. Figures are exportable PNGs.
- **Report/artifacts:** generated technical report, CSVs, exact datasets and truth,
  chain/draw-preserving NPZs, settings, seeds, package versions and source hashes.

All runs live under ignored `.task3-runs/`. Save or back up that directory to
retain results. Uploaded data stay there too. The browser remembers only the
selected run ID. There is no external analysis service.

## Dataset format

Upload a JSON array (maximum 200 KB):

```json
[
  {"user": 0, "item": 0, "r": [1,0,1,1,0,1,0,1], "dt": [1,0.5,1,2,3,1,4,7]},
  {"user": 0, "item": 1, "r": [0,1,0,1,1,1,0,0], "dt": [1,1,2,1,3,4,7,14]}
]
```

Each record is one unique user/item pair, chronologically ordered. `r` is integer
0/1 recall; `dt` is the gap before that event in **days**, in [0.01,365]. The
first gap is ignored. These explicit bounds prevent silent clipping by the model.
Require 2–120 pairs and 8–100 events per pair. IDs are nonnegative integers and
are remapped densely; the mapping is saved. Previous outcomes and cumulative
practice are computed from history. Do not supply estimated outcomes as observed
feedback. The format example is for import testing, not an adequate research sample.

## Reproduce Task 3

From the repository root:

```sh
.venv-diagnostics/bin/python -m model_diagnostics.run \
  --config experiments/task3-study.json --out .task3-runs/task3-study

.venv-diagnostics/bin/python -m model_diagnostics.run \
  --config experiments/task3-longer-chains.json --out .task3-runs/task3-longer-chains
```

Use a new output directory when preserving an existing run. The first study has
five generating conditions × two dataset seeds × four variants × four chains
(160 chains). Each chain has 2,400 iterations, 1,200 warmup, 1,200 retained draws.
The longer-chain sensitivity check uses the same first full dataset, two variants,
8,000 iterations and 4,000 warmup. See the saved technical report and
`experiments/TASK3_FINDINGS.md` for observed results and limits.

The generator uses a crossed 4-user × 5-item design and 32 events per pair.
Global truth is b0=0.2, lambda=0.35/day, kappa=0.6, diffusion variance=0.3/day,
population lag slope=0.7, with user/item/slope standard deviations 0.5/0.6/0.2.
User/item realized effects are centered. Ordinary gaps are log-uniform 0.05–20
 days; the short-gap stress test uses 0.01–0.08 days. No-practice sets kappa=0;
no-hierarchy sets user/item effects to zero and all lag slopes to 0.7;
no-dynamics generates Bernoulli-logistic outcomes directly from the hierarchical
mean. Each run saves the exact realized effects and latent trajectories.

## Interpretation rules

The study targets the time-aware OU implementation in
`AnkiFirstRun/dynamic_retention_model_optimised.py`, not the root AR(1) sampler.
That source filename contains the pure NumPy implementation; the other DT file
uses Numba. The harness reuses the NumPy model's prior, effect conditionals,
FFBS, collapsed Metropolis and approximate prequential prediction. It replaces
the repository's truncated PG approximation with the reference polyagamma package.
Original research models remain unchanged by this harness.

Convergence requires rank-normalized folded split R-hat <1.01 and bulk/tail ESS
≥400 for **every** monitored scalar. A few converged parameters cannot qualify
the whole fit. Fixed ablation constants are excluded. Undefined diagnostics fail.
No-dynamics removes the entire persistence/diffusion/decay block; it is not an
isolated persistence ablation. No-hierarchy retains a shared lag slope.

Held-out predictions use observed test outcomes only after predicting each event.
The inherited logistic-normal/PG-mean online filter is approximate. Synthetic
performance is not evidence of effectiveness for learners. Bootstrap intervals
use users as clusters and are fragile with only four users; they omit MCMC
uncertainty. Few-seed recovery is not calibrated coverage or SBC. Raw intercept
and group effects have additive confounding; centered effects and combined
population level are also monitored. Longer chains can reduce simulation error
but cannot create information missing from the design.

## Verification

```sh
.venv-diagnostics/bin/python -m unittest model_diagnostics.test_diagnostics -v
cd client
npm run test:bayesian
npm run typecheck
npx biome check src/app/bayesian-demo src/app/api/research-runs src/lib/bayesian
npm run build
```

The diagnostics tests deliberately create non-mixing and autocorrelated chains,
check chronological splits and prediction leakage, enforce ablation constraints,
validate input bounds/remapping and verify proper scores with ties/single classes.
Repository-wide `npm run lint` has existing issues outside the workspace.

## Sampler-remedy comparisons (Task 3)

The sampler selector distinguishes `reference` (scalar decay and sequential
coefficients), `joint` (joint log-lambda/kappa proposal), `blocked` (exact joint
Gaussian coefficient update) and `joint_blocked` (both). Model variants and
sampler variants are separate choices. Priors and the statistical model are
identical across these kernels. Reference remains the default, including when
loading older configurations. Joint covariance and scale adapt **only during
warmup**. The MH proposals-per-sweep control defaults to five; joint uses five
bivariate proposals while reference uses five per scalar. Static fits have no MH
step; no-practice uses scalar lambda even under the joint setting.

Reproduce the new 80-chain matched pilot from the repository root:

```sh
for sampler in reference joint blocked joint_blocked; do
  for suffix in '' -stress; do
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-diagnostics/bin/python \
      -m model_diagnostics.run \
      --config "experiments/task3-remedy-${sampler}${suffix}.json" \
      --out ".task3-runs/task3-remedy-${sampler}${suffix}"
  done
done
.venv-diagnostics/bin/python -m model_diagnostics.compare \
  --runs .task3-runs/task3-remedy-* \
  --saved-reference .task3-runs/task3-longer-chains \
  --out .task3-runs/task3-sampler-comparison
```

These commands refuse to overwrite existing studies/comparisons: choose new run
names if they already exist. The comparison checks exact data hashes and matched
settings, reports per-parameter ESS/CPU and MCSE agreement screens, and optionally
checks retained draw equality against the saved longer-chain reference. It never
changes the convergence thresholds. Select the complete descriptive run IDs in
the workspace to inspect/reload settings and view predictions, recovery and
plots. The technical audit and measured decisions are in
`experiments/TASK3_SAMPLER_REPORT.md`; curated comparison tables are under
`experiments/results/task3-sampler-comparison/`.

## Actual Anki data (Task 3)

Choose **Local Anki Parquet data** in the workspace. This reads the repository's
`AnkiFirstRun/train_data.parquet` and `test_data.parquet` through PyArrow, preserves
the supplied partition and disables the synthetic holdout fraction. Users, cards
per user, maximum complete-history length and disjoint cohorts control a bounded
actual-data pilot. No Anki observations are generated synthetically. Histories
with missing/inconsistent gaps or overlapping partition days are excluded and
counted. Original-ID mappings and selected raw records stay in ignored runs.

Exact recorded elapsed seconds supply fractional-day gaps, with no silent clipping;
the model uses the same OU equations and priors over a validated timing domain of
1 second–3650 days. Full cohort selection/exclusions and limitations are documented
in `experiments/TASK3_ANKI_REPORT.md`. No known-truth recovery is computed for Anki.

```sh
for sampler in reference joint blocked joint_blocked; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-diagnostics/bin/python \
    -m model_diagnostics.run --config "experiments/task3-anki-${sampler}.json" \
    --out ".task3-runs/task3-anki-${sampler}"
done
.venv-diagnostics/bin/python -m model_diagnostics.compare \
  --runs .task3-runs/task3-anki-reference .task3-runs/task3-anki-joint \
         .task3-runs/task3-anki-blocked .task3-runs/task3-anki-joint_blocked \
  --out .task3-runs/task3-anki-comparison
```

Use new output names to preserve existing artifacts. These configs fit all four
variants on two fixed disjoint user cohorts, with 8,000/4,000 iterations/warmup.

## Task 3 experimental dynamics investigation

The CLI-only [dynamics report](experiments/TASK3_DYNAMICS_REPORT.md) documents a
read-only provenance audit, a fixed-parameter dynamics ladder, nested synthetic
recovery and centered/innovation interweaving under unchanged priors. Run
`python -m model_diagnostics.investigate --out .task3-runs/NEW` with the diagnostics
interpreter from the root; it reuses the preceding saved Anki cohort files and
refuses overwrite. Extra fixed controls change the fitted model and are not
sampler-only remedies. The original adjacent timing check does not establish
whole-history consistency; the follow-up found one selected trajectory failing
a cumulative check. Existing real-data cohorts remain provisional.

`npm run test:diagnostics` also runs transformation, Jacobian, prior-density,
frozen-adaptation and nested-generator tests in `model_diagnostics/test_innovation.py`.
