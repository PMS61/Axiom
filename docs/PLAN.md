# FSRS Benchmarking Plan

## Intent

Build a reproducible evaluation pipeline that compares **FSRS** (the official
open-spaced-repetition implementation) with **our Bayesian dynamic retention
model** (`AnkiFirstRun/dynamic_retention_dt.py`, `DynamicRetentionModelDT`). It
also compares both against **Half-Life Regression** (Settles & Meeder 2016, "the
Duolingo paper"). The data is the `open-spaced-repetition/anki-revlogs-10k`
dataset. All models are scored on the **exact same held-out review events**,
with no data leakage. The metrics are log loss, Brier score, AUC, and calibration
(ECE, a reliability curve, and mean-bias). We report user-clustered bootstrap
CIs on the paired differences. The protocol, the implementation choices and the
results are documented in `docs/EVALUATION.md`.

## Fixed decisions (orchestrator; do not change without asking)

- **Code location:** a new package `fsrs_benchmark/` at the repo root
  `/home/amal/fyp/Axiom`. Tests go in `fsrs_benchmark/tests/` and run with pytest.
  Do NOT modify `AnkiFirstRun/dynamic_retention_dt.py` or `pg_sampler.py`. Import
  them by adding their folders to `sys.path` inside one adapter module.
- **Environment:** `/home/amal/fyp/Axiom/.venv` (Python 3.12, uv). Run code with
  `.venv/bin/python`. Packages: numpy, pandas, pyarrow, numba, scipy, matplotlib,
  pytest, huggingface_hub, and `fsrs[optimizer]` (py-fsrs 6.x, with torch).
- **Data:** HF gated dataset `open-spaced-repetition/anki-revlogs-10k`. Download only
  `revlogs/user_id=<u>/data.parquet` for the sampled users into `data/anki10k/`,
  which is git-ignored. Users are sampled with a fixed seed (default 100 users,
  seed 0). The `--n_users` flag sets the count.
- **Event definition** (revised in Phase 2; see the phase-2 report):
  - One row is one review in a user's revlog. Rows keep their original
    chronological order within the user.
  - Drop rows with `rating == 0` (manual reschedules; not recall attempts).
  - KEEP `state == 4` rows. In this dataset they carry real ratings 1-4 and real
    durations (filtered-deck reviews), so they are genuine recall events.
  - Drop any card whose first kept review is not `state == 0` (new). Its history
    is truncated, so its memory state cannot be reconstructed.
  - The label is `y = 1 if rating > 1 else 0`.
  - Gaps are recomputed from the KEPT rows only, so every model sees the same gap:
    - `elapsed_days` = the `day_offset` difference to the card's previous kept row.
    - `dt_days` = the sum of raw `elapsed_seconds` since the previous kept row,
      divided by 86400 (which includes the seconds of any dropped rows between
      them). It falls back to `elapsed_days` when seconds are missing.
    - Both are NaN or -1 on first reviews. The raw values stay in `raw_elapsed_days`.
- **Split (no leakage): a per-user time cutoff.**
  - For each user, `cutoff_day` is the 80th percentile of `day_offset` over that
    user's kept reviews.
  - Reviews with `day_offset < cutoff_day` are TRAIN; the rest are TEST.
  - Model parameters are fitted on TRAIN only.
- **Scored test events:** TEST reviews that are not the card's first review and
  whose `elapsed_days >= 1` (inter-day reviews, the srs-benchmark convention).
  Same-day reviews and first reviews are still fed to the models as history, but
  are never scored.
- **Prediction is one-step-ahead.** To predict test event k, a model may use the
  card's whole history before k: its train reviews and any earlier test reviews,
  including their outcomes. It may NOT use event k or anything after it. Fitted
  parameters stay frozen during test.
- **Card cap:** at most `--max_cards` cards per user (default 300). They are
  sampled with a fixed seed, and the same cards go to every model.
- **Models:**
  1. `FSRS-6 default`: py-fsrs with its default parameters.
  2. `FSRS-6 optimised`: py-fsrs `Optimizer`, fitted per user on TRAIN reviews only.
  3. `HLR`: Settles & Meeder half-life regression, written from scratch. Its
     features are `sqrt(1+n_correct)`, `sqrt(1+n_wrong)` and a bias, all taken from
     the card's history before the event. It is fitted on train scored events.
  4. `Ours (learned lambda)`: `DynamicRetentionModelDT`, fitted on train sequences.
     Prediction uses its `predict()` (a Kalman-style one-step update with test
     outcomes).
  5. `Ours (+kappa)`: the same model with `use_kappa=True`.
- **Metrics:** written from scratch in `fsrs_benchmark/metrics.py`:
  - log loss (probabilities clipped to [1e-6, 1-1e-6])
  - Brier score
  - AUC (rank-based)
  - ECE (10 equal-width bins)
  - mean bias (mean p - mean y)
  - reliability-curve data
  - a user-clustered paired bootstrap CI for a metric difference

  srs-benchmark-style RMSE(bins) is optional.
- **Compute:** develop and smoke-test locally on 3 to 5 users. The full run (100
  users) may go on Kaggle (via the Kaggle MCP or Playwright).

## Phases

### Phase 1: Data and protocol — DONE (see docs/reports/phase-1.md)

**Task 1.1: Dataset download and loading** (`fsrs_benchmark/data.py`, tests)
- `download_users(n_users, seed, out_dir)` lists the dataset files with
  `huggingface_hub`, samples user ids deterministically, and downloads their
  revlog parquet. It skips files that already exist, and fails with a clear
  message if there is no HF token or the licence was not accepted.
- `load_user(path) -> DataFrame` applies the event definition above. Its
  columns are: `user_id`, `card_id`, `review_idx` (position within the user),
  `day_offset`, `rating`, `state`, `elapsed_days`, `dt_days`, `y`, `is_first`.
- Done means:
  - The unit tests use a small synthetic revlog frame and cover the filtering
    rules, labels, gap computation and first-review flags.
  - A real download of 2 users works.

**Task 1.2: Split and event table** (`fsrs_benchmark/split.py`, tests)
- `make_split(df, max_cards, seed)` returns the event table with `event_id`
  (unique, stable), `is_train`, and `scored` (a bool marking the test events to
  score).
- `save/load` read and write `data/processed/events.parquet`.
- Done means the tests prove all of the following:
  - No train event has `day_offset >= cutoff_day`.
  - Scored events are test-only, with `elapsed_days >= 1` and not first reviews.
  - The card cap holds, and the output is deterministic for a fixed seed.

**Task 1.3: Metrics** (`fsrs_benchmark/metrics.py`, tests)
- This task covers every metric listed above, plus `paired_bootstrap(user_ids, y,
  p_a, p_b, metric, n_boot, seed)`.
- Done means the tests check known-answer cases: the textbook AUC of 0.75, a
  perfectly calibrated ECE of about 0, log loss and Brier on hand-computed values,
  and a bootstrap CI that contains the point estimate.

### Phase 2: Models — DONE (see docs/reports/phase-2.md)

**Task 2.1: The FSRS adapter** (`fsrs_benchmark/models/fsrs_model.py`, tests)
- Study py-fsrs: `Scheduler`, `Card`, `Rating`, `ReviewLog` and `Optimizer`.
- For each card, replay its reviews in chronological order with
  `enable_fuzzing=False`. Review datetimes are synthetic: a fixed epoch plus the
  cumulative `elapsed_seconds`, or `day_offset` days when seconds are missing.
- Before applying each scored event, record the retrievability at that event's
  time. That is the prediction.
- `fit(train_events)` runs the Optimizer per user on that user's TRAIN reviews
  only, with a fallback to the defaults when the user has too little data.
  `predict(events) -> Series[event_id -> p]`.
- Done means:
  - A test proves no leakage: changing a scored event's own rating, or any later
    one, does not change its prediction.
  - A test proves the replay matches py-fsrs' own retrievability on a
    hand-built card.
  - Both variants produce predictions in (0,1) for every scored event of 2 real
    users.

**Task 2.2: The HLR baseline** (`fsrs_benchmark/models/hlr.py`, tests)
- Use the HLR model of Settles & Meeder: `p = 2^(-dt/h)`,
  `h = 2^(theta·x)`, and the paper's loss, which is squared error on p plus
  alpha times squared error on log2(h), with L2 regularisation. Fit with
  numpy/scipy on train events.
- Done means:
  - Tests on synthetic data recover the true theta roughly.
  - The features use only past history.
  - Predictions are produced for every scored event.

**Task 2.3: Our model's adapter** (`fsrs_benchmark/models/ours.py`, tests)
- Convert the event table to the sequence format of `DynamicRetentionModelDT`:
  - one sequence per (user, card)
  - `r` = y
  - `x` = the lagged y
  - `dt` = dt_days, with the first set to 1.0
  - users and cards remapped to dense ints
- Train sequences use only TRAIN events.
- For prediction, build a test sequence holding each card's TEST events in
  order (cold cards with no train events work through the model's cold path).
  Call `predict()`, then map back to `event_id`. Keep only scored events.
  Unscored test events stay in the sequence as history.
- Done means:
  - The leakage test from 2.1 passes here too.
  - A smoke fit (iters=60, burn=30) on 2 users predicts every scored event.

### Phase 3: Run, compare, document — DONE (see docs/reports/phase-3.md)

**Task 3.1: The runner** (`fsrs_benchmark/run.py`)
- A CLI that runs download, then load, then split, then fits all models, then
  writes `results/<run_name>/predictions.parquet`. The predictions file has
  `event_id`, `user_id`, `y`, `elapsed_days` and one column per model.
- Flags: `--n_users --max_cards --seed --iters --burn --models --run_name`.
- The runner asserts that every model predicted exactly the same event_ids.
- Done means a smoke run on 3 users finishes end to end.

**Task 3.2: The report generator** (`fsrs_benchmark/report.py`)
- Reads `predictions.parquet` and writes `metrics.csv` and `report.md`
  containing:
  - an overall table per model
  - a table per gap bin (0-1, 1-3, 3-14, 14-60, 60+ days)
  - paired bootstrap CIs for each model minus FSRS-6 optimised
  - reliability-diagram PNGs
- Done means the smoke run produces all the files, and the numbers match
  `metrics.py` when recomputed by hand.

**Task 3.3: The full run and documentation**
- Run the full configuration (100 users; locally if feasible, otherwise on
  Kaggle).
- Write `docs/EVALUATION.md`: the protocol, the leakage guarantees,
  implementation choices, how to reproduce it, the results tables and figures,
  and limitations.
- Done means the results are in the repo and the doc is complete.

### Phase 4: Backlog fixes before the long run — DONE (see docs/reports/phase-4.md)

**Task 4.1: Progress logging and durable run info** (`fsrs_benchmark/models/ours.py`, `fsrs_benchmark/run.py`, tests)
- `OursModel` takes an optional `progress_every` (the runner passes 100). Every N Gibbs sweeps it logs one line to stderr: the sweep number out of the total, elapsed time, ETA, and the current λ (and κ when on). Hook it into the existing per-instance `_update_lam_sigma2` wrapper, and do not modify AnkiFirstRun/.
- The runner also writes per-model info (fit and predict seconds, fallback users, HLR epochs) into the `pred_<model>.json` sidecar, in a `details` key that the `--resume` check ignores. On `--resume`, `run_info.json` takes that info from the sidecar, so an interrupted run no longer loses it.
- Done means:
  - Tests prove logging does not change the samples (same seed, logging on vs off).
  - Tests prove `run_info` gets its details from the sidecar after a resume.
  - Old sidecars without `details` still work.

**Task 4.2: Report and supplementary polish** (`fsrs_benchmark/report.py`, `fsrs_benchmark/supplementary.py`, tests)
- Remove the large blank gap between the reliability panel and the legend.
- Add `fsrs_benchmark/supplementary.py` (`python -m fsrs_benchmark.supplementary --run_dir ...`). It regenerates `supplementary.json`, and the regenerated file must match the committed `results/full_u100_c1000/supplementary.json` values. Update its `generated_by` note and EVALUATION.md's mention of the script.
- Silence the "Mean of empty slice" RuntimeWarning in `hlr.py` for an empty train set, by fixing it in the code rather than filtering it in the test.
- Done means the full suite passes with 0 warnings from our code, and the plot has been regenerated and checked.

**Task 4.3 (orchestrator):** commit to `BayesianScheduler`, then start a long run (`ours`, 8000 iterations, 4000 burn-in) in `results/full_u100_c1000_long`, reusing the FSRS and HLR predictions.

### Phase 5: The same benchmark on the Duolingo dataset — DONE (2026-11-04, report: docs/reports/phase-5.md)

**Intent.** Run the identical protocol (split, scoring rule, models, metrics) on Duolingo's public
half-life-regression data (Settles & Meeder 2016, `settles.acl16.learning_traces.13m.csv.gz`,
Harvard Dataverse doi:10.7910/DVN/N8XJME, CC BY-NC 4.0, about 13M rows). This tests whether the
Anki ranking (FSRS > ours > HLR) holds on HLR's home data. Only the data layer is new: the
Duolingo loader must return the same columns as `data.load_user` (LOAD_COLUMNS), so `split`, the
models, `run` and `report` are reused unchanged.

**Fixed decisions (Duolingo event definition):**
- Raw file: `data/duolingo/settles.acl16.learning_traces.13m.csv.gz` (git-ignored via `data/`).
  Columns: p_recall, timestamp (unix s), delta (s since this lexeme was last seen), user_id
  (string), learning_language, ui_language, lexeme_id, lexeme_string, history_seen,
  history_correct, session_seen, session_correct.
- One row = one practice session of one lexeme by one user = one review event.
- card = (user, lexeme_id). Card ids are dense ints per user, from the sorted lexeme_id strings.
- user_id (int) = the user's index in the sorted list of ALL user_id strings in the file, so the
  id does not depend on which users were sampled.
- Order inside a user: timestamp, then lexeme_id, then original file row (stable sort).
  review_idx = 0..n-1 in that order.
- day_offset = floor((timestamp - T0) / 86400), where T0 = the minimum timestamp in the whole
  file (computed once in the scan and stored in the cache).
- y = 1 if p_recall >= 0.5 else 0 (the group's existing convention in `data_pipeline.py`).
- rating = 3 (Good) if y == 1 else 1 (Again). FSRS only sees Again/Good on this data.
- is_first = the card's first row in the log. Its history before the log window is unknown
  (history_seen > 0), so this row starts the replay and is never scored. state = 0 on is_first,
  else 2.
- elapsed_days = day_offset difference to the card's previous row (-1 on is_first).
  dt_days = timestamp difference to the card's previous row / 86400 (NaN on is_first).
  raw_elapsed_days = floor(delta / 86400).
- history_seen/history_correct, lexeme features and session counts are NOT used by any model
  (every model only sees the replayed history), so the comparison stays like-for-like. This is a
  known limitation versus the original HLR paper and must be written up.
- User sampling: only users with at least `--min_user_rows` rows are eligible; sample
  `--n_users` of them with `np.random.default_rng(seed).choice` over the sorted eligible ids.

**Scale decision (user, 2026-11-03): run on ALL Duolingo data.** Measured: 12,854,226 rows,
115,222 users, 5,861,629 (user, lexeme) cards, card length median 1 / p99 12 / max 352, 11.9 days
span, y rate 0.927, 47% of rows have a gap >= 1 day. Consequences:
- `--n_users 0` means all users; `--min_user_rows 1`; `--max_cards` large enough to never bind.
- The numba `ours` cannot run on this (~270 s/sweep estimated); only `ours_gpu` runs on the full
  data. Equivalence of the GPU port with the original is established by the 5.3 tests.
- One padded N x Tm matrix would be 5.9M x 352 doubles (~16 GB), so the GPU port stores
  sequences in LENGTH BUCKETS (see 5.3). Kept posterior draws are limited (thin so that about
  100 draws are kept) because beta and psi_last are per card (millions of entries per draw).
- FSRS predict must run in parallel over users (`--workers`), since py-fsrs replays in Python.

**Task 5.1: Duolingo loader** (`fsrs_benchmark/duolingo.py`, tests in `fsrs_benchmark/tests/test_duolingo.py`)
- `scan(path, cache_path)`: one chunked pass (pandas `chunksize`, `usecols=["user_id","timestamp"]`)
  that returns and caches (parquet) a table of user_id string -> row count, plus T0. The cache is
  reused when it exists and the source file's size and mtime match the stored values.
- `sample_users(counts, n_users, seed, min_rows)` -> sorted list of user_id strings. It raises a
  clear ValueError when fewer eligible users exist than requested.
- `load_users(path, user_strings, counts_table, T0)` -> one DataFrame with exactly LOAD_COLUMNS
  (same dtypes as `data._empty_frame()`), built with the rules above, chunked so memory stays
  bounded.
- Done means: unit tests on a tiny synthetic csv.gz (written in the test) cover ordering, ties,
  y/rating mapping, is_first, elapsed_days/dt_days, int user ids that are independent of the
  sample, the cache reuse/invalidation, and the not-enough-users error; `split.make_split` accepts
  the output; and a real-file check loads 3 users.

**Task 5.2: Runner and report integration** (`fsrs_benchmark/run.py`, `fsrs_benchmark/report.py` if needed, tests)
- New flags: `--dataset {anki,duolingo}` (default anki), `--duolingo_path` (default the path
  above), `--min_user_rows` (default 100). With duolingo, the runner uses the 5.1 functions
  instead of `data.download_users`/`load_user`; everything after loading is unchanged.
- `model_config` adds `dataset` (and `min_user_rows`) only when the dataset is not anki, so
  existing Anki sidecars still resume. `run_info.json` records the dataset.
- The report title/text names the dataset taken from run_info (Anki stays the default wording).
- Done means: existing tests still pass; new tests cover the flag parsing and the resume config;
  a smoke run `--dataset duolingo --n_users 5 --iters 60 --burn 30` finishes end to end and the
  report builds.

**Task 5.3: GPU port of our model, same equations** (`fsrs_benchmark/models/dynamic_retention_gpu.py`, tests in `fsrs_benchmark/tests/test_dynamic_retention_gpu.py`)
- User request: a GPU (Kaggle) version of `DynamicRetentionModelDT` in a NEW file. The equations,
  priors, defaults, update order and the predict() filter must be IDENTICAL to
  `AnkiFirstRun/dynamic_retention_dt.py` and `pg_sampler.py` (do not modify those two files). Only
  the array library changes: PyTorch (already installed; CPU build locally, CUDA on Kaggle).
- Class `DynamicRetentionModelGPU(sequences, rng=None, ..., device=None, dtype=torch.float64)`:
  same constructor arguments and defaults as the original plus `device` (default: "cuda" if
  available else "cpu") and `dtype`. Same public attributes (N, Tm, lens, n_users, n_items, lam,
  kappa, sigma2, b0, g0, alpha, beta, gamma, tau2, step, step_k, acc_rate, use_kappa, samples,
  key2idx, R, init_var, dt_bounds) and the same methods: `_dyn, _mu, _ffbs, _gauss,
  _inv_gamma_tau2, _update_effects, _update_lam_sigma2(psi, it, adapt), run, fit, summary,
  print_summary, predict`. `run()` must call `self._update_lam_sigma2` once per sweep (the
  progress/acceptance wrapper in ours.py replaces that attribute). After `run()`, `samples` holds
  NUMPY arrays with the same keys and shapes as the original; scalar parameters (lam, kappa,
  sigma2, b0, g0, acc_rate, tau2 values) are Python floats. Reuse `split_rhat` and `ess` by
  importing them from the original module (add AnkiFirstRun and the repo root to sys.path the
  way ours.py does).
- Randomness: a `torch.Generator` on the device seeded from `rng.integers(2**62)`; the small scalar
  draws (MH proposals/accepts, inverse-gamma draws, b0/g0 normals) may use the numpy `rng` as the
  original does. Bitwise equality with numpy is NOT required; equality in distribution is.
- Polya-Gamma: same truncated series with n_terms=200, computed on the device. Gamma(1,1) draws
  may be generated as Exponential(1) (identical distribution). Process events in chunks so memory
  stays bounded (chunk size parameter, default about 2e7 elements per chunk).
- Memory layout (needed for the full data): sequences are grouped into length buckets
  (lengths 1, 2, 3-4, 5-8, ..., powers of two up to the max); each bucket has its own padded
  (N_b x T_b) tensors and the FFBS runs per bucket. This is mathematically identical, since
  sequences are independent given the parameters and every other update sums over observed
  entries only. Flattened per-event views (u, i, x, r, dt, C, transition mask) are kept for the
  effects and lambda/sigma2 updates. Attribute names N, Tm, lens etc. keep their meaning.
- predict() processes test sequences in chunks so S x K x T never has to fit at once.
- FFBS: the forward filter loops over t and is vectorised over the N sequences exactly as the
  numba kernel; the backward sampler is the same recursion vectorised over n (the numba code
  loops over n only for speed). Same 1e-12 floors, same `last[n] == t` and `t <= last[n]` rules.
  Expose `_ffbs_forward(omega, phi, q, mu)` returning (m, P, a, Rp) and `_ffbs_backward(m, P, a,
  Rp, phi, eps)` so tests can pass a fixed eps.
- predict(): the same Kalman one-step filter over posterior draws as the original, on the device,
  returning a list of numpy arrays like the original (`return_draws` supported).
- Done means (tests run on CPU torch locally):
  1. Deterministic parity with the original within 1e-9 when both are given the same state:
     `_dyn`, `_mu`, the forward filter (compare against a NumPy re-implementation of the numba
     forward loop inside the test), the backward recursion for a fixed eps, and `predict()` when
     the GPU model is given a copy of a fitted original model's `samples` (and R, lens, key2idx).
  2. PG sampler: empirical mean matches tanh(c/2)/(2c) within 0.01 for c in {0, 0.5, 1, 2, 5, -3}.
  3. Statistical parity: on a small synthetic data set, fit the original and the GPU model with
     the same settings (e.g. 400 sweeps, 200 burn) and check that posterior means of b0, lam and
     sigma2 and the mean predicted probability agree within tolerances set from the chains' own
     spread; log loss of the two prediction sets differs by < 0.01.
  4. `samples` keys, shapes and types match the original; `summary()` works.
  5. A rough timing print (not an assertion) of one sweep on CPU for N about 5000 sequences.

**Task 5.4: Wire the GPU model into the pipeline** (`fsrs_benchmark/models/ours.py`, `fsrs_benchmark/run.py`, tests)
- `OursModel(..., backend="numba")`: backend "gpu" builds `DynamicRetentionModelGPU` (device from a
  new `device=None` argument) instead of `DynamicRetentionModelDT`; everything else (sequence
  building, diagnostics, progress logging, predict mapping) is shared.
- Runner: new model keys `ours_gpu` and `ours_kappa_gpu`, a `--device` flag (default auto) and a
  `--thin` flag (default 4) passed to OursModel.
- `FSRSModel.predict` runs per-user replays in parallel when n_workers > 1 (same results as serial;
  test it), and the runner passes `--workers` to both FSRS variants.
  The keys are opt-in: ALL_MODELS default list behaviour for existing runs must not change
  (add the new keys to the allowed set, not to the default `--models`).
- `model_config` for the GPU keys records iters/burn like the numba ones, plus `backend`.
- Teammates' work (pulled 2026-11-03, docs/reports/VALIDATION_REPORT.md): add the validated
  LagLogit+dt baseline as model key `laglogit`. New adapter `fsrs_benchmark/models/laglogit.py`
  wraps `baselines.LaggedLogistic(use_dt=True)` (do not modify baselines.py) using the same
  sequence building as ours.py (`_sequences`, dense ids from TRAIN rows, unseen users/cards get
  the padding index, which LaggedLogistic already ignores). fit on TRAIN sequences, predict on TEST
  sequences, keep scored rows only, clip to [1e-6, 1-1e-6]. Opt-in key like the GPU keys. Test:
  no leakage (changing a scored event's own outcome or later ones does not change its prediction)
  and every scored event is predicted.
- Done means: existing tests pass; new tests cover the backend switch and the new keys; a smoke
  run `--dataset duolingo --n_users 5 --models fsrs_default,hlr,ours,ours_gpu --iters 60 --burn 30`
  finishes on CPU and the report builds.

Note: VALIDATION_REPORT.md flags an unverified FFBS padding concern; the 5.3 bucketed-vs-padded
parity test addresses it and must be reported in phase-5.md.

**Task 5.5 (orchestrator): Kaggle GPU run, docs, PDF.** Run on a Kaggle GPU notebook (planned P100; run on T4 x2 because the P100 was not offered; float64):
clone the public repo (github.com/PMS61/Axiom, branch BayesianScheduler) and download the
Duolingo file from Dataverse inside the notebook, so no data is uploaded. Full run in
`results/duolingo_*`, then add a Duolingo section to `docs/EVALUATION.md` and to the PDF report,
and write `docs/reports/phase-5.md`.

### Phase 6: An experimental Bayesian model that beats FSRS — PLANNED (user request 2026-11-03)

**Intent.** A NEW, separate model, with the original `DynamicRetentionModelDT` and its GPU port left
untouched. Its formulas may differ freely. The goal is to beat FSRS-6 (default and optimised) on log
loss, and ideally AUC and ECE, on BOTH datasets: Anki (`full_u100_c1000` events) and Duolingo (all
data). The claim must be honest, so model selection never touches the test events.

**Why FSRS currently wins (diagnosis):**
- (a) FSRS uses the 4 Anki grades; ours uses only pass/fail.
- (b) FSRS uses a power-law forgetting curve with stability growth that depends on retrievability at
  the review (the spacing effect). Ours uses OU mean reversion of the recall logit, which is
  over-optimistic at long gaps (+0.09 bias at 61+ days).
- (c) FSRS-opt fits each user separately or falls back to defaults. A hierarchical prior should win
  on users with little data, which is most Duolingo users.

**Protocol (fixed):**
- Validation split: inside each user's TRAIN period, the last 20% of train days (same quantile rule)
  is VALIDATION. Candidates are fitted on train-minus-validation and compared on validation events
  scored with the same rule (not first review, gap >= 1 day).
- The test set is used ONCE, for the final chosen model, and every experiment tried is listed in the
  report (no silent cherry-picking).
- The final model is refitted on the full TRAIN period with the frozen choice, then scored on the
  same test events through the existing runner and report (new model key `ours_v2`).

**Candidate family (to explore in order; stop when validation beats FSRS-opt on both datasets):**
1. Hierarchical Bayesian FSRS-style memory model:
   - forgetting curve R(t,S) = (1 + F*t/S)^C, with C < 0 learned;
   - latent log-stability per card updated on each review by a learned function of grade,
     retrievability at review, difficulty and the current stability;
   - user-level random effects on the update parameters, partially pooled with a global prior;
   - fit by MAP or variational inference in torch (GPU-capable), with uncertainty from the posterior.
2. Add process noise on log-stability (a state-space version, filtered like the current FFBS idea).
3. Calibration layer only if fitted on validation, never on test.

### Phase 7: The best model (user request 2026-11-04: "use everything you got and make the best model") — IN PROGRESS

**Intent.** Get the best possible log loss on BOTH Anki and Duolingo, using every idea that
validation supports. The Phase 6 protocol still applies: selection on VALIDATION only (last 20% of
each user's train days; Anki 100 users; Duolingo 5000-user dev sample, with a full-data check for
the final pick). The test set is used ONCE more, for the final pick only. The report says this is
the second look at test (the first was v2 in Phase 6), and lists every candidate tried.

**What we know (validation and test, Phases 5-6):**
- v1 (OU-logit, Gibbs/PG, pass/fail) is strong on Duolingo (test 0.268) and weak on Anki (0.422).
- v2 (FSRS-style curve, grades, hierarchical user effects, MAP) is at parity on Anki (0.379 =
  FSRS-opt) and beats FSRS on Duolingo, but is weaker than v1 there (0.289).
- On Duolingo, `card_id` is per user, so nothing pools the same lexeme across users. HLR's
  official version uses lexeme features, and lexeme difficulty is likely a large signal.

**Tasks:**
- 7.1 (Phase 6 builder, Task E, running): v1+v2 weighted ensemble on validation.
- 7.2 v3 = v2 plus the strengths of v1 and HLR, in a NEW file `models/ours_v3.py` (v2 is not changed):
  (a) an optional item-level random effect shared across users (Duolingo: the lexeme; the loader
      gains an optional `lexeme` column, with Anki leaving it absent);
  (b) a lagged-outcome term and a per-card random effect (what v1 has);
  (c) anything else that validation supports. Each component is ablated on validation.
  Done when: tests pass, there is no leakage (perturbation test as in Phase 6), the validation table
  covers both datasets with CIs vs FSRS-opt, and `ours_v3` is registered in run.py and report.py.
- 7.3 Stacking: a cross-fitted (folds by user) logistic stacker on VALIDATION predictions of v1,
  v2/v3, FSRS-opt and LagLogit, plus simple features (log gap, review count). It is selected only
  if it beats the best single model on validation by a margin whose CI excludes 0.
- 7.4 Final pick by a fixed rule (lowest mean validation log loss over the two datasets among
  candidates that do not lose to FSRS-opt on either). Then ONE test run, the docs, and the report.

**Phase 7 scope change (user, 2026-11-04): "you can use anything not just stuck on baysian any ml or non ml
techniques is fine we just have to make the ultimate model".** Any technique is allowed. The constraints
that stay:
- Same inputs as every other model: the event table columns (plus the Duolingo lexeme id).
  Duolingo's history_seen/history_correct are NOT used (they count exposures before the log window,
  which no other model sees).
- Predictions are one-step-ahead. A feature for review t uses only that card's and that user's
  reviews before t. Train-derived statistics (e.g. lexeme or user recall rates) come from fit rows
  only (train rows for the final model, fit-part rows for validation).
- Selection is on validation only. The test set is used once for the final pick, with every
  candidate listed.
- **Resources (the PC crashed on 2026-11-04):** at most ONE heavy local job at a time, peak RAM
  under about 6 GB, at most 4 threads. Development uses the Anki validation split and the
  5000-user Duolingo dev sample. Full-Duolingo runs go to Kaggle. Anki stays local (its data is gated).

**Task status:** 7.1 ensemble paused (the v1 validation fits are too slow locally, 3-9 h). 7.2 v3 done
and verified (validation Anki 0.3792, Duolingo dev 0.2707; FROZEN q2); its test run has not been made.
- 7.3 GBDT (`models/gbdt.py`, LightGBM 4.6.0): past-only features per review (gap, counts, lapses,
  the last k grades/outcomes/gaps, the FSRS-6 default state S/D/R replayed from history, running user
  stats, a fit-rows-only lexeme/user prior, etc.), trained on train rows. Ablate feature groups on
  validation.
- 7.4 (optional, later) a GRU sequence model on Kaggle GPU, if 7.3 leaves a gap.
- 7.5 Stacking or blending of v3, GBDT and FSRS-opt on validation (cross-fitted by user).
- 7.6 Final pick, ONE test run (Anki local, Duolingo on Kaggle), docs.

**Status at pause (2026-11-04, user said "pause we will continue later"):**
- 7.3 GBDT built (`models/gbdt.py`, FROZEN `nf_l15_lr02_sm3_nup`): validation Anki 0.3614 and Duolingo dev
  0.2473 (FSRS-opt 0.3851 / 0.2894). NOT YET VERIFIED and NOT committed. The FIRST thing to do on resume:
  a verifier must check that the running USER stats (the biggest signal) use only reviews with a
  STRICTLY EARLIER timestamp. Duolingo logs a whole session's words at one timestamp, so counting
  same-timestamp rows would leak simultaneous outcomes. The same check applies to the card/lexeme features.
- Then: 7.5 stacking (v3 + GBDT + FSRS-opt, cross-fitted on validation), 7.6 final pick and ONE test run
  (Anki local; full Duolingo on Kaggle, never locally, because of the PC RAM limit), docs.
- 7.1 ensemble (v1 validation fits) is paused; it is optional.
- The v3 test run was aborted (the PC crashed); no v3/GBDT test numbers exist yet.
