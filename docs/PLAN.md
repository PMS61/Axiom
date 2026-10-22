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
