# Phase 5 report: the benchmark on the full Duolingo dataset

## What was built

- **Duolingo loader** (`fsrs_benchmark/duolingo.py`). `scan` makes a chunked pass over the raw csv.gz and caches user counts and T0. `sample_users` draws users with a seed and raises if too few are eligible. `load_users` returns the `data.load_user` columns, so `split`, the models, `run` and `report` are reused unchanged. Event rules are in `docs/EVALUATION.md` Section 10.1.
- **GPU port of our model** (`fsrs_benchmark/models/dynamic_retention_gpu.py`). `DynamicRetentionModelGPU` is the same equations as `AnkiFirstRun/dynamic_retention_dt.py` in PyTorch. Sequences are grouped into length buckets, and FFBS runs per bucket. The original files were not changed.
- **LagLogit + dt baseline** (`fsrs_benchmark/models/laglogit.py`, key `laglogit`). It wraps `baselines.LaggedLogistic(use_dt=True)`. The adapter does not modify `baselines.py`.
- **Parallel FSRS predict** (`FSRSModel.predict` with `n_workers > 1`). Per-user replays run in a process pool.
- **Runner flags** (`fsrs_benchmark/run.py`). `--dataset {anki,duolingo}`, `--duolingo_path`, `--min_user_rows`, `--device {auto,cpu,cuda}`, `--thin`. New opt-in keys: `ours_gpu`, `ours_kappa_gpu`, `laglogit`. The default model list is unchanged. `model_config` adds `dataset` and `min_user_rows` only off Anki, and `backend` only for GPU keys, so existing sidecars still resume.

## How it was tested

Test files under `fsrs_benchmark/tests/`:
- `test_duolingo.py`, `test_duolingo_verify.py`: scan cache reuse and invalidation, sampling, ordering and ties, y and rating mapping, first-row state and gaps, dense ids, chunk-size invariance, error cases, and `make_split` accepting the output.
- `test_dynamic_retention_gpu.py`, `test_dynamic_retention_gpu_verify.py`: parity with the original within tolerance (dyn, mu, FFBS with fixed eps, predict given the same samples), PG sampler mean against tanh(c/2)/(2c), statistical parity on synthetic data, output types and shapes. Bucketed FFBS is checked against a single bucket and against a NumPy reference, and padding does not leak into real events. This covers the padded-vs-bucketed concern in `docs/reports/VALIDATION_REPORT.md`.
- `test_laglogit.py`: every scored event is predicted, cold cards are predicted, and own and later outcomes do not change a prediction.
- `test_ours_backend.py`: backend switch and GPU fit and predict.
- `test_fsrs_parallel.py`: parallel predict matches serial.
- `test_run_duolingo.py`: default model list unchanged, opt-in keys allowed, flags, model_config, resume config, and an end-to-end Duolingo run on a synthetic csv.
- `test_task52_54_verify.py`: adversarial checks across Tasks 5.2 to 5.4.

These tests were not re-run in this documentation pass. The results below come from the recorded Kaggle run.

## The Kaggle run

- Data: `settles.acl16.learning_traces.13m.csv.gz`, all users, no card cap, `min_user_rows = 1`, seed 0.
- Models: `ours_gpu`, `hlr`, `laglogit`, `fsrs_default`, `fsrs_opt`. `ours_gpu` used 2000 sweeps, 1000 burn-in and thin 10 (100 kept draws), in float64, on a GPU.
- Two notebooks: the GPU notebook `amalverma27/notebook2ef9000f5a` produced `ours_gpu`, `hlr` and `laglogit`. The CPU notebook `notebooke7d641abde` produced `fsrs_default` and `fsrs_opt`.
- The two sets were merged locally with `--resume`. This wrote `predictions.parquet` (1,189,355 rows) and `run_info.json`.
- Output folder: `results/duolingo_full/`. Logs are in `results/duolingo_full/kaggle_logs/`.
- Long run (sensitivity check): `ours_gpu` at 8000 sweeps, 6000 burn-in, thin 20, in `results/duolingo_full_long/` (log `run_gpu_long.log`). Test log loss 0.2679 against 0.2682 at 2000 sweeps, Brier 0.0714 against 0.0715, prediction correlation 0.992.
- Git commit `ea51d42`, with `dirty: true`.

## Results

- Data: 115,222 users, 5,861,629 cards, 12,854,226 events (6,468,512 train, 6,385,714 test), 1,189,355 scored, recall rate 0.920.
- Overall log loss: Ours (GPU port) 0.268, LagLogit + dt 0.291, FSRS-6 (optimised) 0.300, FSRS-6 (default) 0.302, HLR 0.418. Ours also has the best Brier, AUC (0.656), ECE and RMSE. The ordering is the reverse of the Anki result.
- Paired bootstrap: Ours minus FSRS-6 (optimised) is -0.032 [-0.033, -0.031] in log loss, and excludes zero.
- Full tables and CIs are in `docs/EVALUATION.md` Section 10 and `docs/report/fsrs_benchmark_report.tex`.

## Known limits

- **FSRS-6 (optimised) is nearly the default model.** 115,110 of 115,222 users fell back to default weights, because the optimiser needs 512 informative train reviews per user.
- **64,254 users have zero train rows.** They are predicted from population-level parameters.
- **Lambda is not identified.** At 2000 sweeps, lambda has split R-hat 2.48 and ESS 3.0 on one chain. The 8000-sweep run drifts further (lambda 0.95 at sweep 1000, 1.24 at 4000, 1.73 at 8000; R-hat 2.33), and sigma2 drifts with it (mean 0.60 to 1.14). Lambda and sigma2 should not be interpreted on Duolingo. The predictions are robust: test log loss 0.2682 and 0.2679, Brier 0.0715 and 0.0714. The headline uses the 2000-sweep run.
- **The label is a threshold.** `y` is `p_recall >= 0.5`, and FSRS sees only Good or Again.
- **HLR has no lexeme features.**
- **The data span is about 12 days.** The 15-60d and 61d+ bins are empty.
- **FSRS predict ran serially in the full run.** `workers = 1` was recorded, so the parallel path was not used at full scale.
- **The GPU device.** Both Kaggle GPU runs used "GPU T4 x2" (the notebook accelerator setting; the P100 was not offered), in float64 on one T4. The plan originally named the P100.
- **Resume config.** `--thin` is not part of the resume config check (see `docs/BACKLOG.md`).
- **Single seed and single chain.**

## For the next phase

- Phase 6 starts from the Phase 5 result that the Anki ranking (FSRS-opt > ours > HLR) does not hold on Duolingo. Ours beats FSRS-opt there, but the caveats above apply.
- The Anki Ours numbers still need a multi-chain run before they are final. On Duolingo, the predictive results are robust to the non-convergence, but lambda is not interpretable.
