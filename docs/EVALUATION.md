# Evaluation protocol: FSRS-6 vs our dynamic retention model vs HLR

## 1. Goal and summary

We compare five recall predictors on the same held-out review events:

| Key | Name in tables | What it is |
|---|---|---|
| `fsrs_default` | FSRS-6 (default) | py-fsrs 6.3.2 with its default 21 weights |
| `fsrs_opt` | FSRS-6 (optimised) | py-fsrs Optimizer fitted per user on training reviews |
| `hlr` | HLR (Settles & Meeder) | Half-life regression, trained as in the official code |
| `ours` | Ours (dynamic, learned λ) | `DynamicRetentionModelDT` (Bayesian, Gibbs sampling) |
| `ours_kappa` | Ours (+κ) | the same model with `use_kappa=True` |

Data: the `open-spaced-repetition/anki-revlogs-10k` review logs (Section 2). Metrics: log loss, Brier score, AUC, expected calibration error (ECE), mean bias and RMSE over bins (the overall table in `report.md` shows all of them; its per-gap tables omit RMSE, which `metrics.csv` has), with user-clustered bootstrap confidence intervals on paired differences (Section 6).

This document describes the protocol and the code. The numerical results are in Section 9 (Anki) and Section 10 (Duolingo).

### 1.1 Glossary

- **Rating.** Anki's answer buttons: 1 = Again (failed), 2 = Hard, 3 = Good, 4 = Easy. Recall (`y = 1`) means rating 2, 3 or 4.
- **Gibbs sampling.** An MCMC method that updates each latent variable in turn from its conditional distribution. Here the sampler alternates between the latent memory states, the hyper-parameters and the per-user and per-card effects.
- **Pólya-Gamma augmentation.** A data-augmentation trick that makes a logistic likelihood conditionally Gaussian, so the latent states can be sampled exactly.
- **FFBS.** Forward-filtering backward-sampling: draws a whole latent trajectory of a sequence from its exact conditional distribution, in one forward and one backward pass.
- **κ (kappa).** In `ours_kappa`, the decay rate is `λ · (1 + n)^(-κ)`, where `n` is the number of earlier correct recalls of the card (the model's covariate is `log(1 + n)`). With `use_kappa=False`, κ is fixed at 0.
- **λ (lambda).** The base decay rate of the model's memory, learned from data.
- **Posterior draws.** Samples of the model's parameters and latent states from the posterior, after burn-in and thinning. `predict()` uses a subset of them (at most 100).
- **Cold path.** The prediction rule the model uses for a user or card with no TRAIN history, which has no fitted per-user or per-card effect.
- **Split R-hat.** The Gelman-Rubin statistic computed on the two halves of one chain. Values near 1 indicate that the two halves agree. It is not a check between separate chains.
- **ESS.** Effective sample size: the number of independent-equivalent draws in an autocorrelated chain.

### 1.2 References

- Settles, B. and Meeder, B. (2016). A Trainable Spaced Repetition Model for Language Learning. ACL 2016.
- Duolingo. halflife-regression (MIT licence), `experiment.py`. https://github.com/duolingo/halflife-regression
- open-spaced-repetition. py-fsrs 6.3.2 (FSRS-6 scheduler and optimiser). https://github.com/open-spaced-repetition/py-fsrs
- open-spaced-repetition. anki-revlogs-10k dataset (gated). https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k
- Polson, N. G., Scott, J. G. and Windle, J. (2013). Bayesian inference for logistic models using Pólya-Gamma latent variables. JASA.
- Carter, C. K. and Kohn, R. (1994). On Gibbs sampling for state space models. Biometrika.

## 2. Dataset and sampling

- **Source.** Hugging Face dataset `open-spaced-repetition/anki-revlogs-10k`. It is gated: the user needs a Hugging Face token and must accept the dataset licence on its page.
- **Files used.** One parquet file per user, `revlogs/user_id=<u>/data.parquet`. The loader requires the columns `card_id`, `day_offset`, `rating`, `state` and `elapsed_days`, and uses `elapsed_seconds` for `dt_days` when that column is present. Other columns in the file are ignored.
- **User sampling.** `download_users(n_users, seed, out_dir)` takes the sorted list of user ids in the repository and draws `n_users` of them with `numpy.random.default_rng(seed)`, without replacement. `out_dir` is the data directory (`data/anki10k` by default). The sorted id list is cached in `<out_dir>/user_ids.json` on the first successful listing; later runs read the cache, so no network call is needed once the files exist. Files already on disk are not downloaded again.
- **Card cap.** `make_split(max_cards, seed)` keeps at most `max_cards` cards per user, sampled with `default_rng(seed)` from the sorted card ids. Cards not chosen are dropped entirely, so each card keeps its full history. The same cards go to every model. The cap is needed because the Gibbs sampler's cost grows with the number of cards and their sequence lengths. The full run uses `max_cards = 1000`. The timing study on 5 users (`--iters 200 --burn 100`) gave a total run time of 184 s at `max_cards = 300` and 340 s at `max_cards = 1000`. At 300 cards, one of the five users fell back to default FSRS weights; at 1000 cards, none did. That is why 1000 was chosen.

## 3. Event definition

One row of a user's revlog is one review. Rows keep their file order, which is chronological within a user. The rules, from `fsrs_benchmark/data.py` (`load_user`):

1. **Drop `rating == 0`.** These are manual reschedules, not recall attempts.
2. **Keep `state == 4`.** These rows are filtered-deck reviews and carry real ratings (1-4) and real durations. An earlier version dropped them, and FSRS then saw wrong gaps: one example had a gap of 515 days where the real gap was 1 day.
3. **Drop cards whose first kept review is not `state == 0` (new).** Their earlier history is truncated, so the memory state of the card cannot be rebuilt.
4. **Label.** `y = 1` if `rating > 1` (Hard, Good or Easy), else `0`.
5. **Gaps are recomputed over kept rows only,** so every model sees the same gap.
   - `elapsed_days`: the `day_offset` difference to the card's previous kept review. It is -1 on a first review. The file's own value is kept as `raw_elapsed_days`.
   - `dt_days`: the sum of raw `elapsed_seconds` since the previous kept row, divided by 86400. This includes the seconds of any dropped rows in between. It falls back to `elapsed_days` if any of those seconds are missing. It is NaN on a first review.
6. `is_first` marks the first kept review of each card. `review_idx` is the position within the user, and `event_id = user_id * 10^9 + review_idx` is unique.

## 4. Split and leakage guarantees

**Split.** `make_split` applies the card cap first, then computes a per-user cutoff: `cutoff_day` is the 80th percentile (linear interpolation) of `day_offset` over that user's kept reviews. A review is TRAIN if `day_offset < cutoff_day`, and TEST otherwise. Model parameters are fitted on TRAIN rows only.

**Scored events.** A TEST event is scored when it is not a card's first review and `elapsed_days >= 1`. This is the srs-benchmark convention of scoring inter-day reviews. Same-day reviews and first reviews are still fed to the models as history, but they are never scored.

**Prediction is one-step-ahead.** To predict scored event k, a model may use all history of that card before k: its TRAIN reviews and any earlier TEST reviews, including their outcomes. It may not use event k or anything after it. Fitted parameters are frozen during the test period.

**Tests that enforce this** (all under `fsrs_benchmark/tests/`):

| Guarantee | Test |
|---|---|
| No train event at or after the cutoff; cutoff is the 80th percentile; scored rule; card cap applied before the cutoff; determinism; unique event ids | `test_split.py`: `test_train_is_strictly_before_cutoff_per_user`, `test_cutoff_is_80th_percentile_of_kept_reviews`, `test_scored_rules`, `test_card_cap_applied_before_cutoff`, `test_deterministic_for_fixed_seed_and_seed_changes_cards`, `test_event_id_unique_int64_and_sorted` |
| Own and later outcomes never change a prediction | `test_fsrs_model.py`: `test_prediction_ignores_own_and_later_ratings`; `test_ours.py`: `test_prediction_ignores_own_and_later_outcomes`; `test_hlr.py`: `test_prediction_ignores_own_and_later_outcomes`; `test_ours_verify.py`: `test_own_and_later_outcomes_do_not_move_p` |
| Fit ignores test rows (including their gaps) | `test_fsrs_model.py` and `test_hlr.py`: `test_fit_ignores_test_rows`; `test_hlr_verify.py`: `test_fit_ignores_test_rows_including_their_gaps`; `test_run_verify.py`: `test_fit_does_not_see_test_labels` (flips every TEST label and checks that all five models give identical predictions) |
| Ids for `ours` come from TRAIN rows only; unseen users and cards take the cold path | `test_ours.py`: `test_unseen_user_and_card_take_the_cold_path`; `test_ours_verify.py`: `test_fit_and_predict_ids_agree_for_train_cards` |
| Real-data fit sends only TRAIN rows to the optimiser | `test_fsrs_model_verify.py`: `test_fit_on_real_users_sends_only_train_rows` |

## 5. Models

### 5.1 FSRS-6 (py-fsrs 6.3.2), both variants

- **Replay.** For each card, reviews are replayed in `review_idx` order through py-fsrs' `Scheduler` with `enable_fuzzing=False`. Review times are synthetic: a fixed UTC epoch plus `day_offset` days, plus the position of the review within its card in seconds. The seconds keep the order strict and keep same-day reviews distinct. py-fsrs measures gaps in whole days, so the gap it sees equals the `day_offset` difference.
- **Prediction.** Before a scored event is applied, its retrievability is read with py-fsrs' `get_card_retrievability`. That value is the prediction. The card's memory state is then updated with the event's own rating.
- **Default variant.** `optimize=False` uses py-fsrs' default parameters for everyone.
- **Optimised variant.** `optimize=True` runs py-fsrs' `Optimizer` per user on that user's TRAIN reviews only. Two py-fsrs limits apply:
  - the Optimizer trains on the first 64 reviews of each card (`max_seq_len = 64`);
  - py-fsrs returns the default weights, without a warning, when a user has fewer than 512 informative train reviews. We apply the same rule up front (`MIN_INFORMATIVE_TRAIN = 512`, counting non-same-day reviews) so the fallback is recorded, and we also record any user whose optimiser returned the defaults. Fallback users and reasons are in `run_info.json`.
- **Parallel fit.** `FSRSModel(n_workers > 1)` runs the per-user optimisations in a spawn-context process pool. Workers set torch to one thread. Each user is independent, and a test checks that parameters and predictions match the serial fit.

### 5.2 HLR (Settles & Meeder 2016), written from the official code

- **Model.** `p = 2^(-dt/h)` with `h = 2^(theta · x)`, where `x = [sqrt(1 + n_right), sqrt(1 + n_wrong), 1]` counts the card's PRIOR kept reviews of that user. Same-day and first reviews count as history.
- **Training.** A port of `HalfLifeRegression.train_update` in `duolingo/halflife-regression` `experiment.py` (MIT): SGD with the paper's loss (squared error on p plus `hlwt` times squared error on h, with L2 on the weights). Constants: `lrate = 0.001`, `hlwt = 0.01`, `l2wt = 0.1`, `sigma = 1`, `h` clipped to [15 min, 274 days], `p` clipped to [1e-4, 1 - 1e-4]. Per-feature learning rates and the recall-halving rule are as in the official code. The SGD step is checked against the official arithmetic to 1e-12.
- **Deviations** (documented in `hlr.py`):
  - The official code has per-lexeme indicator features. These are not in the Anki data, so they are not used.
  - The official code makes one shuffled pass. We repeat seeded shuffled passes until the largest weight change in one pass is below 1e-4, with at most 1000 passes. The number of passes is recorded as `epochs` in `run_info.json`. If it equals 1000, the fit did not converge.
- **Training rows.** TRAIN, non-first, `elapsed_days >= 1` rows.

### 5.3 Ours: `DynamicRetentionModelDT`, with and without κ

- **Code.** The unmodified `AnkiFirstRun/dynamic_retention_dt.py` is imported through an adapter (`fsrs_benchmark/models/ours.py`). Nothing in that file was changed.
- **Sequences.** One sequence per (user, card), built from TRAIN rows only in `review_idx` order. `r` is the outcome, `x` is the lagged outcome with `x[0] = 0`, `dt` is `dt_days` with NaN replaced by 1.0.
- **Ids.** User and card ids are remapped to dense integers from TRAIN rows only. A user or card with no TRAIN rows maps to the padding index, so the model's cold path applies to it.
- **Sampler.** `run(n_iter=iters, burn_in=burn, thin=4)` with `iters` and `burn` set by the run (defaults 2000 and 1000 in the CLI; the full run's values are in `run_info.json`). The posterior is sampled by Gibbs sweeps with forward-filter backward-sampling (FFBS) steps in the model file.
- **Prediction.** `predict()` runs the model's own one-step filtered prediction on the TEST sequences. Each card's TEST events are in order, and history events stay in the sequence. Predictions come from the posterior draws the model selects in `predict()` (at most `max_draws = 100`, evenly spaced over the saved chain). Only scored rows are kept.
- **Variant.** `ours_kappa` is the same model with `use_kappa=True`.

## 6. Metrics

All metrics are written from scratch in `fsrs_benchmark/metrics.py`. y is 0 or 1, and p is the predicted recall probability. N is the number of scored events.

| Metric | Definition |
|---|---|
| Log loss | `-(1/N) Σ [y log p + (1-y) log(1-p)]`, with p clipped to [1e-6, 1 - 1e-6] |
| Brier | `(1/N) Σ (p - y)²` |
| AUC | Rank-based (Mann-Whitney): the probability that a random positive event gets a higher p than a random negative one, with ties counted as one half |
| ECE | Ten equal-width bins on p: `Σ_b (n_b / N) · |mean p_b - mean y_b|` |
| Mean bias | `mean(p) - mean(y)`. Positive means too optimistic |
| RMSE (bins) | Twenty equal-width bins: `sqrt( Σ_b (n_b / N) · (mean p_b - mean y_b)² )` |
| Reliability curve | Per bin: `mean p_b` against `mean y_b`, with the bin counts |

**Gap bins.** Scored events are grouped by `elapsed_days` (whole days, integer edges): `1d`, `2-3d`, `4-14d`, `15-60d`, `61d+`. Because scored events have `elapsed_days >= 1`, the first bin is exactly one day.

**Paired differences.** For a model m and reference r, `diff = metric(m) - metric(r)`. Users are resampled with replacement (1000 resamples, seed 0). Each resample takes all events of the drawn users, and both models are resampled with the same users. The 95% CI is the 2.5 and 97.5 percentiles of the bootstrap differences. Reference models are `fsrs_opt` and `ours`. Metrics are log loss, Brier, AUC and ECE.

**Why cluster by user.** Reviews from one person are correlated (the same memory habits and the same schedule). Resampling single events would treat them as independent and give intervals that are too narrow. Resampling users keeps that correlation.

## 7. Reproducibility

**Environment.**

```
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python numpy pandas pyarrow numba scipy matplotlib pytest huggingface_hub "fsrs[optimizer]==6.3.2"
```

The optimiser extra pulls in torch (CPU build is enough). Exact versions used for a run are in its `run_info.json`.

**Data access.** Log in with `huggingface-cli login` (or set `HF_TOKEN`), and accept the licence of `open-spaced-repetition/anki-revlogs-10k` on its Hugging Face page.

**Run.**

```
.venv/bin/python -m fsrs_benchmark.run --run_name <name> --n_users 100 --max_cards 1000 --seed 0 --iters 2000 --burn 1000 --workers 8
```

`--models` selects a subset (keys in Section 1). `--resume` reuses finished per-model predictions. Each `pred_<model>.parquet` has a `pred_<model>.json` sidecar with the config it came from: `n_users`, `max_cards`, `seed`, the SHA-256 of the scored event ids, and `iters` and `burn` for the `ours` models. A resumed run refuses any file whose sidecar differs from the current config, and says to delete the file or drop `--resume`.

**Report.**

```
.venv/bin/python -m fsrs_benchmark.report --run_dir results/<name>
.venv/bin/python -m fsrs_benchmark.supplementary --run_dir results/<name>
```

The second command writes `supplementary.json` (the FSRS-opt fallback count, the `ours` vs `ours_kappa` correlation and the `ours` reliability bins). It is recomputed from `events.parquet` and `predictions.parquet`, so it needs the local-only parquet files. Its output for `full_u100_c1000` matches the committed `supplementary.json`, and `fsrs_benchmark/tests/test_supplementary.py` checks that.

**Tests.** `.venv/bin/python -m pytest fsrs_benchmark/tests -q`. Tests that need the real revlogs (`data/anki10k/revlogs/user_id=...`) skip themselves if those files are missing.

**Outputs.** Each run folder `results/<name>/` contains:
- `events.parquet` (the full event table), `pred_<model>.parquet` and `.json`, `predictions.parquet` (scored rows only);
- `diagnostics_ours.json`, `diagnostics_ours_kappa.json`: MCMC summaries for the two `ours` variants, written only when those models are fitted (not on `--resume`);
- `run_info.json`: all arguments; git commit hash and a dirty flag; package versions (fsrs, numpy, pandas, torch, numba, Python); counts (users, cards, events, train, test, scored); users with zero train rows; FSRS-opt fallback count and user ids; HLR epochs; per-model fit and predict seconds; total seconds;
- from the report step: `metrics.csv`, `paired.csv`, `users.csv`, `user_wins.csv`, `reliability.png`, `gapbins_logloss.png`, `report.md`;
- from the supplementary step: `supplementary.json`.

For the full run `full_u100_c1000`, `run_info.json` has no FSRS-opt fallback fields and no HLR epochs, because those models were reused with `--resume` and not refitted. Section 9.7 gives the fallback count recomputed from the rule.

**Determinism.** The user sample, the card sample, the HLR shuffles and the `ours` sampler all take fixed seeds (`--seed`, default 0). The FSRS optimiser was checked to give the same parameters serially and in parallel.

**Runtime.** Fit and predict times per model are in `run_info.json`. Timing study on 5 users with `--iters 200 --burn 100`: total 184 s at `max_cards = 300` and 340 s at `max_cards = 1000`. On those 5 users, per-model fit times at 1000 cards were about 200 s for `fsrs_opt` with serial fitting and 115 s with `--workers 8`, and 49 s and 44 s for the two `ours` variants. The machine was shared with other jobs, so these are indicative. These numbers are for 5 users. The full 100-user run took much longer; its timings are in Section 9.6.

## 8. Limitations

- **Subsample.** The benchmark uses `--n_users` users (100 in the full run), with at most `max_cards` cards each (1000 in the full run). Results should not be compared directly with published numbers from other protocols.
- **FSRS pipeline.** FSRS-6 is run through py-fsrs 6.3.2 in Python, not through the srs-benchmark pipeline. Its data handling and split differ: the protocol here uses the per-user time cutoff above, not srs-benchmark's 5-fold TimeSeriesSplit, and it does not apply srs-benchmark's outlier filtering. The optimiser's limits (first 64 reviews per card, 512 informative reviews per user) apply.
- **HLR.** The model lacks the per-lexeme features of the original implementation. The comparison is with this simplified HLR.
- **MCMC convergence: not converged in the full run.** At 2000 iterations (1000 burn-in), the lambda R-hat is about 2.4 and its ESS about 3, against a target of R-hat below 1.01 and ESS in the hundreds (Section 9.8). The `ours` scores are therefore provisional. A longer run with several chains is future work.
- **Single seed and single chain.** The full run uses seed 0 only, and one MCMC chain per `ours` variant. There is no spread across seeds or chains to report, so the paired intervals above reflect only user sampling in the bootstrap, not run-to-run variation.
- **MCMC detail.** The `ours` models are sampled from a single MCMC chain per fit. Each run reports, per variant, the posterior mean, 95% interval, effective sample size (ESS) and split R-hat for the hyper-parameters (`diagnostics_ours.json`, `diagnostics_ours_kappa.json`, rendered in the report's "MCMC diagnostics" section). Split R-hat on one chain can detect within-chain drift but not disagreement between chains. Multiple chains are not run. Lambda acceptance is recorded for every sweep, but kappa acceptance is not exposed by the sampler, so it is not reported. ESS and R-hat are NaN or unreliable when the chain is very short (the smoke runs).
- **Scored events.** Same-day reviews and first reviews are never scored, so the results say nothing about them.
- **Bootstrap precision.** Cluster bootstrap intervals are only meaningful with many users. With a handful of users the intervals are wide or degenerate.
- **Event definition.** Cards whose first kept review is not new are dropped, and so are rating-0 rows. Results describe the cards that remain, not the whole collection.

## 9. Results

### 9.1 Run configuration

Run name `full_u100_c1000`: 100 users sampled with seed 0, at most 1000 cards per user, 2000 Gibbs iterations with 1000 burn-in for the two `ours` variants. The command is the one in [Section 7](#7-reproducibility).

| Quantity | Value |
|---|---|
| Users | 100 |
| Cards | 99,074 |
| Review events (all) | 768,658 |
| Train events | 606,364 |
| Test events | 162,294 |
| Scored events | 95,400 |
| Recall rate on scored events | 0.831 |

Git commit `0b6259e` with uncommitted changes (`dirty: true`), so the exact code state is not fully pinned.

### 9.2 Overall results

All five models are scored on the same 95,400 events. Best value in each column is in bold. Lower is better for log loss, Brier, ECE and RMSE; higher is better for AUC; for mean bias, closest to zero is best.

| Model | Log loss | Brier | AUC | ECE | RMSE (bins) | Mean bias |
|---|---|---|---|---|---|---|
| FSRS-6 (default) | 0.396 | 0.121 | 0.745 | 0.012 | 0.017 | 0.011 |
| FSRS-6 (optimised) | **0.379** | **0.117** | **0.777** | **0.009** | **0.014** | **-0.002** |
| HLR (Settles & Meeder) | 0.680 | 0.170 | 0.555 | 0.146 | 0.176 | 0.060 |
| Ours (dynamic, learned λ) | 0.422 | 0.131 | 0.692 | 0.025 | 0.032 | 0.011 |
| Ours (+κ) | 0.422 | 0.131 | 0.693 | 0.026 | 0.032 | 0.011 |

FSRS-6 (optimised) is best on every overall metric. Our model is clearly better than HLR on every metric, but worse than both FSRS-6 variants.

### 9.3 Results by gap

The gap is the number of days since the same card was last reviewed. Each bin has the same events for every model, so the recall rate is the same across models within a row.

Log loss by gap bin (lower is better):

| Gap | n | FSRS-6 (default) | FSRS-6 (optimised) | HLR | Ours | Ours (+κ) |
|---|---|---|---|---|---|---|
| 1d | 16,983 | 0.409 | **0.396** | 0.942 | 0.423 | 0.423 |
| 2-3d | 16,080 | 0.402 | **0.389** | 0.734 | 0.426 | 0.426 |
| 4-14d | 29,204 | 0.369 | **0.361** | 0.555 | 0.398 | 0.398 |
| 15-60d | 15,626 | 0.401 | **0.374** | 0.473 | 0.419 | 0.419 |
| 61d+ | 17,507 | 0.418 | **0.389** | 0.771 | 0.460 | 0.460 |

AUC by gap bin (higher is better):

| Gap | FSRS-6 (default) | FSRS-6 (optimised) | HLR | Ours | Ours (+κ) |
|---|---|---|---|---|---|
| 1d | 0.748 | **0.769** | 0.658 | 0.708 | 0.708 |
| 2-3d | 0.730 | **0.755** | 0.637 | 0.676 | 0.677 |
| 4-14d | 0.737 | **0.760** | 0.616 | 0.680 | 0.681 |
| 15-60d | 0.725 | **0.781** | 0.599 | 0.703 | 0.705 |
| 61d+ | 0.771 | **0.817** | 0.628 | 0.763 | 0.764 |

![Log loss by gap bin](../results/full_u100_c1000/gapbins_logloss.png)

The ordering of the models does not change across gap bins: FSRS-6 (optimised) is best, then FSRS-6 (default), then our model, then HLR, on both log loss and AUC. Our model's log-loss gap to FSRS-6 (optimised) is widest at 61d+ (0.460 against 0.389), and its AUC gap is narrowest there (0.763 against 0.817).

### 9.4 Calibration

![Reliability curve](../results/full_u100_c1000/reliability.png)

The curve pools all gaps, with 10 equal-width bins on the predicted recall. The FSRS-6 variants lie close to the diagonal, with FSRS-6 (default) somewhat below it in the middle bins. Our model sits clearly above it in the middle bins. In the 0.5 to 0.6 bin it predicts 0.555 on average, and 0.629 is observed (`supplementary.json`). HLR's pooled curve is far from the diagonal across the low and middle bins. The per-gap bias numbers show its direction more clearly. Split by gap, HLR over-predicts at short gaps (mean bias +0.170 at 1d) and under-predicts at 61d+ (mean bias -0.249). Our model over-predicts at 61d+ (mean bias +0.090), while at 1d its bias is -0.006.

### 9.5 Paired differences and user wins

Each difference is model minus reference, computed on the same events. The 95% CI comes from a user-clustered paired bootstrap (1000 resamples of users, seed 0). An asterisk marks a CI that excludes zero.

| Comparison | Log loss diff [95% CI] | Brier diff [95% CI] | AUC diff [95% CI] |
|---|---|---|---|
| FSRS-6 (default) - FSRS-6 (optimised) | 0.017 [0.008, 0.025] * | 0.004 [0.002, 0.006] * | -0.032 [-0.047, -0.016] * |
| Ours - FSRS-6 (optimised) | 0.043 [0.033, 0.055] * | 0.014 [0.011, 0.019] * | -0.084 [-0.110, -0.064] * |
| FSRS-6 (default) - Ours | -0.026 [-0.042, -0.013] * | -0.011 [-0.015, -0.007] * | 0.053 [0.023, 0.087] * |
| Ours (+κ) - Ours | -0.000 [-0.001, 0.000] | -0.000 [-0.000, 0.000] | 0.001 [-0.000, 0.002] |

Our model has a higher (worse) log loss than FSRS-6 (optimised) by 0.043, and a higher log loss than FSRS-6 (default) by 0.026. Both paired CIs exclude zero. Our model's ECE is higher than FSRS-6 (optimised) by 0.016 [0.001, 0.027]. The FSRS-6 default vs optimised ECE difference is not significant (0.003 [-0.010, 0.019]). The `FSRS-6 (default) - Ours` row is written with the sign used in `paired.csv`, so a negative value means FSRS-6 (default) is better.

User wins: the number of the 100 users on which each model has a lower log loss than the reference (`user_wins.csv`).

| Model (wins) | Reference | Users won | Share |
|---|---|---|---|
| FSRS-6 (default) | FSRS-6 (optimised) | 28 | 0.280 |
| HLR | FSRS-6 (optimised) | 7 | 0.070 |
| Ours | FSRS-6 (optimised) | 22 | 0.220 |
| Ours (+κ) | FSRS-6 (optimised) | 25 | 0.250 |
| FSRS-6 (default) | Ours | 68 | 0.680 |
| FSRS-6 (optimised) | Ours | 78 | 0.780 |
| HLR | Ours | 14 | 0.140 |
| Ours (+κ) | Ours | 52 | 0.520 |

### 9.6 Timings

Fit and predict wall-clock seconds on the full run. Three models were reused from an interrupted run with `--resume`, so `run_info.json` has no timing for them. Their values are taken from `run.log`, which is why they have one decimal place. `run_info.json` marks `ours` as `resumed: true`. That is because the final invocation (which fitted `ours_kappa`) reused the `ours` predictions from the previous invocation, which started after the 19:47 restart and wrote them at 04:02, according to `run.log`.

| Model | Fit (s) | Predict (s) | Source |
|---|---|---|---|
| FSRS-6 (default) | 0.0 | 99.0 | `run.log` |
| FSRS-6 (optimised) | 652.7 | 98.9 | `run.log` |
| HLR | 45.0 | 0.3 | `run.log` |
| Ours (dynamic, learned λ) | 27529.624 | 1266.342 | `run_info.json` |
| Ours (+κ) | 26129.754 | 1339.690 | `run_info.json` |

The `total_seconds` value of 27478.179 s in `run_info.json` is the wall-clock time of the final invocation only (the `ours_kappa` run). It is not the time of the whole project. These are single wall-clock measurements from one run, so they are indicative only.

### 9.7 FSRS-6 (optimised) fallback users

`run_info.json` does not record the fallback count for this run, because `fsrs_opt` was reused with `--resume`. The `events.parquet` and `predictions.parquet` files used here are local-only (git-ignored, `results/**/*.parquet`), and the Section 7 command regenerates them. The summary values are committed in `results/full_u100_c1000/supplementary.json`. The count below was recomputed from the rule in `fsrs_benchmark/models/fsrs_model.py` (`MIN_INFORMATIVE_TRAIN = 512`, counting train rows that are not first reviews and have `elapsed_days >= 1`) applied to the train rows of `results/full_u100_c1000/events.parquet`. It was not taken from the fit.

Result: **1 of 100 users** has fewer than 512 informative train reviews (user 7171, with 292 informative train reviews; `supplementary.json`). The other 99 users are above the threshold, with a median of 2,674 informative train reviews. The rule above does not cover the second fallback case in the code (the optimiser returning the default weights), which can only be seen during a refit, so the true fallback count may be higher.

### 9.8 MCMC convergence

**The chains did not converge at 2000 iterations.** Across both `ours` variants, the lambda R-hat is about 2.4 and its ESS is about 3. The usual targets are R-hat below 1.01 and ESS in the hundreds. These are split R-hat values on one chain (there is no second chain to compare), and each run stores 250 kept draws. Our model's scores in Section 9.2 to 9.5 are therefore **provisional**.

Posterior summaries from `diagnostics_ours.json` (`Ours`) and `diagnostics_ours_kappa.json` (`Ours (+κ)`). The interval is the 95% posterior interval.

| Variant | Parameter | Mean | 95% interval | ESS | Split R-hat |
|---|---|---|---|---|---|
| Ours | lam | 1.871 | [1.312, 2.646] | 3.046 | 2.439 |
| Ours | sigma2 | 0.204 | [0.189, 0.231] | 3.341 | 2.071 |
| Ours | b0 | 2.332 | [2.314, 2.343] | 3.940 | 1.854 |
| Ours | tau2_beta | 0.499 | [0.482, 0.513] | 4.826 | 1.879 |
| Ours (+κ) | lam | 1.455 | [0.898, 2.228] | 2.971 | 2.376 |
| Ours (+κ) | kappa | -0.160 | [-0.211, -0.076] | 3.500 | 1.939 |
| Ours (+κ) | sigma2 | 0.257 | [0.245, 0.277] | 9.306 | 1.161 |
| Ours (+κ) | b0 | 2.317 | [2.296, 2.331] | 4.434 | 1.283 |
| Ours (+κ) | tau2_beta | 0.480 | [0.465, 0.492] | 8.953 | 1.257 |

The other hyper-parameters mix better (for `Ours`, `g0`, `tau2_alpha` and `tau2_gamma` have ESS of 81, 250 and 143, and R-hat of 1.001, 0.996 and 1.064). The lambda Metropolis-Hastings acceptance rate is 0.240 after burn-in (0.236 over all sweeps) for `Ours`, and 0.271 after burn-in (0.259 over all sweeps) for `Ours (+κ)`. The acceptance rate for kappa is not recorded by the sampler.

**Why the scores still look usable, and why this is not proof.** The two variants give different lambda posteriors (means 1.871 and 1.455) and yet their predictions are almost the same. Over the 95,400 scored events, the correlation between `ours` and `ours_kappa` predictions is 0.992 (`supplementary.json`, computed from the local-only `predictions.parquet`). Single events can still differ by up to 0.094, and every overall metric agrees to about three decimals (Section 9.2). This suggests the predictions depend little on the poorly mixed parameters. It does not show that the chains have converged, and a flat prediction surface is only one explanation.

**Recommendation.** Run a longer job, for example 8,000 to 10,000 iterations with several chains from different starts, and report R-hat between chains. That is the step needed before the `ours` numbers can be treated as final.

### 9.9 Interpretation

- FSRS-6 (optimised) is the best model on every overall metric and on every gap bin for log loss and AUC.
- Our model is much worse than FSRS-6 (optimised) and somewhat worse than FSRS-6 (default), but far better than HLR. Given the non-convergence in Section 9.8, the size of the gap to FSRS-6 is a provisional result.
- κ adds nothing measurable here. The paired difference is about 0.000 in log loss, with a CI that includes zero, and the two variants' predictions correlate at 0.992. The fitted kappa of about -0.16 is not interpretable, because its chain did not converge.
- HLR's weak AUC of 0.555 is in line with the AUC of about 0.54 that the original paper reports for HLR, which is expected here because this run has no lexeme features. The original paper's figure of about 0.54 is quoted from memory and was not in any file in this repo, so check it against the paper before submission.

## 10. Duolingo (full dataset)

This section runs the same protocol on Duolingo's public learning traces (Settles & Meeder 2016), using every user and every card. Only the data layer is new. The split, the scoring rule, the metrics, the runner and the report are the same code as for Anki. The GPU port (`ours_gpu`) and LagLogit + dt (`laglogit`) are the models added for this run. The run folder is `results/duolingo_full/`.

### 10.1 Event definition

- **Source.** `data/duolingo/settles.acl16.learning_traces.13m.csv.gz` (Harvard Dataverse, doi:10.7910/DVN/N8XJME, CC BY-NC 4.0). The loader is `fsrs_benchmark/duolingo.py`. It returns the columns of `data.load_user`, so `split`, the models, `run` and `report` are reused.
- **Event.** One row of the file is one review: one practice session of one lexeme by one user. A card is a (user, lexeme) pair.
- **Order.** Within a user, rows are sorted by timestamp, then lexeme id, then file row (a stable sort). `review_idx` is dense within the user.
- **Label.** `y = 1` if `p_recall >= 0.5`, else `0`. `p_recall` is the value the dataset records for that session, so `y` is a threshold of it and not a single yes or no answer.
- **FSRS rating.** `rating = 3` (Good) when `y = 1`, else `1` (Again). FSRS therefore sees only two grades on this data.
- **Time.** `day_offset = floor((timestamp - T0) / 86400)`, where `T0` is the earliest timestamp in the whole file. `elapsed_days` is the day-offset difference to the card's previous row (-1 on a first row). `dt_days` is the timestamp difference to the previous row, in days (NaN on a first row).
- **First rows.** A card's first row in the log starts its replay and is never scored, because its history before the log window is unknown.
- **Unused fields.** `history_seen`, `history_correct`, the lexeme strings and the session counts are not used by any model, so every model sees only the replayed history.
- **Sample.** All users (`--n_users 0`), `--min_user_rows 1`, and no card cap (`--max_cards 1000000000`).
- **Split.** The same per-user 80th-percentile day cutoff as Section 4. A test event is scored when it is not a card's first row and `elapsed_days >= 1`.

### 10.2 Data counts

| Quantity | Value |
|---|---|
| Users | 115,222 |
| Cards | 5,861,629 |
| Review events (all) | 12,854,226 |
| Train events | 6,468,512 |
| Test events | 6,385,714 |
| Scored events | 1,189,355 |
| Recall rate on scored events | 0.920 |
| Users with zero train rows | 64,254 |
| FSRS-6 (optimised) users that kept the default weights | 115,110 |

### 10.3 Overall results

All five models are scored on the same 1,189,355 events. The best value in each column is in bold, as in Section 9.2. The `ours` and `ours_kappa` rows of Section 9 are not run here.

| Model | Log loss | Brier | AUC | ECE | RMSE (bins) | Mean bias |
|---|---|---|---|---|---|---|
| FSRS-6 (default) | 0.302 | 0.078 | 0.566 | 0.051 | 0.070 | -0.025 |
| FSRS-6 (optimised) | 0.300 | 0.078 | 0.570 | 0.050 | 0.070 | -0.026 |
| HLR (Settles & Meeder) | 0.418 | 0.079 | 0.521 | 0.072 | 0.072 | 0.072 |
| Ours (GPU port) | **0.268** | **0.071** | **0.656** | **0.014** | **0.017** | 0.004 |
| LagLogit + dt (baseline) | 0.291 | 0.076 | 0.653 | 0.020 | 0.068 | **-0.001** |

On this dataset the ordering reverses the Anki result (Section 9.2): Ours (GPU port) has the lowest log loss, and FSRS-6 (optimised) is no longer the best model. Section 10.7 explains why this does not show that our model beats FSRS in general.

### 10.4 Results by gap

The gap is the number of days since the card's previous review. The data span is about 12 days, so the 15-60d and 61d+ bins have no scored events.

| Gap | n | FSRS-6 (default) | FSRS-6 (optimised) | HLR | Ours (GPU port) | LagLogit + dt |
|---|---|---|---|---|---|---|
| 1d | 444,662 | 0.290 / 0.580 | 0.288 / 0.584 | 0.481 / 0.543 | **0.261 / 0.660** | 0.290 / 0.652 |
| 2-3d | 376,692 | 0.293 / 0.564 | 0.292 / 0.567 | 0.407 / 0.530 | **0.270 / 0.652** | 0.291 / 0.650 |
| 4-14d | 368,001 | 0.325 / 0.554 | 0.324 / 0.556 | 0.353 / 0.522 | **0.275 / 0.655** | 0.292 / 0.658 |
| 15-60d | 0 | - | - | - | - | - |
| 61d+ | 0 | - | - | - | - | - |

Each cell is log loss / AUC. Log loss is lower-is-better and AUC is higher-is-better. The per-bin ECE and mean bias are in `results/duolingo_full/report.md`. The log-loss-by-gap figure is `results/duolingo_full/gapbins_logloss.png`.

### 10.5 Paired differences and user wins

Each difference is model minus reference, on the same events. The 95% interval comes from a user-clustered paired bootstrap (1000 resamples of users, seed 0). The reference is FSRS-6 (optimised). All intervals exclude zero. Values are from `paired.csv`, rounded to four decimals.

| Comparison | Log loss diff [95% CI] | Brier diff [95% CI] | AUC diff [95% CI] |
|---|---|---|---|
| FSRS-6 (default) - FSRS-6 (optimised) | 0.0015 [0.0011, 0.0020] * | 0.0002 [0.0001, 0.0004] * | -0.0033 [-0.0048, -0.0018] * |
| HLR - FSRS-6 (optimised) | 0.1178 [0.1124, 0.1236] * | 0.0010 [0.0003, 0.0016] * | -0.0489 [-0.0555, -0.0428] * |
| Ours (GPU port) - FSRS-6 (optimised) | -0.0320 [-0.0330, -0.0311] * | -0.0064 [-0.0066, -0.0062] * | 0.0865 [0.0826, 0.0906] * |
| LagLogit + dt - FSRS-6 (optimised) | -0.0092 [-0.0117, -0.0068] * | -0.0017 [-0.0022, -0.0013] * | 0.0838 [0.0797, 0.0880] * |

User wins on log loss, over the 41,223 users with a finite log loss for both models: Ours (GPU port) beats FSRS-6 (optimised) for 30,511 users (0.740). LagLogit + dt beats it for 29,525 (0.716). HLR beats it for 23,766 (0.577). FSRS-6 (default) beats FSRS-6 (optimised) for 10 users (0.000).

### 10.6 MCMC convergence of Ours (GPU port)

The run kept 100 posterior draws (2000 sweeps, 1000 burn-in, thin 10). Its lambda Metropolis-Hastings acceptance was 0.152 after burn-in (0.160 over all sweeps). Values are from `diagnostics_ours_gpu.json`. R-hat is a split R-hat on one chain.

| Parameter | Mean | 95% interval | ESS | Split R-hat |
|---|---|---|---|---|
| lam | 1.034 | [0.956, 1.136] | 3.0 | 2.48 |
| sigma2 | 0.603 | [0.571, 0.648] | 2.9 | 2.45 |
| b0 | 3.673 | [3.649, 3.686] | 4.0 | 1.77 |
| g0 | -0.184 | [-0.206, -0.171] | 9.7 | 1.16 |
| tau2_alpha | 1.341 | [1.314, 1.363] | 20.0 | 1.01 |
| tau2_beta | 0.205 | [0.198, 0.218] | 3.1 | 2.15 |
| tau2_gamma | 0.340 | [0.302, 0.373] | 3.0 | 2.64 |

**The chain did not converge, and lambda is not identified here.** At 2000 sweeps, lambda has an R-hat of 2.48 and an ESS of 3, against a target of R-hat below 1.01 and an ESS in the hundreds. The same is true for sigma2, tau2_beta and tau2_gamma. A longer run (8000 sweeps, 6000 burn-in, thin 20, in `results/duolingo_full_long/`) shows the drift directly. Lambda is 0.95 at sweep 1000, 1.24 at sweep 4000 and 1.73 at sweep 8000, with R-hat 2.33. Sigma2 drifts with it, from a mean of 0.60 at 2000 sweeps to 1.14 in the long run (R-hat 2.89). Lambda and sigma2 are therefore weakly identified (they lie along a ridge), and their values on Duolingo should not be interpreted.

The predictions are insensitive to this. Test log loss is 0.2682 at 2000 sweeps and 0.2679 at 8000 sweeps, Brier 0.0715 and 0.0714, and the correlation between the two runs' per-event predictions is 0.992. The headline results (Sections 10.3 to 10.5) use the 2000-sweep run, which matches the Anki protocol. The long run is a sensitivity check.

### 10.7 Caveats

- **(a) FSRS-6 (optimised) is nearly FSRS-6 (default).** The per-user optimiser needs at least 512 informative train reviews (`MIN_INFORMATIVE_TRAIN` in `fsrs_benchmark/models/fsrs_model.py`). Only 112 of the 115,222 users reach it, so the default weights were kept for 115,110 users. The "optimised" row is almost the default model, and its difference from FSRS-6 (default) is small (Section 10.5).
- **(b) 64,254 users have zero train rows.** Their cards have no TRAIN history, so they are predicted from population-level parameters. This affects Ours (which uses the padding index, the cold path) and both FSRS variants through the default weights.
- **(c) Ours (GPU port).** The equations are the same as `AnkiFirstRun/dynamic_retention_dt.py`, ported to torch in `fsrs_benchmark/models/dynamic_retention_gpu.py` (the file is unchanged). The run used a Kaggle T4 GPU in float64 for the computation, with 2000 sweeps, 1000 burn-in and thin 10. The kept draws are stored as float32 to save host memory. Its lambda chain did not converge (Section 10.6).
- **(d) The label is a threshold.** `y` thresholds the recorded `p_recall` at 0.5. It is not an observed yes or no recall, so the metrics measure how well each model predicts that threshold.
- **(e) FSRS sees two grades.** The rating is 3 or 1 only, so FSRS does not use the Hard and Easy distinction that it uses on Anki.
- **(f) HLR has no lexeme features.** The official Duolingo model uses lexeme features that this comparison does not use, so this HLR is weaker than the published one (as on Anki, Section 8).
- **(g) No long gaps.** The data span is about 12 days, so the comparison covers only the 1d, 2-3d and 4-14d bins.
- **(h) One seed, one chain.** The run used seed 0 and one MCMC chain. The paired intervals reflect only user sampling.
- **(i) FSRS predict ran serially.** The run recorded `workers = 1`, so the parallel predict path (Section 5.1) was not used for the full run.

### 10.8 Run notes

The run was split over two Kaggle notebooks, one GPU and one CPU. The GPU notebook (`amalverma27/notebook2ef9000f5a`) produced `ours_gpu`, `hlr` and `laglogit`. The CPU notebook (`notebooke7d641abde`) produced `fsrs_default` and `fsrs_opt`. The two sets of prediction files were merged locally with `--resume`, which skips finished per-model files whose sidecars match the config. The merge wrote `predictions.parquet` (1,189,355 rows) and `run_info.json`. The git commit recorded in `run_info.json` is `ea51d42`, with `dirty: true`.

Total fit and predict times per model are in `results/duolingo_full/report.md`. The FSRS-6 predict times (about 2,500 s each, run serially) and the `ours_gpu` fit (about 2,600 s) are the largest per-model costs.

## 11. Improved models (Phases 6-7)

Phases 6 and 7 built three new models and a stacker, selected them on validation only, and ran one test look per model family. The experiment logs are `docs/reports/phase-6-experiments.md` and `docs/reports/phase-7-experiments.md`. The test runs are `results/full_u100_c1000_final/` (Anki, 100 users) and `results/duolingo_final/` (all Duolingo users). Both runs use the same split, scoring rule, metrics and runner as Sections 9 and 10. The models are in `fsrs_benchmark/models/ours_v2.py`, `ours_v3.py` and `gbdt.py`. The original `DynamicRetentionModelDT` and its GPU port are unchanged.

### 11.1 The models

- **v2 (`ours_v2`, Phase 6).** A hierarchical FSRS-style memory model. Its forgetting curve and stability updates use FSRS-6 functional forms, with 21 weights learned jointly across users. Five user-level random effects shift these weights for each user, and they are partially pooled towards the global weights. The model is fitted by MAP with torch, so it gives point estimates and not posterior samples. The frozen setting is 20 epochs and no process noise.
- **v3 (`ours_v3`, Phase 7).** v2 plus three changes, each ablated on validation. (1) An L2 penalty on the scales of the user effects (`tau_pen = 300`). This regularisation was the component that mattered, because v2's learned user-effect scale overfits on Duolingo. (2) A lagged-outcome term, so the previous review's outcome also enters the prediction. (3) A per-card logit effect with a fixed scale (0.2). The card effect is not separately supported: the same model without it is within 0.0001 log loss. The frozen setting is `epochs=20, tau_pen=300, lag=True, card_bias=True, card_tau=0.2`. It is still a MAP fit.
- **GBDT (`models/gbdt.py`, LightGBM 4.6.0, Phase 7).** A gradient-boosted tree model with no memory equation. Each review is described by features that use only the same card's and the same user's reviews before that review time: the gap since the card's previous review; the card's history (review count, lapses, and the last few grades, outcomes and gaps); the FSRS-6 default state (S, D, R) replayed over the card's history; running user statistics (recall and counts so far for the user); and a lexeme recall rate (Duolingo only) with smoothing (m = 3). Lexeme and user rates are computed from fit rows only. The frozen model is trained on all non-first rows (including same-day rows), with 15 leaves, learning rate 0.02 and smoothing m = 3, and it has no per-user prior group.

### 11.2 Combined test results

Test events are the same as in Sections 9.2 and 10.3. Reference is FSRS-6 (optimised). Each difference is model minus reference, with a 95% user-clustered paired bootstrap (1000 resamples, seed 0). "Users beating FSRS-opt" is the number of users (Anki, out of 100) or the share of users (Duolingo, out of 41,223) whose log loss is lower than FSRS-opt's. Anki v1 is "Ours (dynamic, learned lambda)" and Duolingo v1 is "Ours (GPU port)". Both are the same model family. LagLogit was not run on Anki.

| Model | Anki log loss | Anki diff vs FSRS-opt [95% CI] | Anki users better than FSRS-opt (of 100) | Duolingo log loss | Duolingo diff vs FSRS-opt [95% CI] | Duolingo share of users better than FSRS-opt (of 41,223) |
|---|---|---|---|---|---|---|
| FSRS-6 (default) | 0.396 | +0.017 [+0.008, +0.025] | 28 | 0.302 | +0.002 [+0.001, +0.002] | 0.000 (10 users; most tie, since FSRS-opt kept defaults) |
| FSRS-6 (optimised) | **0.379** (reference) | - | - | 0.300 (reference) | - | - |
| HLR (Settles & Meeder) | 0.680 | +0.301 [+0.246, +0.361] | 7 | 0.418 | +0.118 [+0.112, +0.124] | 0.577 |
| Ours v1 (MCMC) | 0.422 | +0.043 [+0.033, +0.055] | 22 | 0.268 | -0.032 [-0.033, -0.031] | 0.740 |
| Ours v1 (+kappa) | 0.422 | +0.043 [+0.033, +0.055] | 25 | not run | - | - |
| LagLogit + dt | not run | - | - | 0.291 | -0.009 [-0.012, -0.007] | 0.716 |
| Ours v2 (MAP, Phase 6) | 0.379 | -0.000 [-0.003, +0.003] | 56 | 0.289 | -0.011 [-0.012, -0.010] | 0.704 |
| Ours v3 (MAP, Phase 7) | 0.378 | -0.001 [-0.004, +0.002] | 58 | 0.283 | -0.017 [-0.018, -0.016] | 0.715 |
| **GBDT (LightGBM, Phase 7)** | **0.358** | **-0.021 [-0.028, -0.014]** | **82** | **0.259** | **-0.041 [-0.042, -0.041]** | **0.780** |

The best log loss on each dataset is in bold. Brier, AUC and ECE for every model are in `results/full_u100_c1000_final/report.md` and `results/duolingo_final/report.md`. GBDT has the best value of every metric on both datasets (log loss, Brier, AUC, ECE, RMSE and mean bias). The Duolingo v1 result (0.268, from Phase 5) is the second-best model there.

### 11.3 Selection protocol

- **Validation split.** Inside each user's train period, the last 20% of train days (same quantile rule as Section 4) is validation, and candidates are fitted on the first 80%. Validation events are scored with the same rule as the test events (not a card's first row, gap at least one day). Anki validation uses all 100 users (74,092 scored events). Duolingo validation uses a 5,000-user development sample (26,084 scored events). The 5,000-user sample is the development set (an orchestrator decision in Phase 6). The final Duolingo test uses all users.
- **No test labels in selection.** No test event is read while a candidate is chosen. The test set is looked at only for the final pick of each phase.
- **Fixed rules.** v3 and GBDT were chosen by a rule fixed before their runs: the lowest mean validation log loss over the two datasets, among variants whose validation CI does not show a loss against FSRS-opt on either dataset. v2 was chosen by a rule that was revised afterwards (disclosure (b)). Stacking was accepted only if it beats GBDT alone on both datasets with a CI that excludes zero. No stacker met this, so GBDT alone is the final model. The stacking candidates are listed in the Phase 7 experiments doc, Task 7.5.

### 11.4 Disclosures

- **(a) The test set was looked at twice.** v2 was tested in Phase 6. v3 and GBDT were tested together in Phase 7, after GBDT was frozen on validation. The GBDT choice was fixed before the test run, and the test run was used only to report it.
- **(b) The v2 selection rule was revised after seeing validation CIs.** The first rule chose v5, but v5 wins only on Duolingo. The rule was then changed to "prefer variants that win on both datasets, then lowest mean validation log loss", which chose v2. This change was made after reading the validation CIs, not the test set. v2's Duolingo validation log loss (0.277) is worse than v4/v5 (0.272), and the choice trades that margin for a significant Anki win.
- **(c) Number of variants tried.** v2: 9 (v1 to v9). v3: 23 (round 1: 12, round 2: 8, round 3: 3). GBDT: 38 variants per dataset in the first sweep, which had a leak (disclosure (d)), then 16 variants per dataset on the fixed code. Stacking: 6 (stack3, stack4, stack3x, stack4x, stack_gv, blend_gv). Every variant run is listed in the experiment docs.
- **(d) A same-session leak was found in the first GBDT and fixed.** Duolingo logs a whole practice session at one timestamp. The first user-statistics code let a row see the outcomes of other rows at that same timestamp, which inflated validation (frozen: 0.2473 to 0.2578 on the dev sample). The fix groups same-timestamp rows so that only strictly earlier reviews are used. Same-card duplicate rows at one timestamp are handled the same way. The invalid results are kept in `results/phase7_validation/invalid_leak/` and in the Phase 7 experiments doc. Anki reviews are individually timed and its file is in order, so Anki uses per-row updates. The Anki validation number changed after the fix (0.3742 to 0.3614) because the day grouping had also removed legitimate same-day history.
- **(e) The lexeme rate uses other users' train rows.** The lexeme rate pools fit rows from all users. Because users have different cutoff days, a fit row of one user can be later in calendar time than a scored row of another. No validation or test label is used, which is the same situation as any global parameter learned from all users' train rows under the per-user split. The lexeme feature changes log loss by about 0.001 on Duolingo, so the effect on the result is small.
- **(f) v2 and v3 are MAP fits.** Their uncertainty is not from full posterior sampling. Their intervals on the test set are the user-clustered bootstrap, not posterior intervals.
- **(g) The v1 MCMC model is still strong on Duolingo (0.268), but its lambda does not converge there.** At 2,000 sweeps lambda has split R-hat 2.48 and ESS 3. Section 10.6 gives the diagnostics. Its predictions are robust to this, but its lambda should not be interpreted.
- **(h) FSRS-opt falls back to defaults for most Duolingo users.** 115,110 of 115,222 users kept the default weights, so "FSRS-6 (optimised)" is almost the default model on Duolingo (Section 10.7(a)).
- **(i) GBDT inputs.** GBDT uses the same event-table inputs as the other models, plus the Duolingo lexeme id. It does not use `history_seen` or `history_correct`, because those count exposures before the log window, which no other model sees. The Anki `state` column is not used either. On Anki, the lexeme group is absent because there is no lexeme column.

### 11.5 What this means

GBDT is the best model on both datasets and on every metric, and the gap to FSRS-opt is significant on both (Anki -0.021, Duolingo -0.041 log loss). Most of that gain comes from running user performance (dropping the user group raises Anki log loss from 0.363 to 0.377 on validation) and from training on all non-first reviews, not from a memory model. The Bayesian models (v2 and v3) are competitive with FSRS-opt, and they are interpretable and make a per-user memory state explicit. They are not the best predictors here. On Anki, v2 and v3 tie FSRS-opt, and on Duolingo they beat it but trail GBDT and v1. The stacker found no gain over GBDT, so the final result is one model with no blend. Its only limitations are those in disclosures (a) to (i) and Section 11.6.

### 11.6 Known limits

- The GBDT is a pure predictor. It has no memory state (stability or difficulty) that a scheduler could use directly.
- Anki validation and the Duolingo dev sample are the only validation sets. No full-Duolingo validation run is recorded in the experiment docs.
- One seed and one split. The bootstrap intervals reflect user sampling only.
- The GBDT features are built for one-step-ahead prediction, not for multi-step forecasting.
- The GRU sequence model (task 7.4) and the v1 plus v2 ensemble (task 7.1) were not completed. See `docs/BACKLOG.md`.
