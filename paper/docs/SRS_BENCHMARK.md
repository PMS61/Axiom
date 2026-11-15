# srs-benchmark scoping (Phase 2.1, orchestrator, 2026-11-15)

Clone: `external/srs-benchmark` (git-ignored; shallow clone of open-spaced-repetition/srs-benchmark, commit dated 2026-10-25).

## Its protocol, and how it differs from ours
- **Data:** the same anki-revlogs-10k (9,999 users, about 727M reviews). Per user, `features.create_features` builds review rows
  with card history (`t_history`, `r_history`), `elapsed_days`, `review_th` (the global review index inside the user), `card_id` and so on.
- **Split:** per user, sklearn `TimeSeriesSplit(n_splits=5)` over the user's rows in time order. The first chunk is
  train-only. Each later chunk is a test fold, and its model is trained on everything before it. This differs from our
  single 80th-percentile day cutoff.
- **Scored rows:** in the "without same-day reviews" table, same-day reviews are not scored but can feed history
  (the `--short` flag). A per-user outlier filter is applied inside `create_features`.
- **Metrics:** per-user LogLoss, RMSE(bins) (their bin definition), AUC, ICI and MBE. The headline is the **mean over users
  weighted by review count** (`size`), with 99% CIs from a weighted bootstrap (`evaluate.py`). Ours pools events instead.
- **Training scope:** most models are fitted per user. RWKV is the exception. It is trained on 5k users, evaluated on the other
  5k, and then the roles swap. A cross-user GBDT fits this RWKV-style protocol.
- **Published per-user results** are in `result/*.jsonl`, one line per user: `{"metrics": {...}, "user": id, "size": n}`.
  So a new model can be compared **paired, user by user**, on exactly the same users, if it reproduces their test rows
  (check `size` per user).

## Published numbers (README, without same-day reviews, weighted mean ± 99% CI)
| Model | LogLoss | RMSE(bins) | AUC |
|---|---|---|---|
| RWKV-Instant | 0.2773 | 0.0250 | 0.8329 |
| RWKV-Curve | 0.3193 | 0.0540 | 0.7683 |
| GRU | 0.3328 | 0.0549 | 0.7324 |
| LSTM | 0.3332 | 0.0538 | 0.7329 |
| MOVING-AVG | 0.3369 | 0.0592 | 0.7001 |
| FSRS-7 recency | 0.3370 | 0.0593 | 0.7220 |
| Logistic Regression | 0.3393 | 0.0604 | 0.7108 |
| FSRS-6 | 0.3460 | 0.0653 | 0.7034 |
| DASH | 0.3682 | 0.0838 | 0.6311 |
| AVG (per-user constant) | 0.3945 | 0.1034 | 0.4997 |
| HLR | 0.4694 | 0.1275 | 0.6369 |

Note that MOVING-AVG, which uses no memory model, beats FSRS-6 here. This supports our "constant floor" framing.

## How a model would be added
- The official way is to add it to `models/` and `model_processors.py`, and run `script.py --algo X [--short]`. Per-user
  processing happens in `script.process()`.
- Our way: a separate driver, `fsrs_benchmark/srsb/run_gbdt.py`. It imports their `Config`, `UserDataLoader` and metric code,
  reproduces their TimeSeriesSplit test folds per user, and writes `result/GBDT-*.jsonl` lines in their format. Their
  code is not modified.

## GBDT protocol inside srs-benchmark (decided)
- **Pooled, RWKV-style.** Train one LightGBM model on a pool of training users (set A) with past-only features. Evaluate it
  on a disjoint set of users (set B), on their TimeSeriesSplit test folds (folds 1-4, no same-day rows scored).
- Features per row use only that user's earlier rows (by `review_th`, with same-timestamp grouping as in our leak fix):
  - the gap;
  - the card's review count, lapses, and the last 3 grades and gaps;
  - the FSRS-6 default state (S, D, R) replayed over the card history;
  - running user statistics (the recall rate so far and the count so far).
- The settings are our frozen GBDT settings (15 leaves, learning rate 0.02) with early stopping on a set-A holdout. There
  is no tuning on set B.
- **Fairness note for the paper:** per-user models such as FSRS-6 are trained on the user's own earlier data. Pooled GBDT
  is trained on other users and sees the target user only through online history features, like RWKV. Both see
  only past data at prediction time.

## Compute
- The average user has about 73k raw reviews. Load and featurize one user at a time, and keep features as float32.
- **Honest subset:** set B is the first 1,000 users in `user_order.jsonl` order, or user ids 1-1000, whichever matches
  the published jsonl best. Set A is a disjoint 500 users. Training rows are subsampled to at most 20M if needed.
- **Where:** Kaggle CPU (30 GB RAM, 4 cores, 12 h). The data is downloaded inside the notebook with the `HF_TOKEN` secret
  (already added to notebook `amalverma27/notebook8f676349c8`; secrets attach per notebook). Local runs only do a
  smoke test (10 users in A, 10 in B).
- **Comparison:** compute the same weighted means on set B for GBDT and for the published jsonl of FSRS-6,
  FSRS-7 recency, GRU, LSTM, RWKV (cross-user), MOVING-AVG, AVG, LogReg and HLR. Add a paired per-user bootstrap of the GBDT
  difference vs FSRS-6 and vs RWKV. Report how many users in set B match on `size`.
