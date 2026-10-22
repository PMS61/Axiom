# Phase 1 report: data and protocol (DONE)

## Built
- `fsrs_benchmark/data.py`: `download_users(n_users, seed, out_dir)` (HF gated dataset, deterministic user sample, skips existing files, clear error on missing token/licence) and `load_user(path)` (event definition from PLAN: drops rating 0 / state 4, drops cards whose first kept review is not state 0, `y = rating>1`, `dt_days = elapsed_seconds/86400` falling back to elapsed_days, NaN on first reviews, `is_first`, `review_idx`, `user_id` from the folder name).
- `fsrs_benchmark/split.py`: `make_split(df, max_cards, seed)` caps cards per user (rng(seed) over sorted card_ids), THEN computes the per-user `cutoff_day` = 80th percentile of day_offset; `is_train = day_offset < cutoff_day`; `scored = test & ~is_first & elapsed_days >= 1`; `event_id = user_id*1e9 + review_idx` (int64). `save_events/load_events` (parquet).
- `fsrs_benchmark/metrics.py`: log_loss, brier, auc, ece, mean_bias, reliability_curve, rmse_bins, summarize, paired_bootstrap (user-clustered, 95% percentile CI). Inputs are validated (NaN/out-of-range p, non-binary y -> ValueError).

## Tested
108 tests pass (`.venv/bin/python -m pytest fsrs_benchmark/tests -q`), including cross-checks against brute-force AUC and torch BCE, and real-data invariants on users 6370 and 8506.
End-to-end on 2 users (300 cards each): 4,214 events, 3,363 train, 851 test, 612 scored; scored recall 0.845.

## Bugs found and fixed
- metrics accepted NaN / out-of-range inputs silently (auc returned 0.5 on NaN). Now raises ValueError.

## Known limits / for next phase
- Real data downloaded so far: users 6370, 8506 in `data/anki10k/revlogs/`.
- Per-card event order inside a user is `review_idx`; models must replay per card sorted by review_idx.
- `scored` events need predictions from every model; unscored test events (same-day, first reviews) are history only.
- Users with all reviews on one day get no train events (runner should warn).
- Bootstrap CIs need many users to be meaningful.
- LOW items: `docs/BACKLOG.md`.
