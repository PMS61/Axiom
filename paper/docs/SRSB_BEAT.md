# Phase 6: beat RWKV-Instant on srs-benchmark (branch exp/srsb-beat, started 2026-11-17)

## Goal
Our pooled GBDT scores 0.3211 log loss (unweighted, eval users 1-1000, `--algo FSRS-6 --short` test folds).
RWKV-Instant scores 0.2860 on the same users (results/srsb_1000/compare.txt). Target: beat RWKV-Instant on users
1-1000, with the paired size-weighted 99% CI of GBDT+ minus RWKV-Instant below 0.

## Honesty rules (non-negotiable)
- **Users 1-1000 are the TEST set.** No feature, setting or model choice may look at them. They are scored ONCE, at
  the end (task 6.5), with the settings frozen on validation.
- **Validation users:** 8001-8300. Model selection uses only these. They are compared against the published
  RWKV-Instant and FSRS-6 per-user results for the same users (fixed published files, so this is fair).
- **Train pool:** users 2001-8000 and 9001-9999 (never 1-1000, never 8001-8300).
- **Past-only:** a feature of review t may use only reviews strictly before t in the user's stream (review_th
  order), plus information shown before the answer: the current gap (elapsed_seconds), the card's note, deck and
  preset ids, and the current row's `state` (Anki review type, set before answering). NEVER the current row's
  `rating`, `duration`, or anything after it.
- Aggregates over many rows (deck/preset rates) are running, past-only, within the user.

## New feature groups (on top of gap, card, fsrs, user)
- `secs`: el_frac = elapsed_seconds/86400 (fractional gap, -1 for a first review); log1p(el_frac); f_R_secs (FSRS-6
  default R evaluated at el_frac instead of whole days).
- `today`: within the same user and day_offset, earlier rows only: n_today, n_first_today (first reviews today),
  rate_today (recall rate over earlier non-first rows today, NaN if none), lapses_today (rating==1 earlier today);
  u_last5 (mean y of the user's last 5 reviews of any card); rows_since_card (user rows since this card's last
  review).
- `dur`: log(1+duration) of the card's last review, the card's running mean log duration, the user's running mean
  log duration, the card's last over the user's mean (difference of logs), the log duration of the user's previous
  review (any card). All from PAST rows only.
- `rcount`: the card's past counts of ratings 1, 2, 3 and 4; the card's longest past gap (days).
- `state`: the current row's state (categorical 0-4; 4 = filtered deck).
- `sib` (needs cards.parquet note_id): over earlier reviews of OTHER cards of the same note: count, last y,
  days since the last sibling review (fractional when possible, else day_offset difference), running rate.
- `deck` (needs cards.parquet deck_id and decks.parquet preset_id/parent_id): past-only running recall rate and
  count per deck, per preset, per parent deck, within the user (non-first rows only for the rates).

The label y: rating > 1. A first review of a card has no features from that card (NaN).

## Data
HF dataset open-spaced-repetition/anki-revlogs-10k has revlogs/, cards/ and decks/ partitioned by user_id. The
token is in .secrets/hf_token (gitignored, never print it). Local users: data/anki10k/revlogs (111 users, revlogs
only).

## Tasks
- 6.1 Features: `fsrs_benchmark/srsb/features_plus.py` + cards/decks download + tests.
- 6.2 Validation harness: `fsrs_benchmark/srsb/val_plus.py` (train pool -> score val users -> per-user jsonl ->
  paired comparison vs RWKV-Instant, FSRS-6 and the old GBDT on the val users). Local smoke test on a few users.
- 6.3 Kaggle validation runs: feature-group ablation and scale (number of train users, round cap, leaves).
- 6.4 If GBDT+ still trails RWKV-Instant on validation: a sequence model (GRU/transformer over the user stream with
  the same features), same harness.
- 6.5 Final: one run on users 1-1000 with the frozen choice; compare.py; FACTS.md; paper text.

## Decisions
- 2026-11-17 (user): cards/decks for eval users 1-1000 may be downloaded. They are downloaded only as part of the
  final 6.5 run (inside the Kaggle notebook), never used for selection.
- 2026-11-18 (user): only SMALL runs (about 150-300 train users, 100 validation users 8001-8100) until some
  model beats RWKV-Instant on validation; scale up only after that.

## Status 2026-11-18 (gap = ours minus RWKV-Instant, size-weighted log loss, about 100 val users)
- GBDT+ new features, 76 train users: +0.029. 300 users: +0.022. +umem features: +0.018. 63 leaves: +0.017.
- Running: TAM on Colab (150 train / 100 val); GBDT+umem 1500 train users on Kaggle notebook52e4915672 v3 (CPU);
  task 6.5a online per-user adaptation layer (local, small).
- Runbook lessons are in memory note colab-kaggle-runbook (Drive uploads, pinned packages, local dry run first).
