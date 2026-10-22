# Phase 2 report: models (DONE)

## Built (all share fit(events) on is_train rows, predict(events) -> Series[event_id] for scored rows)
- `fsrs_benchmark/models/fsrs_model.py` — `FSRSModel(optimize)`: py-fsrs 6.3.2 Scheduler (fuzz off); each card replayed in review_idx order with synthetic UTC timestamps (epoch + day_offset days + tiny per-review seconds), p = `get_card_retrievability` BEFORE `review_card`. `optimize=True` runs py-fsrs `Optimizer` per user on TRAIN logs only; falls back to defaults when < 512 informative (non-same-day) train reviews (py-fsrs silently does this itself) — fallback users are recorded.
- `fsrs_benchmark/models/hlr.py` — `HLRModel`: faithful port of the official duolingo/halflife-regression `experiment.py` HLR (SGD from zero weights, hclip [15 min, 274 d], pclip [1e-4, 0.9999], h_obs term hlwt=0.01, l2wt=0.1, per-feature rate). Numba epoch kernel. Documented deviations: features only [sqrt(1+right), sqrt(1+wrong), bias] (no per-lexeme), multiple shuffled epochs until max|Δθ|<1e-4. Trained on TRAIN, non-first, elapsed_days>=1 rows.
- `fsrs_benchmark/models/ours.py` — `OursModel(use_kappa, iters, burn, thin, seed, max_draws)`: wraps unmodified `AnkiFirstRun/dynamic_retention_dt.py`. One sequence per (user, card) from TRAIN rows; ids from TRAIN rows only; test-only cards/users use the model's pad (cold path). Prediction via the model's own one-step filtered `predict()` (p_t emitted before outcome t is used).

## Event definition change (important)
`state == 4` rows were originally dropped; they are real reviews (ratings 1-4, real durations; filtered-deck reviews). Dropping them made FSRS see wrong gaps (e.g. 515 d instead of 1 d). Now kept; only rating 0 is dropped; `elapsed_days` / `dt_days` recomputed over kept rows (`raw_elapsed_days` keeps the file value). PLAN.md updated.

## Tested
174 tests pass (~62 s). Highlights: FSRS replay matches an independent py-fsrs reference to 1e-12 on all real scored events; HLR SGD step matches official arithmetic to 1e-12; no-leakage tests for every model (own/later outcomes never change p_k; fit ignores test rows); all 5 models predict an identical event_id set.

## Smoke numbers (users 6370+8506, 300 cards, 626 scored; NOT results)
| Model | log loss | Brier | AUC | ECE |
|---|---|---|---|---|
| FSRS-6 default | 0.460 | 0.134 | 0.619 | 0.050 |
| FSRS-6 optimised | 0.449 | 0.133 | 0.615 | 0.049 |
| HLR | 0.575 | 0.155 | 0.583 | 0.126 |
| Ours (200 it) | 0.440 | 0.134 | 0.564 | 0.042 |
| Ours +kappa (200 it) | 0.443 | 0.134 | 0.549 | 0.040 |

## For Phase 3
- FSRS optimiser needs >= 512 informative train reviews per user; with max_cards=300 many users fall back to defaults. Pick max_cards so most users are actually optimised, and report the fallback count.
- Ours timing: ~10 s for 400 iters on 600 cards; scale accordingly (Gibbs cost ~ N_cards × max seq length).
- Shared real-data test cache: `fsrs_benchmark/tests/_real.py`.
