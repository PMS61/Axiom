# GBDT-Curve: spec (Phase 8, branch exp/srsb-beat)

## Intent
Our final model (GBDT + online layer) is an **Instant** model: it predicts just before a review, using
everything known at that moment. srs-benchmark also has a **Curve** category (RWKV-Curve,
`external/srs-benchmark/result/RWKV.jsonl`). A Curve model must fix the card's state at the card's
**previous review** j and output recall as a function of the elapsed time t only; recall must not rise
as t grows. Goal: a fair GBDT-Curve (+ curve online layer) that beats RWKV-Curve on validation users
8001-8100 (train users 2001-2300; never users 1-1000 until pre-registered).

## Definitions
For a scored row i (stream index `review_th`), j = the previous stream row of the same card
(`st` = `R.review_stream(raw, cfg.max_seq_len*2)`, group by `card_id`). Every scored row has a j
(first reviews are never scored). The full-stream feature frame is
`feats = concat([G.base_features(st, span=20, user_group="row")[R.FEATURE_COLS], plus_features(st, cards, decks)])`
(see `val_plus._full_table`), indexed by stream index; row j's values are "as of just before review j".

## Curve features (one row per scored row i)
Each ALL_COLS column goes in exactly one class. The builder must read the feature code
(`fsrs_benchmark/gbdt.py` base_features, `fsrs_benchmark/srsb/run_gbdt.py`, `fsrs_benchmark/srsb/features_plus.py`)
and classify every column; when unsure, choose the conservative class (FROM_J or DROP).

1. **TIME** (functions of the card's state after j and of t; value from row i): `el, log_el, el_frac,
   log_el_frac, f_R, f_logitR, f_el_over_S, f_R_secs`. Monotone constraints: -1 for el, log_el,
   el_frac, log_el_frac, f_el_over_S; +1 for f_R, f_logitR, f_R_secs.
2. **CARD_I** (depend ONLY on this card's own reviews up to and including j, not on t and not on other
   cards; value from row i): e.g. card_n, card_lapses, card_rate, g1-g3, y1-y3, gap1-gap3, f_S, f_D,
   card_max_gap, card_ewm3, card_mean_loggap, card_r4, card_r5, card_gap4, card_gap5, dur_card_last,
   dur_card_mean, state (if it is the card's state, not the user's). Verify each one in code.
3. **FROM_J** (user, today, deck, sibling, EWMA, residual, bucket and any other column that can change
   because of OTHER cards' reviews; value taken from row j): e.g. u_*, ewm*, resid_*, n_today,
   n_first_today, rate_today, lapses_today, u_last5, dur_user_mean, dur_user_prev, dur_card_last_rel,
   rc1-rc4 (if per user), sib_*, deck_*, preset_*, parent_*, cnt_bkt_*, g1_rate, g1_n, state_rate, state_n.
4. **DROP**: `same_day` (always 0 on scored rows), `rows_since_card` (counts activity after j),
   `gap_bkt_rate`, `gap_bkt_n` (mix t with user stats after j), `since_first` replaced by
   `since_first_j = since_first(i) - el(i)` in the same units (card age at j; check units in code).

Cache: new dir `data/anki10k/feat_cache_curve/<hash>/u{uid}.npz` (X, y, test, plus `th` = review_th of i
and `prev_th` = stream index of j), same META as val_plus. Build from the same feats frame, no other
data source.

## Curve GBDT
LightGBM, the frozen settings of `sweep_a_l63_side` (objective binary, num_leaves 63, lr 0.05,
min_data_in_leaf 200, feature_fraction 0.8, bagging_fraction 0.8, bagging_freq 1, lambda_l2 1.0,
early stopping 50 on the fixed holdout users from `R.holdout_users`, max 10000 rounds,
max_train_rows 12,000,000 with the same per-user quota logic as val_plus) **plus**
`monotone_constraints` from the TIME list (0 for all other columns). Train users 2001-2300.

## Curve online layer
Per user, over that user's scored rows in stream order, run `online_adapt.online_kernel` logic with a
design WITHOUT time terms: x = [z, 1, (ewm50_j - 0.5)*4, (rate_today_j - 0.5)*4] (ewm50 and rate_today
from row j; NaN -> 0.5), w0 = [1,0,0,0], lr 0.003, lambda 1e-3, and **w[0] clipped to >= 0** after
every update (so recall stays monotone in t). Store the weight vector after each update. The prediction
for row i uses the weights after the last update at a scored row whose stream index is <= prev_th(i)
(w0 if none). This is the curve emitted at review j.

## Outputs and evaluation
`fsrs_benchmark/srsb/curve.py` with CLI: `--data`, `--train_users`, `--eval_users` (files or ranges as
other srsb scripts accept), `--out` (jsonl in srs-benchmark format like val_plus output, one for
"curve" and one for "curve+online"), `--workers`, `--cache_dir`, `--quick` (3 train / 2 eval users).
Refuse users 1-1000 unless `--final`. Print weighted and unweighted log loss and AUC on test folds for:
curve, curve+online, RWKV-Curve (RWKV.jsonl), RWKV-Instant (RWKV-P.jsonl), and paired size-weighted
99% bootstrap CIs (1000 resamples, seed 0) of curve+online minus RWKV-Curve and curve minus RWKV-Curve.
Reuse `compare` / `val_plus.compare_published` where possible.

## Required tests (fsrs_benchmark/tests/test_curve.py)
- Leak test: for a small synthetic or real user, changing any review strictly after j (other cards,
  same day or later) must not change curve row i's FROM_J/CARD_I values or its online prediction.
- Monotone test: for the trained quick model, predicted p for a row decreases (non-increasing) when
  the TIME columns are recomputed for larger t.
- Column classification covers all 78 ALL_COLS exactly once.
