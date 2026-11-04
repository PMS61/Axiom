# Phase 7 report: the best model (ours_v3, GBDT, stacking)

Details: `docs/reports/phase-7-experiments.md` (every variant, CIs, rules, leak record) and `docs/EVALUATION.md` Section 11.

## What was built

- **ours_v3** (`fsrs_benchmark/models/ours_v3.py`, key `ours_v3`). A copy of v2 with extra flags: an L2 penalty on the user-effect scales (`tau_pen`), a lagged-outcome term, and a fixed-scale per-card effect. v2 is unchanged. The loader gained an optional `lexeme` column (Duolingo only). Anki has none.
- **GBDT** (`fsrs_benchmark/models/gbdt.py`, key `gbdt`, LightGBM 4.6.0). Past-only features per review: gap, card history, the FSRS-6 default state replayed over the card, running user statistics, and a smoothed fit-rows lexeme rate. Trained on all non-first rows. The frozen setting is `nf_l15_lr02_sm3_nup`.
- **Stacking and blending** (task 7.5). Cross-fitted by user on validation predictions.
- **Leak fix.** Same-timestamp rows (a Duolingo session) and same-card duplicate rows at one timestamp are grouped, so only strictly earlier reviews feed the running statistics. Anki uses per-row updates because its file is in order.
- **Registration.** `ours_v3` and `gbdt` are registered in `run.py` and `report.py`.

## How it was tested

- `test_ours_v3.py` (8 tests) and `test_ours_v3_verify.py` (9 tests): the test labels never change fitted parameters or any other prediction, the validation table has no test event, and the frozen settings match the documented choice.
- `test_gbdt.py` (12 tests) and `test_gbdt_verify.py` (14 tests): base features are past-only, same-timestamp and same-card duplicate rows cannot see each other, aggregates use fit rows only, the fitted booster ignores validation and test labels, and the state and history columns are not inputs.
- These tests were not re-run in this documentation pass. The leak was found after the first GBDT run. The fixed tests are the record of the fix.

## What was found

Validation (dev sample; log loss, lower is better):

- ours_v3: 23 variants. The frozen choice is q2 (`epochs=20, tau_pen=300, lag, card_tau=0.2`). Anki 0.379, Duolingo 0.271.
- GBDT: 38 variants in the first sweep. Those numbers were invalid because of the leak. The fixed code was re-run with 16 variants per dataset. The frozen choice is unchanged. Anki 0.361, Duolingo 0.258.
- Stacking: 6 candidates. None beat GBDT alone on both datasets, so GBDT alone is the final model.

Test (one look for v3 and GBDT; GBDT was frozen before the look):

| Dataset | GBDT | ours_v3 | ours_v1 (for comparison) | FSRS-6 (optimised) |
|---|---|---|---|---|
| Anki (100 users) | 0.358, -0.021 [-0.028, -0.014], 82 of 100 users win | 0.378, tie | 0.422 | 0.379 |
| Duolingo (all users) | 0.259, -0.041 [-0.042, -0.041], 0.780 of users win | 0.283, -0.017 [-0.018, -0.016] | 0.268 | 0.300 |

GBDT is the best model on both datasets and on every metric. The gain comes mainly from running user performance and from training on all non-first reviews. v3 ties FSRS-opt on Anki and beats it on Duolingo. The Bayesian model is competitive but not the best predictor.

## Known limits

- **GBDT is a pure predictor.** It has no memory state (stability or difficulty), so it cannot be used directly to schedule reviews.
- **Lexeme pooling pools other users' training rows.** These rows can be later in calendar time than a scored row. The effect is about 0.001 log loss on Duolingo.
- **v1 (MCMC) is still second on Duolingo (0.268).** Its lambda does not converge there (Section 10.6). The 7.1 ensemble of v1 and v2 was not completed, because v1 validation fits were too slow locally (3 to 9 hours).
- **Task 7.4 (GRU sequence model) was not tried.**
- **Validation is on the dev sample.** The 5,000-user Duolingo sample was used for selection. No full-Duolingo validation run is recorded.
- **Test was looked at twice** for this model family (v2 in Phase 6, then v3 and GBDT in Phase 7). Disclosed in `docs/EVALUATION.md` Section 11.4.
- **Constant-recall baselines are still pending** (see `docs/BACKLOG.md`).
- **MAP fits.** v3 is a MAP fit, not full posterior sampling.
