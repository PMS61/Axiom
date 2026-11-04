# Phase 6 report: a hierarchical Bayesian FSRS-style model (ours_v2)

Details: `docs/reports/phase-6-experiments.md` (every variant, CIs, rules) and `docs/EVALUATION.md` Section 11.

## What was built

- **ours_v2** (`fsrs_benchmark/models/ours_v2.py`, key `ours_v2`). A new model, separate from `DynamicRetentionModelDT` and its GPU port, which are unchanged. It uses FSRS-6 functional forms, with 21 weights learned jointly across users and 5 hierarchical user effects. It is fitted by MAP in torch.
- **Validation split** (`make_validation_split`). The last 20% of each user's train days is validation. Candidates are fitted on the first 80%. Test rows are dropped before fitting.
- **Runner and report.** `ours_v2` is registered in `run.py` and `report.py`. The validation runner writes to `results/phase6_validation/`.

## How it was tested

- `fsrs_benchmark/tests/test_ours_v2.py` (8 tests): the validation split has no leak, fit ignores validation and test rows, predictions are complete and valid, unseen users are predicted, and `ours_v2` runs end to end through `run` and `report`.
- `fsrs_benchmark/tests/test_ours_v2_verify.py` (11 tests): the fit is invariant to shuffled or flipped non-fit outcomes, predictions are independent of every outcome, of own and future outcomes and of gaps, the validation split matches an independent recomputation, the FSRS optimiser sees only fit rows, and a test-label perturbation leaves all earlier predictions unchanged.
- These tests were not re-run in this documentation pass. The results below come from the recorded runs.

## What was found

Validation (Anki 100 users; Duolingo 5,000-user dev sample; log loss, lower is better; FSRS-opt is 0.385 on Anki and 0.289 on Duolingo):

- Nine variants (v1 to v9). The first rule chose v5. v5 wins on Duolingo but only ties on Anki.
- The rule was revised after seeing validation CIs (post hoc, disclosed). The revised rule chose v2 (20 epochs, 5 user effects, no process noise). It wins on both datasets (Anki -0.0049 [-0.0090, -0.0012], Duolingo -0.0123 [-0.0186, -0.0054]).

Test (one look, all events; the first look at test in the project for this model family):

| Dataset | ours_v2 log loss | FSRS-6 (optimised) | Diff [95% CI] | Users beating FSRS-opt |
|---|---|---|---|---|
| Anki (100 users) | 0.379 | 0.379 | -0.000 [-0.003, +0.003] (tie) | 56 / 100 |
| Duolingo (all users) | 0.289 | 0.300 | -0.011 [-0.012, -0.010] | 0.704 of 41,223 |

So v2 is at parity with FSRS-opt on Anki and beats it on Duolingo. On Duolingo it is worse than v1 (0.268).

## Known limits

- **MAP, not posterior sampling.** The uncertainty is not from full posterior samples.
- **The rule was revised after validation CIs were read.** This is disclosed in `docs/EVALUATION.md` Section 11.4(b).
- **Anki is a tie.** No Anki win may be claimed for v2.
- **Duolingo lexeme pooling was not used.** Phase 6 has no lexeme term. The idea was tested in Phase 7.
- **Naming.** "v2" in the Phase 6 log is a variant number of the first candidate. The model key `ours_v2` is the frozen variant.
- **FSRS-opt is nearly the default model on Duolingo** (115,110 of 115,222 users fell back to defaults), so the Duolingo baseline is weak.
- **Single seed and single split.** The bootstrap reflects user sampling only.
