# Paper Phase 2 report: srs-benchmark comparison (2026-11-17)

## What was built
- `external/srs-benchmark`: a clone at commit 4ddb9fd, git-ignored and unmodified.
- `paper/docs/SRS_BENCHMARK.md`: the scoping notes and the chosen protocol.
- `fsrs_benchmark/srsb/run_gbdt.py`: the pooled GBDT, trained RWKV-style on users disjoint from the evaluation users.
  - It reuses the benchmark's own loader, features, TimeSeriesSplit(5) test folds and metrics.
  - It resumes after interruption, streams users, and runs featurization in parallel.
- `fsrs_benchmark/srsb/compare.py`: reproduces the README numbers as unweighted per-user means and runs a paired, size-weighted user bootstrap (99% CI).
- `paper_export --srsb` writes `paper/data/srsb*.csv` and the facts:srsb block of FACTS.md.

## Run (Kaggle notebook8f676349c8 Version 2, dataset axiom-paper-code-srsb)
- Training used users 9001-9500: 450 to fit and 50 held out, 17,060,060 rows. best_iter was 2000, which is the round cap, so early stopping never triggered and the model is probably under-trained, making these results a lower bound.
- Evaluation users were 1-1000, with no overlap with the training users. All 1000/1000 users match the published FSRS-6-short `size`.

## Results (log loss, unweighted / size-weighted, the 1000 users common to all models)
| Model | Unweighted | Weighted |
|---|---|---|
| GBDT | 0.3211 | 0.3148 |
| FSRS-6 | 0.3580 | 0.3383 |
| RWKV-Instant | 0.2860 | 0.2828 |
| RWKV-Curve | 0.3302 | 0.3143 |
| GRU | 0.3441 | 0.3271 |
| MOVING-AVG | 0.3462 | 0.3372 |
| AVG | 0.4055 | 0.3802 |

Size-weighted paired differences with 99% CIs:
- GBDT minus FSRS-6: −0.0235 [−0.0286, −0.0183].
- GBDT minus GRU: −0.0123 [−0.0174, −0.0067].
- GBDT minus RWKV-Instant: +0.0320 [+0.0287, +0.0358].
- GBDT minus RWKV-Curve: +0.0006, with a CI that includes 0.

## Tests
- test_srsb.py: 15 passed. test_srsb_verify.py: 7 passed. test_paper_export.py: 25 passed.
- The verifier recomputed every number independently from the raw jsonl.

## Known limits
- One training pool and one subset of 1000 users.
- best_iter hit the cap.
- RWKV-Instant uses extra features that GBDT does not have.
- The jsonl has no smECE.
