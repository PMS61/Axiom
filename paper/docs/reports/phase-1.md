# Paper Phase 1 report: closing the evidence gaps (2026-11-15)

## What was built
- **Constant floors in `fsrs_benchmark/report.py`** (1.1, 1.2). There are three: `const_global`, `const_user` and `const_user_smooth`. The smoothed version shrinks each user's rate toward the global rate with a pseudo-count m = 10, fixed in advance. The rates come from TRAIN rows that are not first reviews and have a gap of at least 1 day. A check confirms that events.parquet matches the predictions. Bold marks in the tables skip the constants.
- **Paired references** (1.2c): `fsrs_opt`, `ours`, `ours_gpu` and `const_user_smooth`.
- **Fast weighted paired bootstrap** (1.2b) in `metrics.py`. It uses the same RNG draws as the old code and agrees with it to 1e-12. The Duolingo report now builds in 18 min instead of about 5 h (68 min once the extra references are included).
- **Paper result sets**: `results/paper_anki/` and `results/paper_duolingo/`, which are symlinks plus a rebuilt Duolingo predictions.parquet.
- **`fsrs_benchmark/paper_export.py`** writes `paper/data/*.csv` and `paper/FACTS.md`, the single source of numbers. It has three sections: anki, duolingo and anki_seeds. Each line cites its source file.
- **Anki robustness** (1.3): seeds 1 and 2 were run on Kaggle (notebook `notebook8f676349c8`, private dataset `axiom-paper-code`, `HF_TOKEN` secret). Each seed is 100 users, and the samples overlap seed 0 by only 1 user each.

## Key results (sources in FACTS.md)
- Duolingo: the smoothed per-user constant scores 0.2723. It beats FSRS-opt (0.3003), v3 (0.2830), v2, LagLogit and HLR. Only GBDT (−0.0135 vs that floor) and v1 (−0.0040 [−0.0045, −0.0035]) beat it.
- Anki: every model except HLR beats the floors. v1 beats the smoothed constant by only −0.0086.
- GBDT minus FSRS-opt on Anki is −0.0210, −0.0226 and −0.0210 for seeds 0, 1 and 2. Every CI excludes 0.

## Tests
- Every test file was run one per process, 43 files in all. The only failure is the known `test_task52_54_verify.py::test_resume_config_records_thin_for_gpu_draws`, plus one expected xfail.
- The verifier checked every FACTS line against the reports by script, with no mismatches.

## Known limits / for later phases
- See `paper/docs/BACKLOG.md`. The ones that matter for the paper:
  - the unsmoothed per-user constant is poor on Duolingo (9,351 users have a train rate of exactly 1.0);
  - FSRS-opt fell back to defaults for 4 users in seed 1 and 6 in seed 2 (not yet in FACTS);
  - seed 0's FSRS and HLR settings are not recorded in run_info (that run was resumed).
- The Duolingo report now takes about 68 min, so rebuild it only when necessary.
