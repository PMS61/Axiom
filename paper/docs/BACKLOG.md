# Paper backlog (LOW findings)

## Phase 1
- report.py:435-437: no check that events.parquet matches the predictions (stale events file -> silent wrong baselines). (Being fixed in 1.2.)
- report.py:387,401: "bold best" can land on a constant (Anki ECE 0.004 for const_global). (Being fixed in 1.2.)
- const_user is unsmoothed: on Duolingo 9,351 users have train rate 1.0 and 92 have 0.0, so const_user (0.3725) is worse than const_global (0.2792). 298,282 of 1,189,355 scored Duolingo rows fall back to global. Disclose in FACTS.md and the paper.
- test_report_constants_verify.py:69-71: leftover comment and redundant `events =` assignment.
- paper_export.py update_facts: a half marker (start marker without an end marker) appends a duplicate block; raise instead.
- paper_export.py: unused PAIRED_OUT/OVERALL/GAP constants; "Scored events" n is read from the first model's row.
- report.py check_events_match checks only scored rows; also check train count or cutoff against run_info.
- test_report_constants.py build_report tests take ~35 s each; mark them slow or cache them.
- Duolingo run_info.json "users_zero_train" (64254) is not defined anywhere; document it.
- metrics.py _paired_bootstrap_reference: two docstrings in a row; merge them.
- paper_export.py: smoothed-wins FACTS lines don't name their reference in the line itself.
- srs remove_outliers picks scored rows from full-user counts (shared with FSRS-6, so fair); mention in the paper.
- srsb jsonl omits smECE, precision@90 and recall@90.
- srsb TrainPool.parts holds all train X until finalize() (about 2x peak memory).
- FACTS 'frozen settings' claim for seed 0 hlr/fsrs cannot be checked from run_info (it only says resumed).
- Add the FSRS-opt fallback counts for seeds 1 and 2 (4 and 6 users) to FACTS.
- Figure captions still say 'Caption TODO' (fill in 3.4); the reliability_duolingo legend touches markers; the seeds table sits alone on page 3.
- paired.tex shows v3 Duolingo user win as 72% (0.7150, which rounds half-up); decide on one rounding rule.
- R-hat and ESS are undefined in models.tex:47 and discussion.tex:20.
