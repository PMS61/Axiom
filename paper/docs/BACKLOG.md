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
