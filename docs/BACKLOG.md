# Backlog (LOW findings)

## Phase 1
- data.py:113,125 — NaN rating survives filter and is labelled y=0; validate rating ∈ {0..4}.
- data.py:60,78 — download_users only wraps HfHubHTTPError; network errors propagate unwrapped.
- data/anki10k/revlogs/user_id=1/ stray download (git-ignored).
- data.py:102 — relies on file order being chronological; no enforcement/check.
- split.py:25 — DEFAULT_EVENTS_PATH relative to cwd; resolve from repo root.
- split.py:53-55 — unreachable `if user.empty: continue`.
- Users with all reviews on one day get zero train events; runner should warn/skip.
- Cluster bootstrap CIs degenerate with very few users (2); document that CIs need many users.

## Phase 2
- fsrs_model.py fit(): MIN_INFORMATIVE_TRAIN pre-check ignores py-fsrs max_seq_len=64 per card; fallback still caught by equality check.
- Document in EVALUATION.md: py-fsrs optimizer trains only first 64 reviews/card and silently returns defaults under 512 informative reviews.
- hlr.py _loss: zero-train-row user gives "Mean of empty slice" warning + NaN loss_history_.
- hlr.py fit(): hitting max_epochs is silent (only epochs_ shows it).

## Phase 3
- .gitignore: results/**/*.json not ignored → git.dirty flips after a run (run_info also records dirty=true while fsrs_benchmark/ is uncommitted).
- data.py:_user_ids cache never refreshes; written non-atomically.
- Smoke runs with very few users may not exercise the parallel FSRS path (all users fall back).
- report.py legend sits far below the reliability panel (large blank gap); cosmetic.
- supplementary.json values come from a script not kept in the repo; commit it as e.g. fsrs_benchmark/supplementary.py or keep the inline note.
- test_hlr_verify.py::test_fit_on_empty_train_set_does_not_crash emits "Mean of empty slice" RuntimeWarning; silence it.
- Ours sampler gives no progress output (verbose=False in the adapter) and no checkpointing; a 9 h fit is lost on shutdown. Consider verbose logging every 100 sweeps.
- run_info.json loses earlier timings/fallbacks when the first invocation is interrupted; consider writing per-model info into the pred_<model>.json sidecar.
- ours.py progress: first line's ETA still includes numba warm-up, so it is overstated; later lines are accurate.
