# Phase 4 report: backlog fixes before the long run (DONE)

## Built
- `fsrs_benchmark/models/ours.py`: `OursModel(progress_every=N)` logs `[HH:MM:SS] [ours] sweep k/total, elapsed, ETA, λ (κ)` to stderr every N sweeps and at the last sweep. The ETA uses the rate since the previous line. Logging only reads sampler state, and a test confirms the samples are unchanged. The runner passes N=100.
- `fsrs_benchmark/run.py`: the `pred_<model>.json` sidecars now carry a `details` key (fit and predict seconds, fallback users, HLR epochs). The `--resume` config check ignores it. On resume, `run_info.json` takes details from the sidecar, so an interrupted run no longer loses timings. Old sidecars, and ones with a malformed `details`, fall back cleanly.
- `fsrs_benchmark/hlr.py`: an empty train set returns zero weights with no RuntimeWarning.
- `fsrs_benchmark/report.py`: the reliability legend sits inside the top panel at lower right. The blank gap is gone and no data is covered.
- `fsrs_benchmark/supplementary.py`: `python -m fsrs_benchmark.supplementary --run_dir ...` regenerates `supplementary.json`, which matches the committed values exactly.

## Tested
260 tests pass with `-W error::RuntimeWarning`, with 0 warnings. A smoke run with `--resume` twice kept its timings in `run_info.json`.

## Known limits
- The first progress line's ETA is overstated because it includes numba warm-up. Later lines are accurate (backlog).
- The sampler still cannot checkpoint, because its loop is inside the unmodifiable model file. A shutdown loses a fit in progress.
