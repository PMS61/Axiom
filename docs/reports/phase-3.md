# Phase 3 report: run, compare, document (DONE)

## Built
- `fsrs_benchmark/run.py`: CLI pipeline (download, load, split, fit, predict, merge). It writes `events.parquet`, `pred_<model>.parquet` with a config-fingerprint sidecar, `predictions.parquet`, `run_info.json` and `diagnostics_<ours>.json`. `--resume` reuses predictions only when the fingerprint matches. `--workers` parallelises the FSRS optimiser per user, with results identical to serial.
- `fsrs_benchmark/report.py`: writes `metrics.csv` (overall plus 5 gap bins), `paired.csv` (user-cluster bootstrap), `users.csv`, `user_wins.csv`, `reliability.png`, `gapbins_logloss.png` and `report.md`. It tolerates models that are missing timing info after an interrupted `--resume` run (shows "n/a").
- `docs/EVALUATION.md`: the protocol, leakage guarantees, implementation choices, how to reproduce it, full results (Section 9), and limitations.
- `results/full_u100_c1000/`: the full run (100 users, 1000 cards per user, seed 0, 2000 Gibbs iterations with 1000 burn-in). It also holds `supplementary.json`, the committed numbers that are otherwise derived from git-ignored parquet files.

## Results (95,400 scored test events, 100 users)
| Model | Log loss | Brier | AUC | ECE |
|---|---|---|---|---|
| FSRS-6 optimised | **0.379** | **0.117** | **0.777** | **0.009** |
| FSRS-6 default | 0.396 | 0.121 | 0.745 | 0.012 |
| Ours (learned λ) | 0.422 | 0.131 | 0.692 | 0.025 |
| Ours (+κ) | 0.422 | 0.131 | 0.693 | 0.026 |
| HLR | 0.680 | 0.170 | 0.555 | 0.146 |

Ours minus FSRS-6 optimised: log loss +0.043 [0.033, 0.055]. FSRS-6 optimised beats ours on log loss for 78 of 100 users.

## Known limits (important)
- **The MCMC did not converge** at 2000 iterations. For λ, R-hat is about 2.4 and ESS about 3. Our model's numbers are provisional.
- **But the predictions look insensitive to λ:** the two variants settle at different λ (1.87 vs 1.46), yet their predictions correlate at r = 0.992. A longer run (8-10k iterations, multiple chains) is future work, at about 9 h per 2000 iterations on one core.
- **FSRS-6 optimised fallback:** 1 of 100 users (user 7171) kept the defaults, recomputed from the 512-review rule. Defaults returned by the optimiser itself can't be seen without a refit.
- **Timings for 3 models are missing from run_info.json:** fsrs_default, fsrs_opt and hlr were reused after a laptop shutdown killed the first run. Their timings come from `run.log`.

## Tested
240 tests pass (397 s). The verifier recomputed metrics from `predictions.parquet` and spot-checked the doc tables against the CSVs and JSONs. It also re-derived the fallback count and the correlation, and checked the reproduce flags against `--help`.

## Bugs found and fixed in this phase
- Stale `--resume` reuse: fixed with fingerprint sidecars.
- The MH acceptance diagnostics only covered the last sweep: fixed with a sample-neutral wrapper.
- HLR ignored the run seed: fixed.
- The report crashed on resumed models that had no timing info: fixed.
- Doc: wrong median, unverifiable citations to git-ignored files, and wording issues: fixed.
