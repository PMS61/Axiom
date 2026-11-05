# FSRS benchmark and improved memory models: final summary (2026-11-06)

## What it does
`fsrs_benchmark/` is a reproducible, leakage-audited pipeline that predicts **whether a learner will
recall a flashcard at a review**. This is the core of any spaced-repetition scheduler. It compares
memory models on two datasets under one protocol:

- **Anki**: `open-spaced-repetition/anki-revlogs-10k` (gated; 100 users, 95,400 scored test reviews).
- **Duolingo**: Settles & Meeder 2016 learning traces (public; ALL 115,222 users, 1,189,355 scored test reviews).

Protocol: a per-user time cutoff at the 80th percentile of `day_offset`; one-step-ahead predictions; scored events
are test, non-first, gap >= 1 day. Metrics: log loss, Brier, AUC, ECE, RMSE(bins), mean bias and
reliability curves. Comparisons use a user-clustered paired bootstrap. The full protocol, results and disclosures
are in `docs/EVALUATION.md` (Sections 1-12).

## Models
| Key | Model |
|---|---|
| `fsrs_default`, `fsrs_opt` | FSRS-6 (py-fsrs 6.3.2), default and per-user optimised |
| `hlr` | Half-Life Regression (Duolingo's official training, ported) |
| `laglogit` | lagged-outcome logistic baseline (teammates' validated baseline) |
| `ours`, `ours_kappa`, `ours_gpu` | **v1**: our Bayesian dynamic retention model (Gibbs / Polya-Gamma MCMC; torch GPU port with the same equations) |
| `ours_v2` | **v2**: FSRS-6 functional forms with 21 weights learned jointly, plus 5 hierarchical user effects (MAP) |
| `ours_v3` | **v3**: v2 plus a penalty on the user-effect scales, a lagged outcome and a fixed-scale per-card effect (MAP) |
| `gbdt` | **GBDT**: LightGBM on past-only features (gap, card history, FSRS-6 state replay, running user stats grouped by session timestamp, smoothed lexeme rate). **Final pick.** |

## Headline results (test sets; log loss, lower is better)
| Model | Anki | Duolingo (all users) |
|---|---|---|
| HLR | 0.680 | 0.418 |
| FSRS-6 default | 0.396 | 0.302 |
| FSRS-6 optimised | 0.379 | 0.300 |
| LagLogit + dt | not run | 0.291 |
| v1 (MCMC) | 0.422 | 0.268 |
| v2 | 0.379 (tie with FSRS-opt) | 0.289 |
| v3 | 0.378 (tie) | 0.283 |
| **GBDT** | **0.358** (-0.021 [-0.028, -0.014] vs FSRS-opt) | **0.259** (-0.041 [-0.042, -0.041]) |

**The GBDT is best on every metric on both datasets.** Its ECE is 0.005 (Anki) and 0.001 (Duolingo). It is better
than FSRS-opt for 82% of Anki users and 78% of Duolingo users. The overfitting checks (validation vs test, train vs
test, never-seen users) are in EVALUATION.md 12.1. All six metrics for every model are in
`results/full_u100_c1000_final/report.md` and `results/duolingo_final/report.md`.

**Disclosures** (EVALUATION.md 11.4):
- The test set was looked at twice: v2 first, then v3 and GBDT, with the pick frozen before the run.
- The v2 selection rule was revised after seeing the validation results (not test).
- A same-session leak in the first GBDT was found and fixed.
- The lexeme rate pools other users' calendar-later train rows (about 0.001 log loss).
- v2 and v3 are MAP fits.
- v1's λ does not converge on Duolingo, although its predictions are stable.
- FSRS-opt kept the default parameters for most Duolingo users.

## How to run
```
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python numpy pandas pyarrow numba scipy matplotlib pytest huggingface_hub "fsrs[optimizer]" torch lightgbm==4.6.0
# Anki: HF token with the dataset licence accepted in ~/.cache/huggingface/token
.venv/bin/python -m fsrs_benchmark.run --n_users 100 --max_cards 1000 --seed 0 --iters 2000 --burn 1000 \
    --run_name full_u100_c1000_final --models fsrs_default,fsrs_opt,hlr,ours,ours_kappa,ours_v2,ours_v3,gbdt
# Duolingo: put settles.acl16.learning_traces.13m.csv.gz in data/duolingo/ (Dataverse file 3091087)
.venv/bin/python -m fsrs_benchmark.run --dataset duolingo --n_users 0 --min_user_rows 1 --max_cards 1000000000 \
    --seed 0 --iters 2000 --burn 1000 --thin 10 --run_name duolingo_final \
    --models ours_gpu,hlr,laglogit,fsrs_default,fsrs_opt,ours_v2,ours_v3,gbdt --device cuda
.venv/bin/python -m fsrs_benchmark.report --run_dir results/<run_name>
```
- `--resume` reuses finished models.
- Runtimes:
  - v1 on Anki takes about 7.5 h on one core.
  - v1 on full Duolingo needs a GPU (about 40 min on a Kaggle T4).
  - On full Duolingo, the FSRS predict takes about 40 min per variant on 4 CPUs, v3 fits in about 13 min and the GBDT in about 5 min.
- Full Duolingo needs about 6-9 GB of RAM, so on a 16-20 GB laptop run one heavy job at a time, or use Kaggle (see `docs/reports/phase-5.md` and phase-7.md).
- Validation experiments: `fsrs_benchmark/experiments/` (run_validation*.py, run_stacking.py).

## How to test
Run each file separately, because torch and numba in one pytest process can hang:
```
for f in fsrs_benchmark/tests/test_*.py; do .venv/bin/python -m pytest -q "$f"; done
```
- All files pass except one known backlog item: `test_task52_54_verify.py::test_resume_config_records_thin_for_gpu_draws`, because `--thin` is not in the resume config.
- `test_gbdt_verify.py` has 1 expected xfail: the accepted cross-user lexeme limitation.

## Reports
`docs/reports/phase-1.md` ... `phase-7.md`; the experiment logs in `phase-6-experiments.md` and `phase-7-experiments.md`;
the PDF in `docs/report/FSRS_Benchmark_Report.pdf`; the backlog in `docs/BACKLOG.md`.

## What is missing
- **The srs-benchmark comparison.** The GBDT has not been run inside the open-spaced-repetition srs-benchmark (10k users), so it has no direct comparison with RWKV, LSTM and GRU. This is the strongest next step for a paper.
- **Sample size and repeats.** Anki uses only 100 users, with one seed and one split.
- **Not attempted:**
  - a GRU sequence model (task 7.4);
  - the v1 plus v2 ensemble (task 7.1; the v1 validation fits were too slow locally);
  - constant-recall baselines in the report;
  - a scheduling simulation (better prediction is not yet shown to give better schedules).
- **Unresolved convergence.** v1's λ and σ² are weakly identified on Duolingo, and no converged chain exists.

## Only the user can do
- **Accounts and tokens.** Keep the HF token (gated Anki data) and the Kaggle account (amalverma27). The private Kaggle dataset `axiom-fsrs-benchmark-code` holds the code and the saved Duolingo predictions.
- **Gated data.** Do not redistribute the anki-revlogs-10k data. The Duolingo data is CC BY-NC (academic use is fine).
- **The black book.** `blackbook/` is the separate LaTeX black book. It predates the Phase 5-7 results and is not committed to this repo.
