# FSRS benchmark: final summary

## What it does
`fsrs_benchmark/` is a reproducible evaluation pipeline on the `open-spaced-repetition/anki-revlogs-10k` dataset. It compares the following models on exactly the same held-out review events, with no data leakage:
- FSRS-6 (py-fsrs 6.3.2), with default parameters and optimised per user
- our Bayesian dynamic retention model (`DynamicRetentionModelDT`, with and without κ)
- Half-Life Regression (Settles & Meeder 2016, ported from Duolingo's official code)

The split is a per-user time cutoff at the 80th percentile of `day_offset`. Each card's history is replayed chronologically, and predictions are one-step-ahead. The metrics are log loss, Brier score, AUC, ECE, RMSE(bins), mean bias and reliability curves. Model comparisons use a paired bootstrap clustered by user.

Read `docs/EVALUATION.md` for the full protocol, choices and results.

## Headline result
FSRS-6 optimised is best on every overall metric: log loss 0.379, AUC 0.777. Next come FSRS-6 default (0.396) and then ours (0.422 for both variants). HLR is far behind (0.680). The gap between ours and FSRS-6 optimised is significant: +0.043 log loss, 95% CI [0.033, 0.055]. **Caveat:** our model's MCMC did not converge at 2000 iterations (R-hat for λ about 2.4). The predictions appear insensitive to this (the two variants correlate at 0.992), but a longer, multi-chain run is needed before final claims.

## How to run
```
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python numpy pandas pyarrow numba scipy matplotlib pytest huggingface_hub "fsrs[optimizer]"
# HF token with the dataset licence accepted: ~/.cache/huggingface/token
.venv/bin/python -m fsrs_benchmark.run --n_users 100 --max_cards 1000 --seed 0 --iters 2000 --burn 1000 --workers 8 --run_name full_u100_c1000
.venv/bin/python -m fsrs_benchmark.report --run_dir results/full_u100_c1000
```
On one laptop core, each of our two model fits takes about 7.5 h, plus about 20 min to predict. FSRS and HLR together take about 15 min. Add `--resume` to reuse finished models after an interruption.

## How to test
`.venv/bin/python -m pytest fsrs_benchmark/tests -q`: 240 tests, about 4-7 min.

## What is missing
- A converged MCMC run for our model: 8-10k iterations, ideally several chains and seeds.
- Results for more than one seed or user sample.
- Low-priority items in `docs/BACKLOG.md`. These include the legend spacing in the reliability plot, the lack of progress output and checkpointing in the sampler, and a script for `supplementary.json`.

## Only you can do
- **Commit:** nothing is committed. `fsrs_benchmark/`, the `docs/` files and `results/full_u100_c1000/` (non-parquet) are untracked. The parquet files are git-ignored on purpose.
- **Dataset licence:** the dataset is gated. Don't redistribute the raw revlogs (e.g. as a public Kaggle dataset).
- **Hugging Face token:** the `fyp-anki-revlogs-read` token is stored in `~/.cache/huggingface/token`. Revoke it on huggingface.co when you no longer need it.
- **HLR citation:** check the original paper's HLR AUC (about 0.54 in Settles & Meeder 2016) before submission.
