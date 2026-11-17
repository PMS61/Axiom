# SRS-Beat results (validation only; branch exp/srsb-beat) — interim, 2026-11-18 ~05:00 IST

Target: beat RWKV-Instant on srs-benchmark (`--algo FSRS-6 --short` test folds). All numbers below are on
VALIDATION users 8001-8100 (never on the final test users 1-1000, which are untouched). Gap = ours minus
RWKV-Instant, size-weighted log loss, paired user bootstrap 99% CI. Negative would mean we beat it.

| # | Model | Train users | Gap vs RWKV-Instant | Notes |
|---|---|---|---|---|
| 0 | Old paper GBDT (frozen features) | 450 (test set 1-1000) | +0.0320 | reference from the paper |
| 1 | GBDT+ new features (secs, today, dur, rcount, state, sib, deck) | 76 local | +0.029 | 20 val users, informal |
| 2 | GBDT+ | 300 | +0.0218 [+0.018, +0.028] | 99 users |
| 3 | GBDT+ + umem (user/card memory) | 300 | +0.0180 [+0.015, +0.022] | sib/deck silently missing (bug) |
| 4 | #3 with 63 leaves, lr 0.05 | 300 | +0.0174 [+0.014, +0.022] | sib/deck missing (bug) |
| 5 | #4 with side-table bug fixed | 300 | +0.0174 [+0.014, +0.022] | sib/deck add ~nothing |
| 6 | #4 + online per-user layer | 300 | +0.0139 [+0.011, +0.017] | layer settings chosen on 8001-8050 |
| 7 | **#5 + online per-user layer** | 300 | **+0.0133 [+0.011, +0.017]** (all 100); +0.0118 [+0.009, +0.016] on held-out 8051-8100 | best so far |

Online layer vs plain GBDT: -0.0042 [-0.0063, -0.0027] (all 100).

Pending: TAM (attention model, Colab; training done, best holdout 0.2785 at epoch 6; validation scoring),
GBDT 1500 train users (Kaggle, ~4 h in), residual GRU (Kaggle attempt failed: HF 401 on notebook 2's secret;
will run locally when the PC is free).

Status: NOTHING beats RWKV-Instant yet. Best gap +0.013.
