# SRS-Beat results (validation only; branch exp/srsb-beat) — morning summary, 2026-11-18 ~07:50 IST

Goal: beat RWKV-Instant on srs-benchmark (`--algo FSRS-6 --short` test folds). Every number below is on
VALIDATION users 8001-8100 (about 100 users, 3.7-3.8M test rows). The final test users 1-1000 were NOT touched.
Gap = ours minus RWKV-Instant, size-weighted log loss, paired user bootstrap 99% CI. Negative = we beat it.

## Result: nothing beats RWKV-Instant yet. Best gap is about +0.011 to +0.013 (it started at +0.032).

| # | Model | Train users | Gap vs RWKV-Instant | Notes |
|---|---|---|---|---|
| 0 | Old paper GBDT (frozen features) | 450 | +0.0320 | paper reference (users 1-1000) |
| 1 | GBDT+ new features | 76 | +0.029 | 20 users, informal |
| 2 | GBDT+ | 300 | +0.0218 [+0.018, +0.028] | |
| 3 | + user/card memory (umem) | 300 | +0.0180 [+0.015, +0.022] | sib/deck silently missing (bug) |
| 4 | + 63 leaves, lr 0.05 | 300 | +0.0174 [+0.014, +0.022] | sib/deck missing (bug) |
| 5 | #4 with the side-table bug fixed | 300 | +0.0174 [+0.014, +0.022] | sib/deck add ~0 |
| 6 | #4 + online per-user layer | 300 | +0.0139 [+0.011, +0.017] | layer chosen on 8001-8050 |
| 7 | **#5 + online per-user layer** | 300 | **+0.0133 [+0.011, +0.017]**; held-out 8051-8100: +0.0118 [+0.009, +0.016] | best measured |
| 8 | #4 at scale | **1500** | +0.0151 [+0.012, +0.019] | scale helps ~0.002; no side features, no online layer |

Online layer vs plain GBDT: -0.0042 [-0.0063, -0.0027]. Expected (not yet measured): #8 + online layer ~ +0.011.
Also: #8 beats FSRS-6 by -0.0253 [-0.0355, -0.0177] and ties/edges RWKV-Curve (-0.0036 [-0.0116, +0.0013]).

## Not finished
- TAM (attention model): trained on Colab (150 train users; holdout log loss 0.2846 -> 0.2785), but the Colab runtime
  disconnected during validation scoring, so there is NO validation result. Eval is now crash-safe (commit 2764d3b);
  a Kaggle GPU rerun (300 train users) is prepared (code dataset axiom-srsb-beat-code-v4) but not launched: the
  Playwright browser kept hanging on Kaggle pages after midnight.
- Residual GRU on top of the GBDT: built and tested (commit 3aaa82f), Kaggle attempt failed (the HF_TOKEN secret
  added to notebook3f6f101d42 is rejected with 401; probably the pasted value). Local run queued until the other
  session (CPAD lab project) finishes using the PC.

## What it means
- The GBDT line is close to its ceiling: features, memory, tree size and 5x data each give 0.002-0.006. Closing
  the remaining ~0.011 needs a model that adapts to each user (sequence model: TAM or residual GRU) and/or a blend.
- No final test was run. Nothing is pushed.
