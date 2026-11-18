# GBDT experiment log: everything tested since the GBDT began (as of 2026-11-18 ~19:40 IST)

Covers three places:
- **A. Phase 7** on branch `BayesianScheduler` (the GBDT was born here; our own Anki and Duolingo splits).
- **B. Paper Phase 2** on `BayesianScheduler` (first srs-benchmark comparison).
- **C. Branch `exp/gbdt-learned-fsrs`** (learning FSRS weights inside the GBDT).
- **D. Branch `exp/srsb-beat`** (attempt to beat RWKV-Instant on srs-benchmark; this branch).

Numbers are log loss (lower is better) unless stated. "Gap" means ours minus RWKV-Instant, size-weighted, with paired user-bootstrap 99% CIs. A negative gap would mean we beat it.

---

## A. Phase 7: the GBDT is built (BayesianScheduler; docs/EVALUATION.md section 11, docs/reports/phase-7-experiments.md)

**Model.** LightGBM over past-only features: gap since the card's last review; card history (count, lapses, last 3 grades, outcomes and gaps); FSRS-6 default state (S, D, R) replayed over the card's history; running user statistics; Duolingo lexeme recall rate (smoothed, m = 3). Frozen setting: 15 leaves, lr 0.02, trained on all non-first rows, no per-user prior group.

**Tried on validation (EVALUATION 11.3 split):** 38 variants in a first sweep (invalid, see the leak below), then 16 per dataset on fixed code. Six stacking variants (stack3/4, stack3x/4x, stack_gv, blend_gv): none beat GBDT alone, so no blend.

**Feature ablations (validation):**

| | Anki | Duolingo |
|---|---|---|
| all features | 0.3632 | 0.2622 |
| without the FSRS state | 0.3645 (+0.0013) | 0.2624 (+0.0002) |
| without the user group | 0.377 (+0.014) | — |

Most of the gain comes from running user performance, not from FSRS.

**Leak found and fixed.** Duolingo logs a session at one timestamp. The first user-statistics code let rows see same-timestamp outcomes (dev sample 0.2473 was invalid; 0.2578 after the fix). Invalid results are kept in `results/phase7_validation/invalid_leak/`.

**Final test (one look).**

| Model | Anki (100 users) | Duolingo (all users) |
|---|---|---|
| **GBDT** | **0.3584** (vs FSRS-opt −0.021 [−0.028, −0.014]; wins on 82/100 users) | **0.2588** (vs FSRS-opt −0.041) |
| FSRS-6 optimised | 0.3794 | 0.3003 |
| Bayesian v1 (MCMC, unconverged) | 0.4221 | 0.2682 |
| v3 / v2 (MAP) | 0.3780 / 0.3791 | 0.2830 / 0.2892 |

GBDT was best on every metric on both datasets.

**Robustness and overfitting checks.**
- Anki seeds 1 and 2: GBDT − FSRS-opt = −0.0226 and −0.0210.
- Validation → test: Anki 0.3614 → 0.3584; Duolingo 0.2578 → 0.2588.
- Train vs test on Anki: 0.3467 vs 0.3584.
- Duolingo users with no training rows: GBDT 0.2329 vs FSRS-opt 0.2748.

---

## B. Paper Phase 2: first srs-benchmark comparison (BayesianScheduler; results/srsb_1000)

Pooled GBDT, RWKV-style protocol: trained on users 9001–9500, evaluated on test users 1–1000 (`--algo FSRS-6 --short`, TimeSeriesSplit(5) test folds; sizes matched the published files for 1000/1000 users).

| Model | Log loss (unweighted) |
|---|---|
| RWKV-Instant | 0.2860 |
| **GBDT** | **0.3211** |
| RWKV-Curve | 0.3302 |
| GRU / LSTM | 0.3441 / 0.3445 |
| FSRS-6 | 0.3580 |

Size-weighted paired differences, GBDT minus:
- FSRS-6: −0.0235 [−0.0286, −0.0183]
- RWKV-Instant: +0.0320 [+0.0287, +0.0358]
- RWKV-Curve: +0.0006 (a tie)
- GRU: −0.0123

The GBDT hit its 2000-round cap.

---

## C. Branch exp/gbdt-learned-fsrs: learn FSRS inside the GBDT (2026-11-17; validation only)

| Variant | Anki Δ vs GBDT | Duolingo Δ vs GBDT |
|---|---|---|
| Two-stage: per-user optimised FSRS state → GBDT | +0.0016 [−0.0014, +0.0053] (tie) | +0.0001 (tie) |
| Joint: shared FSRS weights + GBDT, alternating | +0.0011 [+0.0001, +0.0024] (worse) | +0.0023 [+0.0010, +0.0037] (worse) |

Conclusion: learning FSRS weights does not help the GBDT. Report: `paper/docs/reports/EXP-learned-fsrs-RESULTS.md` on that branch.

---

## D. Branch exp/srsb-beat: trying to beat RWKV-Instant (2026-11-17 to 18)

**Protocol.**
- Validation users 8001–8100. Settings are chosen on 8001–8050 and reported on 8051–8100 and on all 100.
- Training users come from 2001–8000.
- Test users 1–1000 are touched once, at the end, for the pre-registered model only.
- Every feature is past-only, with leak tests on each new module.

### D1. GBDT feature and scale work

| # | Change | Train users | Gap vs RWKV-Instant |
|---|---|---|---|
| 1 | New feature groups: exact seconds gap, today's activity, answer durations, rating counts, review state, siblings, decks | 76 (local) | +0.029 (20 users, informal) |
| 2 | same | 300 | +0.0218 [+0.018, +0.028] |
| 3 | + `umem`: user and card memory (recall EWMAs, residual vs FSRS, rates by gap/state/card-age bucket) | 300 | +0.0180 [+0.015, +0.022] |
| 4 | 63 leaves, lr 0.05 (tree-size sweep; 255 leaves ran out of memory and crashed the PC) | 300 | +0.0174 [+0.014, +0.022] |
| 5 | Side-table bug fixed (sibling/deck features were silently empty since #3) | 300 | +0.0174: siblings and decks add ~0 |
| 6 | #4 at 1500 training users (Kaggle, 6 h) | 1500 | +0.0151 [+0.012, +0.019]: scale helps ~0.002 |

### D2. Per-user adaptation on top of the GBDT

| Method | Result | Verdict |
|---|---|---|
| **Elo-style online logistic layer** (6 weights per user, one SGD step after each review; variant B, lr 0.003 chosen on 8001–8050) | **−0.0042 [−0.0063, −0.0027] vs GBDT; gap +0.0133** (held-out half +0.0118) | ✅ **best; part of the final model** |
| Residual GRU over the user stream (trained on users 2301–2500, 3 CPU epochs) | −0.0014 vs GBDT; gap +0.0161 | small gain, likely undertrained |
| Per-user boosting (small LightGBM refit on each user's recent past) | log loss 0.314 → 0.369 / 0.397 | ❌ overfits each user |
| Kalman-filter user drift (EKF, 24 settings, K1 and K2) | best 0.394 vs GBDT 0.314; some settings blew up | ❌ over-reacts after breaks |

### D3. Other model families

| Method | Result | Verdict |
|---|---|---|
| TAM: time-aware attention over the last 256 reviews + 32 card reviews, relation and time-decay biases (0.63M params, 300 train users, 8 GPU epochs) | weighted 0.3184; gap +0.0218 | ❌ weaker than the GBDT at this scale |
| User-type experts (forgetful/mid/strong by past-only `u_rate`; from scratch or fine-tuned from the global model) | global +0.0170, from scratch +0.0185, fine-tuned +0.0173 | ❌ the global GBDT already covers user types |
| Mixture-of-experts gate (`moe.py`, cross-fit by user halves) | built and tested; not run at full size (the expert results above made it moot) | — |
| GBDT at 1500 users + online layer | not run (its model exists only in Kaggle output) | — |

### D4. Data analysis (validation users)

**Who we lose to RWKV-Instant on.** Spearman correlation of the per-user gap with:
- retention: −0.72 (lowest-retention quarter gap 0.026 vs highest 0.007);
- same-day review share: +0.49;
- FSRS-6's own gap: +0.75;
- size, intervals and decks: about 0.

We beat RWKV-Instant on only 2/100 users (99.6%+ retention). RWKV covers all 9,999 users, with test sizes identical to FSRS-6.

**Error memory after the online layer.** Correlation of the current error with earlier errors:
- same user: +0.013;
- same card: −0.001;
- same deck: +0.019;
- the mean of today's earlier errors: +0.025.

Almost nothing is left for more Elo-style terms.

**Calibration.** Predicted vs actual recall is within ±0.006 in every gap, card-age, state and probability bucket. The remaining gap is ranking (AUC), not calibration or drift.

**New-attribute probe** (gain on top of the GBDT, cross-fit by user on 8001–8050):
- user running model-error EWMA: +0.0021 (the same signal as the online layer);
- learning history, last-study-day stats, session position: ≈0;
- Elo user/card ratings: negative.

Conclusion: the feature approach is at its ceiling.

**Feature importance.** By gain: `ewm50` 24%, `f_el_over_S` 14%, FSRS R features about 23%, user recency about 15%. The model is essentially user form × FSRS curve.

### D5. Final model and the one-shot test (pre-registered 2026-11-18 ~19:00, commit 38c1199)
- **Final model:** GBDT `results/srsb_beat/sweep_a_l63_side.jsonl.model.txt` (1047 trees, about 196k stored numbers, 78 features, trained on 270 users / 10.4M reviews in about 47 min on 4 CPU cores) plus the Elo online layer (variant B, lr 0.003; 6 weights per user).
- **Validation:**

  | | Log loss (weighted) | Log loss (unweighted) | AUC (weighted) | AUC (unweighted) |
  |---|---|---|---|---|
  | Ours | 0.3098 | 0.2843 | 0.766 | 0.795 |
  | RWKV-Instant (2.76M parameters, trained on 5000 users) | 0.2966 | 0.2683 | 0.796 | 0.826 |

  Ours beats RWKV-Curve, GRU, LSTM and all FSRS versions.
- **Test on users 1–1000:** running (stage 1 GBDT, then stage 2 online layer). Results will go in `results/srsb_beat/final/` and in this file.

### D6. Infrastructure lessons (also in memory note `colab-kaggle-runbook`)
- Pin package versions on Kaggle and Colab. srs-benchmark needs fsrs-optimizer and shape_extensions; pandas 2.x breaks it.
- Dry-run the exact remote command locally first.
- Kaggle hides the logs of failed versions, so wrap cells in try/except.
- A notebook-output input follows the source notebook's latest version.
- Kaggle secrets are shared across notebooks, and re-creating a secret detaches it elsewhere.
- Token-free runs: upload data as a private Kaggle dataset (`axiom-srsb-data-400`).
- Colab browser uploads are slow and idle runtimes disconnect; use Google Drive instead.
- Only one heavy local job at a time on the 19 GB PC (it crashed once).
- The HF token was cycled on 2026-11-18 and is stored only in `.secrets/hf_token`.
