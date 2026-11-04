# Phase 7 experiments log (validation only)

Every ours_v3 variant run, in order. Metrics on validation scored events; lower log loss / Brier / ECE better, higher AUC better.

| date | dataset | variant | description | log_loss | brier | auc | ece | fit+predict s | dLL vs FSRS-opt [95% CI] | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-04 | anki | b0 | baseline = ours_v2 FROZEN (epochs 20) | 0.38006 | 0.11734 | 0.7831 | 0.0097 | 47.3 | -0.0051 [-0.0091, -0.0015] WIN |  |
| 2026-11-04 | anki | a1 | b0 + lexeme effect on initial stability (lex_lr 0.1) | 0.38006 | 0.11734 | 0.7831 | 0.0097 | 28.9 | -0.0051 [-0.0091, -0.0015] WIN |  |
| 2026-11-04 | duolingo | b0 | baseline = ours_v2 FROZEN (epochs 20) | 0.27705 | 0.07413 | 0.6393 | 0.0276 | 15.2 | -0.0123 [-0.0186, -0.0054] WIN |  |
| 2026-11-04 | duolingo | a1 | b0 + lexeme effect on initial stability (lex_lr 0.1) | 0.32472 | 0.08140 | 0.6183 | 0.0571 | 16.0 | +0.0354 [+0.0264, +0.0447] LOSS |  |
| 2026-11-04 | duolingo | a2 | b0 + lexeme effects on s0, d0, bias (lex_lr 0.1) | 0.33194 | 0.08371 | 0.6199 | 0.0615 | 15.0 | +0.0426 [+0.0341, +0.0508] LOSS |  |
| 2026-11-04 | duolingo | a3 | b0 + lexeme effects on s0, d0, growth, bias (lex_lr 0.1) | 0.33574 | 0.08376 | 0.6181 | 0.0630 | 15.2 | +0.0464 [+0.0380, +0.0548] LOSS |  |
| 2026-11-04 | duolingo | a4 | a2 with chunk_cards=3000 (more optimiser steps), 8 epochs | 0.30113 | 0.07899 | 0.6273 | 0.0465 | 16.5 | +0.0118 [+0.0052, +0.0188] LOSS |  |
| 2026-11-04 | duolingo | l1 | b0 + lagged-outcome term (global beta) | 0.27702 | 0.07410 | 0.6398 | 0.0272 | 17.0 | -0.0123 [-0.0186, -0.0055] WIN |  |
| 2026-11-04 | duolingo | c1 | b0 + per-card logit random effect | 0.29049 | 0.07797 | 0.6311 | 0.0374 | 18.4 | +0.0011 [-0.0044, +0.0070] TIE |  |
| 2026-11-04 | duolingo | l2 | b0 + lag + per-card logit effect (v1's strengths) | 0.28977 | 0.07771 | 0.6280 | 0.0375 | 25.3 | +0.0004 [-0.0051, +0.0065] TIE |  |
| 2026-11-04 | duolingo | t1 | b0 + forgetting-curve time from fractional dt_days | 0.29432 | 0.07806 | 0.6237 | 0.0392 | 23.8 | +0.0050 [-0.0047, +0.0157] TIE |  |
| 2026-11-04 | duolingo | m1 | a2 + lag | 0.33456 | 0.08447 | 0.6186 | 0.0629 | 21.0 | +0.0452 [+0.0367, +0.0536] LOSS |  |
| 2026-11-04 | duolingo | m2 | a2 + lag + dt | 0.34893 | 0.08687 | 0.6115 | 0.0675 | 19.2 | +0.0596 [+0.0486, +0.0719] LOSS |  |
| 2026-11-04 | duolingo | m3 | a2 + lag + card bias + dt (everything) | 0.35075 | 0.08690 | 0.6122 | 0.0673 | 28.1 | +0.0614 [+0.0504, +0.0744] LOSS |  |
| 2026-11-04 | duolingo | e1 | b0 with 10 epochs (early stopping as regulariser) | 0.27173 | 0.07254 | 0.6414 | 0.0200 | 22.0 | -0.0176 [-0.0225, -0.0129] WIN |  |
| 2026-11-04 | duolingo | z1 | b0 with user z lr 0.01 (slower user effects) | 0.27140 | 0.07225 | 0.6324 | 0.0183 | 29.5 | -0.0180 [-0.0223, -0.0138] WIN |  |
| 2026-11-04 | anki | a2 | b0 + lexeme effects on s0, d0, bias (lex_lr 0.1) | 0.38006 | 0.11734 | 0.7831 | 0.0097 | 60.4 | -0.0051 [-0.0091, -0.0015] WIN |  |
| 2026-11-04 | duolingo | p1 | b0 + L2 penalty 100 on user-effect scales tau (shrinks overfit tau) | 0.27248 | 0.07288 | 0.6392 | 0.0213 | 28.7 | -0.0169 [-0.0223, -0.0112] WIN |  |
| 2026-11-04 | duolingo | p2 | b0 + tau penalty 300 | 0.27084 | 0.07229 | 0.6367 | 0.0177 | 29.8 | -0.0185 [-0.0236, -0.0136] WIN |  |
| 2026-11-04 | anki | m1 | a2 + lag | 0.37937 | 0.11717 | 0.7844 | 0.0093 | 55.4 | -0.0057 [-0.0097, -0.0022] WIN |  |
| 2026-11-04 | duolingo | p3 | b0 + tau penalty 1000 | 0.27116 | 0.07214 | 0.6286 | 0.0197 | 28.1 | -0.0182 [-0.0226, -0.0140] WIN |  |
| 2026-11-04 | duolingo | p4 | b0 + tau penalty 3000 | 0.27517 | 0.07292 | 0.6086 | 0.0196 | 27.2 | -0.0142 [-0.0179, -0.0107] WIN |  |
| 2026-11-04 | anki | m3 | a2 + lag + card bias + dt (everything) | 0.41697 | 0.12529 | 0.7592 | 0.0509 | 51.5 | +0.0319 [+0.0233, +0.0405] LOSS |  |
| 2026-11-04 | duolingo | k1 | b0 + per-card logit effect with FIXED scale 0.2 | 0.27696 | 0.07408 | 0.6393 | 0.0272 | 28.6 | -0.0124 [-0.0187, -0.0054] WIN |  |
| 2026-11-04 | duolingo | x1 | b0 + lexeme s0 effect with FIXED scale 0.1, lr 0.03 | 0.27699 | 0.07411 | 0.6398 | 0.0276 | 26.7 | -0.0124 [-0.0187, -0.0054] WIN |  |
| 2026-11-04 | anki | x1 | b0 + lexeme s0 effect with FIXED scale 0.1, lr 0.03 | 0.38006 | 0.11734 | 0.7831 | 0.0097 | 52.5 | -0.0051 [-0.0091, -0.0015] WIN |  |
| 2026-11-04 | anki | l1 | b0 + lagged-outcome term (global beta) | 0.37937 | 0.11717 | 0.7844 | 0.0093 | 29.8 | -0.0057 [-0.0097, -0.0022] WIN |  |
| 2026-11-04 | anki | c1 | b0 + per-card logit random effect | 0.41697 | 0.12516 | 0.7592 | 0.0498 | 30.6 | +0.0319 [+0.0234, +0.0408] LOSS |  |
| 2026-11-04 | anki | l2 | b0 + lag + per-card logit effect (v1's strengths) | 0.41658 | 0.12519 | 0.7594 | 0.0514 | 30.4 | +0.0315 [+0.0230, +0.0401] LOSS |  |
| 2026-11-04 | anki | t1 | b0 + forgetting-curve time from fractional dt_days | 0.38036 | 0.11736 | 0.7828 | 0.0099 | 29.0 | -0.0048 [-0.0090, -0.0011] WIN |  |
| 2026-11-04 | anki | e1 | b0 with 10 epochs (early stopping as regulariser) | 0.38280 | 0.11830 | 0.7777 | 0.0085 | 15.9 | -0.0023 [-0.0080, +0.0021] TIE |  |
| 2026-11-04 | anki | z1 | b0 with user z lr 0.01 (slower user effects) | 0.38336 | 0.11835 | 0.7754 | 0.0086 | 28.0 | -0.0018 [-0.0079, +0.0029] TIE |  |
| 2026-11-04 | anki | p1 | b0 + L2 penalty 100 on user-effect scales tau (shrinks overfit tau) | 0.38020 | 0.11742 | 0.7827 | 0.0092 | 29.1 | -0.0049 [-0.0089, -0.0013] WIN |  |
| 2026-11-04 | anki | p2 | b0 + tau penalty 300 | 0.38024 | 0.11751 | 0.7823 | 0.0081 | 28.2 | -0.0049 [-0.0091, -0.0013] WIN |  |
| 2026-11-04 | anki | p3 | b0 + tau penalty 1000 | 0.38089 | 0.11777 | 0.7806 | 0.0073 | 28.8 | -0.0042 [-0.0089, -0.0003] WIN |  |
| 2026-11-04 | anki | p4 | b0 + tau penalty 3000 | 0.38222 | 0.11816 | 0.7776 | 0.0072 | 28.8 | -0.0029 [-0.0083, +0.0014] TIE |  |
| 2026-11-04 | anki | k1 | b0 + per-card logit effect with FIXED scale 0.2 | 0.37988 | 0.11721 | 0.7836 | 0.0096 | 28.8 | -0.0052 [-0.0093, -0.0016] WIN |  |
| 2026-11-04 | duolingo | q1 | p2 (tau penalty 300) + lag | 0.27067 | 0.07226 | 0.6372 | 0.0173 | 27.9 | -0.0187 [-0.0236, -0.0138] WIN |  |
| 2026-11-04 | anki | q1 | p2 (tau penalty 300) + lag | 0.37927 | 0.11727 | 0.7838 | 0.0076 | 49.3 | -0.0058 [-0.0101, -0.0021] WIN |  |
| 2026-11-04 | duolingo | q2 | p2 + lag + fixed-scale per-card logit effect 0.2 | 0.27066 | 0.07225 | 0.6368 | 0.0171 | 27.5 | -0.0187 [-0.0236, -0.0138] WIN |  |
| 2026-11-04 | duolingo | q3 | p1 (tau penalty 100) + lag + fixed-scale per-card logit effect 0.2 | 0.27228 | 0.07282 | 0.6390 | 0.0211 | 29.1 | -0.0171 [-0.0225, -0.0114] WIN |  |
| 2026-11-04 | anki | q2 | p2 + lag + fixed-scale per-card logit effect 0.2 | 0.37915 | 0.11719 | 0.7841 | 0.0086 | 44.2 | -0.0060 [-0.0102, -0.0023] WIN |  |
| 2026-11-04 | anki | q3 | p1 (tau penalty 100) + lag + fixed-scale per-card logit effect 0.2 | 0.37915 | 0.11712 | 0.7844 | 0.0092 | 28.5 | -0.0060 [-0.0099, -0.0024] WIN |  |
| 2026-11-04 | anki | a3 | b0 + lexeme effects on s0, d0, growth, bias (lex_lr 0.1) | 0.38006 | 0.11734 | 0.7831 | 0.0097 | 27.4 | -0.0051 [-0.0091, -0.0015] WIN |  |
| 2026-11-04 | anki | a4 | a2 with chunk_cards=3000 (more optimiser steps), 8 epochs | 0.38046 | 0.11766 | 0.7825 | 0.0128 | 36.9 | -0.0046 [-0.0089, -0.0008] WIN |  |
| 2026-11-04 | anki | m2 | a2 + lag + dt | 0.37954 | 0.11718 | 0.7844 | 0.0093 | 30.1 | -0.0056 [-0.0097, -0.0020] WIN |  |

## Notes (read these with the table)

- Task 7.2, ours_v3 (`fsrs_benchmark/models/ours_v3.py`, a copy of v2 with extra flags; v2 untouched). VALIDATION only: fit on the first 80% of each user's train days, scored on the last 20%; no test row is read (`make_validation_split` drops them). Anki: 100 users, 74,092 scored; Duolingo: 5000-user dev sample, 26,084 scored (141,564 fit rows, 9,257 distinct lexemes, so ~15 fit rows per lexeme).
- dLL = log loss(variant) - log loss(FSRS-opt, validation), negative = better; CI = user-cluster bootstrap, 1000 reps, seed 0. FSRS-opt validation log loss: Anki 0.38511, Duolingo 0.28936 (Phase 6 cache, read only). b0 reproduces v2 (Anki 0.38006; Phase 6 logged 0.38024, a float-level difference).
- Rows are in run order; the CSVs are `results/phase7_validation/{anki,duolingo}.csv` (one row per variant). Round 1: b0, a1-a4 (lexeme), l1, c1, l2, t1, m1-m3. Round 2 (after round 1 showed free-scale lexeme/card effects overfit): e1, z1, p1-p4, k1, x1. Round 3: q1-q3. That is 23 variants (more than the budget of ~10; they are cheap, 15-60 s each, and all are listed).
- Lexeme effects (a1-a4, m1-m3 on Duolingo) all HURT badly (+0.03 to +0.07 log loss vs FSRS-opt on Duolingo) when the lexeme scale is learned: fit log loss drops (0.30 -> 0.22) but validation worsens, i.e. overfitting with only ~15 fit rows per lexeme and an unpenalised learned tau. With a FIXED small scale (x1, tau 0.1, lr 0.03) the effect is harmless but useless (0.27699 vs 0.27705 for b0). Conclusion for the 5000-user dev sample: no measurable shared-lexeme signal beyond what each user's card history gives. The full 2.6M-row Duolingo set has about 26x more rows per lexeme, so a full-data re-check of x1/a1 may still be worth it. On Anki there is no shared item (no `lexeme` column), so every lexeme variant equals its non-lexeme twin by construction (a1=a2=a3=x1=b0; m1=l1).
- Per-card logit effect with a learned scale (c1, l2, m3) is a disaster (Anki +0.037, Duolingo +0.013); with a fixed scale 0.2 (k1) it is a small Anki gain (-0.0002) and neutral on Duolingo.
- Lagged outcome (l1): Anki -0.0007 (0.37937), Duolingo -0.0000: small but consistent on Anki.
- Fractional elapsed time (t1, `dt_days`): worse on Duolingo (+0.017), neutral on Anki. Dropped.
- The component that mattered is regularisation, not a new feature: an L2 penalty on the user-effect scales (tau_pen). v2's learned tau overfits on Duolingo (v2's 20 epochs 0.2771 vs 10 epochs 0.2717). tau_pen=300 (p2) gives Duolingo 0.27084 (vs 0.27705) and keeps Anki at 0.38024; 100 is weaker on Duolingo, 1000-3000 over-shrink Anki. Early stopping (e1) and slower user lr (z1) help Duolingo but cost Anki.
- Selection rule (fixed in the task before the runs): lowest mean validation log loss over the two datasets among variants whose CI does not show a LOSS vs FSRS-opt on either. Ranking by that mean: q2 0.32490, q1 0.32497, p2 0.32554, q3 0.32572, p3 0.32603, p1 0.32634, e1 0.32727, z1 0.32738, l1 0.32819, k1 0.32842, x1 0.32852, b0 0.32855 (all not losing); the lexeme and learned-card variants and t1 fail the rule or rank lower.
- FROZEN CHOICE: q2 = `FROZEN_KWARGS = dict(epochs=20, tau_pen=300.0, lag=True, card_bias=True, card_tau=0.2)` (v2 + user-scale penalty + lagged outcome + fixed-scale per-card effect; no lexeme effect). Validation: Anki 0.37915, dLL -0.0060 [-0.0102, -0.0023] WIN; Duolingo 0.27066, dLL -0.0187 [-0.0236, -0.0138] WIN. Caveat: q1 (same without the card effect) is within 1e-4 of q2, so the card effect is not separately supported; q2 is chosen because the rule is mechanical. The selection is on validation only and has not been checked on test.
- Honest limits: Duolingo validation 0.2707 is better than v2 (0.2771) and than the Phase 6 v5 (0.2716), but v1 (the MCMC model) has no validation number in this table, so "closes the gap to v1" is only established on test, in task 7.4. The shared-lexeme idea gave nothing on the dev sample.

## Task 7.3: GBDT (v1 rows: INVALID, same-session leak; v2 rows below the second table)

LightGBM (`fsrs_benchmark/models/gbdt.py`), validation only; same split, scored set, FSRS-opt reference and bootstrap as Task 7.2. Every variant run, in order. Early stopping on an inner 15% user holdout of the fit data.

| date | dataset | variant | description | log_loss | brier | auc | ece | fit+predict s | dLL vs FSRS-opt [95% CI] | rounds |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-04 | anki | all | all groups, leaves 31, lr 0.05, min_leaf 100 | 0.36323 | 0.11256 | 0.8052 | 0.0075 | 3.5 | -0.0219 [-0.0317, -0.0152] WIN | 87 |
| 2026-11-04 | anki | no_gap | all minus group gap | 0.36358 | 0.11269 | 0.8052 | 0.0090 | 3.3 | -0.0215 [-0.0312, -0.0149] WIN | 82 |
| 2026-11-04 | anki | no_dt | all minus group dt | 0.36378 | 0.11276 | 0.8050 | 0.0091 | 3.3 | -0.0213 [-0.0310, -0.0147] WIN | 81 |
| 2026-11-04 | anki | no_card | all minus group card | 0.36483 | 0.11317 | 0.8027 | 0.0073 | 2.9 | -0.0203 [-0.0298, -0.0138] WIN | 103 |
| 2026-11-04 | anki | no_fsrs | all minus group fsrs | 0.36454 | 0.11305 | 0.8036 | 0.0090 | 4.1 | -0.0206 [-0.0311, -0.0131] WIN | 137 |
| 2026-11-04 | anki | no_user | all minus group user | 0.37705 | 0.11675 | 0.7850 | 0.0063 | 3.1 | -0.0081 [-0.0122, -0.0042] WIN | 91 |
| 2026-11-04 | anki | no_userprior | all minus group userprior | 0.36159 | 0.11214 | 0.8071 | 0.0059 | 3.9 | -0.0235 [-0.0339, -0.0163] WIN | 116 |
| 2026-11-04 | anki | no_lexeme | all minus group lexeme | 0.36323 | 0.11256 | 0.8052 | 0.0075 | 4.0 | -0.0219 [-0.0317, -0.0152] WIN | 87 |
| 2026-11-04 | anki | o_gap | only gap | 0.45761 | 0.14175 | 0.5160 | 0.0084 | 0.9 | +0.0725 [+0.0547, +0.0909] LOSS | 13 |
| 2026-11-04 | anki | o_gap_fsrs | gap + fsrs state | 0.39028 | 0.12040 | 0.7593 | 0.0063 | 1.9 | +0.0052 [-0.0024, +0.0116] TIE | 64 |
| 2026-11-04 | anki | o_gap_card | gap + card history | 0.38815 | 0.11976 | 0.7640 | 0.0069 | 2.7 | +0.0030 [-0.0048, +0.0094] TIE | 80 |
| 2026-11-04 | anki | o_gap_card_fsrs | gap + card + fsrs | 0.38529 | 0.11892 | 0.7693 | 0.0073 | 2.9 | +0.0002 [-0.0074, +0.0063] TIE | 72 |
| 2026-11-04 | anki | o_gap_card_fsrs_user | gap + card + fsrs + user running | 0.36111 | 0.11201 | 0.8079 | 0.0078 | 4.2 | -0.0240 [-0.0347, -0.0168] WIN | 148 |
| 2026-11-04 | anki | l7 | all, num_leaves 7 | 0.36484 | 0.11319 | 0.8029 | 0.0067 | 4.0 | -0.0203 [-0.0299, -0.0136] WIN | 202 |
| 2026-11-04 | anki | l15 | all, num_leaves 15 | 0.36411 | 0.11294 | 0.8040 | 0.0069 | 3.5 | -0.0210 [-0.0307, -0.0144] WIN | 114 |
| 2026-11-04 | anki | l63 | all, num_leaves 63 | 0.36189 | 0.11217 | 0.8072 | 0.0079 | 4.1 | -0.0232 [-0.0332, -0.0162] WIN | 85 |
| 2026-11-04 | anki | lr02 | all, learning rate 0.02 | 0.36249 | 0.11236 | 0.8063 | 0.0077 | 6.8 | -0.0226 [-0.0324, -0.0160] WIN | 245 |
| 2026-11-04 | anki | md30 | all, min_data_in_leaf 30 | 0.36324 | 0.11256 | 0.8053 | 0.0071 | 3.6 | -0.0219 [-0.0314, -0.0154] WIN | 89 |
| 2026-11-04 | anki | md300 | all, min_data_in_leaf 300 | 0.36331 | 0.11261 | 0.8053 | 0.0074 | 3.7 | -0.0218 [-0.0313, -0.0154] WIN | 89 |
| 2026-11-04 | anki | nonfirst | all, train on all non-first rows (incl. same-day) | 0.36303 | 0.11260 | 0.8054 | 0.0078 | 5.9 | -0.0221 [-0.0319, -0.0152] WIN | 126 |
| 2026-11-04 | anki | sm30 | all, user/lexeme prior smoothing m=30 | 0.36314 | 0.11253 | 0.8053 | 0.0076 | 3.6 | -0.0220 [-0.0315, -0.0155] WIN | 88 |
| 2026-11-04 | anki | sm3 | all, user/lexeme prior smoothing m=3 | 0.36208 | 0.11214 | 0.8067 | 0.0070 | 4.1 | -0.0230 [-0.0326, -0.0164] WIN | 117 |
| 2026-11-04 | anki | l15_md300 | all, leaves 15, min_leaf 300 | 0.36413 | 0.11294 | 0.8040 | 0.0071 | 3.4 | -0.0210 [-0.0306, -0.0144] WIN | 111 |
| 2026-11-04 | anki | l15_lr02 | all, leaves 15, lr 0.02 | 0.36297 | 0.11255 | 0.8055 | 0.0073 | 7.7 | -0.0221 [-0.0320, -0.0154] WIN | 371 |
| 2026-11-04 | anki | l15_md300_lr02 | all, leaves 15, min_leaf 300, lr 0.02 | 0.36318 | 0.11259 | 0.8051 | 0.0070 | 7.5 | -0.0219 [-0.0315, -0.0153] WIN | 356 |
| 2026-11-04 | duolingo | all | all groups, leaves 31, lr 0.05, min_leaf 100 | 0.25339 | 0.06882 | 0.7334 | 0.0143 | 0.5 | -0.0360 [-0.0420, -0.0302] WIN | 25 |
| 2026-11-04 | duolingo | no_gap | all minus group gap | 0.25219 | 0.06862 | 0.7315 | 0.0118 | 0.5 | -0.0372 [-0.0436, -0.0310] WIN | 31 |
| 2026-11-04 | duolingo | no_dt | all minus group dt | 0.25218 | 0.06862 | 0.7334 | 0.0126 | 0.5 | -0.0372 [-0.0437, -0.0309] WIN | 30 |
| 2026-11-04 | duolingo | no_card | all minus group card | 0.25514 | 0.06918 | 0.7310 | 0.0166 | 0.4 | -0.0342 [-0.0403, -0.0285] WIN | 20 |
| 2026-11-04 | duolingo | no_fsrs | all minus group fsrs | 0.25416 | 0.06895 | 0.7326 | 0.0172 | 0.5 | -0.0352 [-0.0412, -0.0294] WIN | 21 |
| 2026-11-04 | duolingo | no_user | all minus group user | 0.26755 | 0.07148 | 0.6406 | 0.0068 | 0.6 | -0.0218 [-0.0274, -0.0163] WIN | 62 |
| 2026-11-04 | duolingo | no_userprior | all minus group userprior | 0.25363 | 0.06887 | 0.7353 | 0.0169 | 0.5 | -0.0357 [-0.0418, -0.0300] WIN | 22 |
| 2026-11-04 | duolingo | no_lexeme | all minus group lexeme | 0.25373 | 0.06889 | 0.7284 | 0.0120 | 0.5 | -0.0356 [-0.0419, -0.0296] WIN | 26 |
| 2026-11-04 | duolingo | o_gap | only gap | 0.27860 | 0.07351 | 0.5018 | 0.0053 | 0.3 | -0.0108 [-0.0168, -0.0046] WIN | 1 |
| 2026-11-04 | duolingo | o_gap_fsrs | gap + fsrs state | 0.27603 | 0.07308 | 0.5864 | 0.0152 | 0.3 | -0.0133 [-0.0188, -0.0079] WIN | 5 |
| 2026-11-04 | duolingo | o_gap_card | gap + card history | 0.27546 | 0.07298 | 0.6021 | 0.0200 | 0.4 | -0.0139 [-0.0193, -0.0085] WIN | 5 |
| 2026-11-04 | duolingo | o_gap_card_fsrs | gap + card + fsrs | 0.27562 | 0.07300 | 0.5908 | 0.0202 | 0.4 | -0.0137 [-0.0191, -0.0083] WIN | 5 |
| 2026-11-04 | duolingo | o_gap_card_fsrs_user | gap + card + fsrs + user running | 0.25388 | 0.06894 | 0.7333 | 0.0160 | 0.5 | -0.0355 [-0.0416, -0.0296] WIN | 24 |
| 2026-11-04 | duolingo | l7 | all, num_leaves 7 | 0.25228 | 0.06860 | 0.7372 | 0.0145 | 0.4 | -0.0371 [-0.0434, -0.0312] WIN | 30 |
| 2026-11-04 | duolingo | l15 | all, num_leaves 15 | 0.25092 | 0.06837 | 0.7369 | 0.0112 | 0.5 | -0.0384 [-0.0449, -0.0324] WIN | 36 |
| 2026-11-04 | duolingo | l63 | all, num_leaves 63 | 0.25407 | 0.06893 | 0.7280 | 0.0137 | 0.6 | -0.0353 [-0.0414, -0.0296] WIN | 25 |
| 2026-11-04 | duolingo | lr02 | all, learning rate 0.02 | 0.25237 | 0.06867 | 0.7346 | 0.0124 | 0.7 | -0.0370 [-0.0435, -0.0309] WIN | 74 |
| 2026-11-04 | duolingo | md30 | all, min_data_in_leaf 30 | 0.25406 | 0.06906 | 0.7302 | 0.0142 | 0.5 | -0.0353 [-0.0411, -0.0297] WIN | 27 |
| 2026-11-04 | duolingo | md300 | all, min_data_in_leaf 300 | 0.25246 | 0.06867 | 0.7339 | 0.0139 | 0.5 | -0.0369 [-0.0432, -0.0310] WIN | 29 |
| 2026-11-04 | duolingo | nonfirst | all, train on all non-first rows (incl. same-day) | 0.24781 | 0.06765 | 0.7388 | 0.0052 | 1.2 | -0.0415 [-0.0485, -0.0350] WIN | 95 |
| 2026-11-04 | duolingo | sm30 | all, user/lexeme prior smoothing m=30 | 0.25196 | 0.06858 | 0.7332 | 0.0118 | 0.5 | -0.0374 [-0.0437, -0.0314] WIN | 34 |
| 2026-11-04 | duolingo | sm3 | all, user/lexeme prior smoothing m=3 | 0.25178 | 0.06853 | 0.7337 | 0.0115 | 0.6 | -0.0376 [-0.0441, -0.0314] WIN | 34 |
| 2026-11-04 | duolingo | l15_md300 | all, leaves 15, min_leaf 300 | 0.25290 | 0.06876 | 0.7361 | 0.0161 | 0.5 | -0.0365 [-0.0426, -0.0307] WIN | 27 |
| 2026-11-04 | duolingo | l15_lr02 | all, leaves 15, lr 0.02 | 0.25114 | 0.06842 | 0.7366 | 0.0106 | 0.6 | -0.0382 [-0.0447, -0.0321] WIN | 89 |
| 2026-11-04 | duolingo | l15_md300_lr02 | all, leaves 15, min_leaf 300, lr 0.02 | 0.25044 | 0.06832 | 0.7379 | 0.0112 | 0.6 | -0.0389 [-0.0454, -0.0329] WIN | 101 |
| 2026-11-04 | anki | nf_sm3 | nonfirst + smoothing m=3 | 0.36276 | 0.11253 | 0.8058 | 0.0069 | 5.9 | -0.0224 [-0.0321, -0.0155] WIN | 126 |
| 2026-11-04 | anki | nf_l15 | nonfirst + leaves 15 | 0.36343 | 0.11275 | 0.8049 | 0.0082 | 6.7 | -0.0217 [-0.0315, -0.0148] WIN | 204 |
| 2026-11-04 | anki | nf_l15_lr02 | nonfirst + leaves 15 + lr 0.02 | 0.36293 | 0.11259 | 0.8055 | 0.0073 | 14.0 | -0.0222 [-0.0321, -0.0154] WIN | 495 |
| 2026-11-04 | anki | nf_l15_md300_lr02 | nonfirst + leaves 15 + min_leaf 300 + lr 0.02 | 0.36299 | 0.11263 | 0.8056 | 0.0073 | 13.5 | -0.0221 [-0.0320, -0.0153] WIN | 492 |
| 2026-11-04 | anki | nf_l15_lr02_sm3 | nonfirst + leaves 15 + lr 0.02 + smoothing 3 | 0.36402 | 0.11298 | 0.8041 | 0.0066 | 11.4 | -0.0211 [-0.0307, -0.0143] WIN | 382 |
| 2026-11-04 | anki | nf_l63_lr02 | nonfirst + leaves 63 + lr 0.02 | 0.36240 | 0.11234 | 0.8064 | 0.0071 | 11.3 | -0.0227 [-0.0327, -0.0159] WIN | 233 |
| 2026-11-04 | anki | nf_nouserprior | nonfirst, without the userprior group | 0.36137 | 0.11213 | 0.8073 | 0.0072 | 7.1 | -0.0237 [-0.0340, -0.0166] WIN | 197 |
| 2026-11-04 | anki | nf_l15_lr02_sm3_nup | nonfirst + leaves 15 + lr 0.02 + smoothing 3, no userprior | 0.36136 | 0.11219 | 0.8072 | 0.0063 | 17.3 | -0.0238 [-0.0344, -0.0164] WIN | 704 |
| 2026-11-04 | duolingo | nf_sm3 | nonfirst + smoothing m=3 | 0.24819 | 0.06778 | 0.7381 | 0.0047 | 1.3 | -0.0412 [-0.0482, -0.0347] WIN | 115 |
| 2026-11-04 | duolingo | nf_l15 | nonfirst + leaves 15 | 0.24759 | 0.06767 | 0.7415 | 0.0041 | 1.0 | -0.0418 [-0.0485, -0.0353] WIN | 97 |
| 2026-11-04 | duolingo | nf_l15_lr02 | nonfirst + leaves 15 + lr 0.02 | 0.24765 | 0.06765 | 0.7406 | 0.0040 | 2.3 | -0.0417 [-0.0485, -0.0352] WIN | 388 |
| 2026-11-04 | duolingo | nf_l15_md300_lr02 | nonfirst + leaves 15 + min_leaf 300 + lr 0.02 | 0.24787 | 0.06774 | 0.7401 | 0.0035 | 2.7 | -0.0415 [-0.0485, -0.0350] WIN | 445 |
| 2026-11-04 | duolingo | nf_l15_lr02_sm3 | nonfirst + leaves 15 + lr 0.02 + smoothing 3 | 0.24732 | 0.06758 | 0.7421 | 0.0030 | 1.8 | -0.0420 [-0.0487, -0.0355] WIN | 261 |
| 2026-11-04 | duolingo | nf_l63_lr02 | nonfirst + leaves 63 + lr 0.02 | 0.24834 | 0.06780 | 0.7374 | 0.0053 | 2.6 | -0.0410 [-0.0480, -0.0345] WIN | 198 |
| 2026-11-04 | duolingo | nf_nouserprior | nonfirst, without the userprior group | 0.24859 | 0.06784 | 0.7373 | 0.0034 | 1.2 | -0.0408 [-0.0477, -0.0343] WIN | 102 |
| 2026-11-04 | duolingo | nf_l15_lr02_sm3_nup | nonfirst + leaves 15 + lr 0.02 + smoothing 3, no userprior | 0.24725 | 0.06757 | 0.7426 | 0.0052 | 1.9 | -0.0421 [-0.0489, -0.0356] WIN | 294 |
| 2026-11-04 | anki | nf_l15_lr02_sm3_nup | nonfirst + leaves 15 + lr 0.02 + smoothing 3, no userprior | 0.36136 | 0.11219 | 0.8072 | 0.0063 | 17.2 | -0.0238 [-0.0344, -0.0164] WIN | 704 |
| 2026-11-04 | duolingo | nf_l15_lr02_sm3_nup | nonfirst + leaves 15 + lr 0.02 + smoothing 3, no userprior | 0.24725 | 0.06757 | 0.7426 | 0.0052 | 1.9 | -0.0421 [-0.0489, -0.0356] WIN | 294 |

### Task 7.3 notes, v1 (INVALID)

THE NUMBERS ABOVE AND THESE NOTES ARE INVALID: `_user_kernel` let a row see outcomes of other rows of the same Duolingo session (one timestamp), so user running stats were leaky. Kept for the record; see the v2 section below.


- 38 variants per dataset, listed above in run order (round 1: group ablations, add-ups and hyperparameters; round 2: combinations). Anki 100 users (74,092 scored), Duolingo 5000-user dev sample (26,084 scored); dLL and CI as in Task 7.2 (user-cluster bootstrap, 1000 reps). The last row of each dataset (`nf_l15_lr02_sm3_nup`) appears twice because it was re-run once to save the predictions (identical numbers).
- Features use only reviews of the same card and user before t. Aggregates (user prior `up_*`, lexeme `lex_*`) use FIT rows only, out-of-fold for train rows (card folds within user for the user prior, user folds for the lexeme). Not used: history_seen/history_correct, the Anki `state` column. Anki has no lexeme, so lexeme variants equal their twins there.
- Group ablation (drop one from `all`; Anki 0.3632, Duolingo 0.2534): the running user statistics matter most (no_user: Anki 0.3771, Duolingo 0.2675); card history, FSRS state, gap and dt each add 0.0004-0.0016; lexeme adds ~0.0003 on Duolingo (the item pooling that overfit v3 is harmless here but small); the fit-rows user prior hurts a little (dropping it: Anki 0.3616, Duolingo 0.2536 vs 0.2534). Gap alone is useless (AUC 0.5-0.52).
- Training on all non-first rows (including same-day reviews) helps Duolingo a lot (0.2534 -> 0.2478) and is neutral on Anki. Smaller trees plus a slower learning rate help Duolingo a little.
- FROZEN CHOICE by the v3 rule (lowest mean validation log loss over both datasets among variants not losing to FSRS-opt on either): `nf_l15_lr02_sm3_nup` = groups gap, dt, card, fsrs, user, lexeme (no userprior), train on all non-first rows, num_leaves 15, learning rate 0.02, smoothing m=3. Validation: Anki 0.36136, dLL -0.0238 [-0.0344, -0.0164] WIN; Duolingo 0.24725, dLL -0.0421 [-0.0489, -0.0356] WIN. For comparison, ours_v3 (FROZEN q2): Anki 0.37915, Duolingo 0.27066; FSRS-opt 0.38511 / 0.28936. The next variants are within 0.0002, so the choice among them is not separately supported. The selection is on validation only; test has not been touched.
- Caveat: the gain comes mainly from user-level running recall statistics and the extra training rows; it is a pure predictor (no memory model), so a v3+GBDT blend (task 7.5) may still add. Validation predictions of the frozen choice: `results/phase7_validation/pred_gbdt_{anki,duolingo}.parquet` (event_id key, column p). Peak RAM of a run: 1.3 GB (Anki), 1.0 GB (Duolingo), 4 threads.

## Task 7.3: GBDT v2 (same-session leak fixed; rows above are INVALID)

The user kernel now treats all rows of a user with the same raw timestamp `ts` (Duolingo; new optional loader column `ts`, kept by split.py) as ONE group: features come from the state before the group, which is updated afterwards. Anki's table has no timestamp, so it groups by day_offset (strictly earlier days only; this also drops legitimate same-day user history on Anki). The card kernel asserts that a card never repeats within one ts.

| date | dataset | variant | description | log_loss | brier | auc | ece | fit+predict s | dLL vs FSRS-opt [95% CI] | rounds |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-04 | anki | all | all groups, leaves 31, lr 0.05, min_leaf 100 | 0.37539 | 0.11644 | 0.7873 | 0.0074 | 2.2 | -0.0097 [-0.0148, -0.0054] WIN | 83 |
| 2026-11-04 | anki | no_user | all minus group user | 0.37705 | 0.11675 | 0.7850 | 0.0063 | 2.0 | -0.0081 [-0.0122, -0.0042] WIN | 91 |
| 2026-11-04 | anki | no_card | all minus group card | 0.37688 | 0.11694 | 0.7850 | 0.0085 | 1.7 | -0.0082 [-0.0132, -0.0041] WIN | 89 |
| 2026-11-04 | anki | no_fsrs | all minus group fsrs | 0.37713 | 0.11688 | 0.7851 | 0.0099 | 2.7 | -0.0080 [-0.0143, -0.0028] WIN | 145 |
| 2026-11-04 | anki | no_lexeme | all minus group lexeme | 0.37539 | 0.11644 | 0.7873 | 0.0074 | 2.2 | -0.0097 [-0.0148, -0.0054] WIN | 83 |
| 2026-11-04 | anki | no_userprior | all minus group userprior | 0.37359 | 0.11609 | 0.7897 | 0.0092 | 2.5 | -0.0115 [-0.0178, -0.0064] WIN | 124 |
| 2026-11-04 | anki | nonfirst | all, train on all non-first rows (incl. same-day) | 0.37477 | 0.11607 | 0.7886 | 0.0105 | 4.7 | -0.0103 [-0.0158, -0.0061] WIN | 186 |
| 2026-11-04 | anki | nf_l15_lr02_sm3_nup | nonfirst + leaves 15 + lr 0.02 + smoothing 3, no userprior | 0.37419 | 0.11624 | 0.7882 | 0.0080 | 10.4 | -0.0109 [-0.0171, -0.0057] WIN | 659 |
| 2026-11-04 | anki | nf_l15_lr02 | nonfirst + leaves 15 + lr 0.02 | 0.37577 | 0.11657 | 0.7868 | 0.0095 | 8.7 | -0.0093 [-0.0145, -0.0050] WIN | 494 |
| 2026-11-04 | anki | nf_l63_lr02 | nonfirst + leaves 63 + lr 0.02 | 0.37462 | 0.11604 | 0.7885 | 0.0082 | 7.1 | -0.0105 [-0.0160, -0.0062] WIN | 232 |
| 2026-11-04 | anki | l15_md300_lr02 | all, leaves 15, min_leaf 300, lr 0.02 | 0.37516 | 0.11642 | 0.7876 | 0.0088 | 4.2 | -0.0100 [-0.0153, -0.0056] WIN | 303 |
| 2026-11-04 | anki | sm3 | all, user/lexeme prior smoothing m=3 | 0.37469 | 0.11617 | 0.7885 | 0.0073 | 2.3 | -0.0104 [-0.0156, -0.0060] WIN | 91 |
| 2026-11-04 | anki | nf_sm3 | nonfirst + smoothing m=3 | 0.37603 | 0.11653 | 0.7862 | 0.0081 | 3.6 | -0.0091 [-0.0145, -0.0046] WIN | 109 |
| 2026-11-04 | anki | nf_nouserprior | nonfirst, without the userprior group | 0.37404 | 0.11615 | 0.7885 | 0.0088 | 4.3 | -0.0111 [-0.0172, -0.0059] WIN | 172 |
| 2026-11-04 | anki | nf_l15_md300_lr02 | nonfirst + leaves 15 + min_leaf 300 + lr 0.02 | 0.37650 | 0.11682 | 0.7854 | 0.0089 | 7.2 | -0.0086 [-0.0137, -0.0042] WIN | 380 |
| 2026-11-04 | anki | nf_l15_lr02_sm3 | nonfirst + leaves 15 + lr 0.02 + smoothing 3 | 0.37561 | 0.11647 | 0.7870 | 0.0096 | 9.0 | -0.0095 [-0.0147, -0.0053] WIN | 524 |
| 2026-11-04 | duolingo | all | all groups, leaves 31, lr 0.05, min_leaf 100 | 0.26218 | 0.07062 | 0.6791 | 0.0084 | 0.4 | -0.0272 [-0.0324, -0.0219] WIN | 35 |
| 2026-11-04 | duolingo | no_user | all minus group user | 0.26755 | 0.07148 | 0.6406 | 0.0068 | 0.5 | -0.0218 [-0.0274, -0.0163] WIN | 62 |
| 2026-11-04 | duolingo | no_card | all minus group card | 0.26287 | 0.07081 | 0.6759 | 0.0079 | 0.4 | -0.0265 [-0.0317, -0.0212] WIN | 41 |
| 2026-11-04 | duolingo | no_fsrs | all minus group fsrs | 0.26237 | 0.07074 | 0.6764 | 0.0076 | 0.4 | -0.0270 [-0.0324, -0.0215] WIN | 57 |
| 2026-11-04 | duolingo | no_lexeme | all minus group lexeme | 0.26333 | 0.07083 | 0.6695 | 0.0082 | 0.4 | -0.0260 [-0.0314, -0.0208] WIN | 35 |
| 2026-11-04 | duolingo | no_userprior | all minus group userprior | 0.26239 | 0.07057 | 0.6859 | 0.0115 | 0.3 | -0.0270 [-0.0322, -0.0216] WIN | 21 |
| 2026-11-04 | duolingo | nonfirst | all, train on all non-first rows (incl. same-day) | 0.25932 | 0.07013 | 0.6904 | 0.0044 | 0.7 | -0.0300 [-0.0353, -0.0248] WIN | 62 |
| 2026-11-04 | duolingo | nf_l15_lr02_sm3_nup | nonfirst + leaves 15 + lr 0.02 + smoothing 3, no userprior | 0.25777 | 0.06976 | 0.6978 | 0.0055 | 0.9 | -0.0316 [-0.0372, -0.0263] WIN | 179 |
| 2026-11-04 | duolingo | nf_l15_lr02 | nonfirst + leaves 15 + lr 0.02 | 0.25817 | 0.06986 | 0.6968 | 0.0055 | 1.0 | -0.0312 [-0.0365, -0.0262] WIN | 191 |
| 2026-11-04 | duolingo | nf_l63_lr02 | nonfirst + leaves 63 + lr 0.02 | 0.26041 | 0.07041 | 0.6877 | 0.0048 | 1.2 | -0.0290 [-0.0340, -0.0238] WIN | 131 |
| 2026-11-04 | duolingo | l15_md300_lr02 | all, leaves 15, min_leaf 300, lr 0.02 | 0.26142 | 0.07041 | 0.6867 | 0.0101 | 0.4 | -0.0279 [-0.0333, -0.0226] WIN | 76 |
| 2026-11-04 | duolingo | sm3 | all, user/lexeme prior smoothing m=3 | 0.26306 | 0.07074 | 0.6748 | 0.0115 | 0.4 | -0.0263 [-0.0316, -0.0210] WIN | 32 |
| 2026-11-04 | duolingo | nf_sm3 | nonfirst + smoothing m=3 | 0.25888 | 0.07012 | 0.6946 | 0.0044 | 0.8 | -0.0305 [-0.0356, -0.0253] WIN | 85 |
| 2026-11-04 | duolingo | nf_nouserprior | nonfirst, without the userprior group | 0.25871 | 0.06992 | 0.6926 | 0.0048 | 0.6 | -0.0307 [-0.0364, -0.0255] WIN | 48 |
| 2026-11-04 | duolingo | nf_l15_md300_lr02 | nonfirst + leaves 15 + min_leaf 300 + lr 0.02 | 0.25815 | 0.06988 | 0.6964 | 0.0045 | 1.1 | -0.0312 [-0.0365, -0.0261] WIN | 230 |
| 2026-11-04 | duolingo | nf_l15_lr02_sm3 | nonfirst + leaves 15 + lr 0.02 + smoothing 3 | 0.25819 | 0.06988 | 0.6962 | 0.0035 | 1.1 | -0.0312 [-0.0363, -0.0261] WIN | 252 |

### Task 7.3 notes, v2 (valid)

- 16 variants per dataset (table above; the frozen row was re-run once more, without logging, only to save predictions). Same validation split, scored sets and bootstrap.
- Effect of the fix: the leak was large on Duolingo (frozen 0.2473 -> 0.2578; all groups 0.2534 -> 0.2622) and on Anki (0.3614 -> 0.3742; the day-level grouping also removes legitimate same-day history there).
- Ranking by the fixed rule is unchanged, so the FROZEN CHOICE stays `nf_l15_lr02_sm3_nup` (FROZEN_KWARGS in gbdt.py). Validation: Anki 0.3742, dLL -0.0109 [-0.0171, -0.0057] WIN; Duolingo 0.2578, dLL -0.0316 [-0.0372, -0.0263] WIN. ours_v3: 0.37915 / 0.27066; FSRS-opt 0.38511 / 0.28936. On Anki the GBDT is now only ~0.0050 better than v3, so a blend (7.5) matters.
- Running user stats still matter (no_user: Anki 0.3771, Duolingo 0.2675); training on all non-first rows still helps Duolingo (0.2622 -> 0.2593).
- Invalid first-run CSVs/predictions are kept in `results/phase7_validation/invalid_leak/`. Peak RAM 1.3 GB (Anki), 1.0 GB (Duolingo).

**Anki per-row update (v3 of the user kernel, 2026-11-04):** Anki reviews are individually timed and the file order is chronological (day_offset never decreases in any of 100 users; 2.0M same-day consecutive card pairs all have elapsed_seconds >= 0 in file order), so Anki uses per-row updates (`user_group="row"`, the original behaviour); Duolingo keeps ts grouping. The Anki rows of the v2 table above (day grouping) are SUPERSEDED by the rows below; Duolingo rows are unchanged.

| date | dataset | variant | description | log_loss | brier | auc | ece | fit+predict s | dLL vs FSRS-opt [95% CI] | rounds |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-04 | anki | all | all groups, leaves 31, lr 0.05, min_leaf 100 | 0.36323 | 0.11256 | 0.8052 | 0.0075 | 2.4 | -0.0219 [-0.0317, -0.0152] WIN | 87 |
| 2026-11-04 | anki | no_user | all minus group user | 0.37705 | 0.11675 | 0.7850 | 0.0063 | 2.0 | -0.0081 [-0.0122, -0.0042] WIN | 91 |
| 2026-11-04 | anki | no_card | all minus group card | 0.36483 | 0.11317 | 0.8027 | 0.0073 | 1.8 | -0.0203 [-0.0298, -0.0138] WIN | 103 |
| 2026-11-04 | anki | no_fsrs | all minus group fsrs | 0.36454 | 0.11305 | 0.8036 | 0.0090 | 2.7 | -0.0206 [-0.0311, -0.0131] WIN | 137 |
| 2026-11-04 | anki | no_lexeme | all minus group lexeme | 0.36323 | 0.11256 | 0.8052 | 0.0075 | 2.4 | -0.0219 [-0.0317, -0.0152] WIN | 87 |
| 2026-11-04 | anki | no_userprior | all minus group userprior | 0.36159 | 0.11214 | 0.8071 | 0.0059 | 3.1 | -0.0235 [-0.0339, -0.0163] WIN | 116 |
| 2026-11-04 | anki | nonfirst | all, train on all non-first rows (incl. same-day) | 0.36303 | 0.11260 | 0.8054 | 0.0078 | 4.0 | -0.0221 [-0.0319, -0.0152] WIN | 126 |
| 2026-11-04 | anki | nf_l15_lr02_sm3_nup | nonfirst + leaves 15 + lr 0.02 + smoothing 3, no userprior | 0.36136 | 0.11219 | 0.8072 | 0.0063 | 15.2 | -0.0238 [-0.0344, -0.0164] WIN | 704 |
| 2026-11-04 | anki | nf_l15_lr02 | nonfirst + leaves 15 + lr 0.02 | 0.36293 | 0.11259 | 0.8055 | 0.0073 | 9.2 | -0.0222 [-0.0321, -0.0154] WIN | 495 |
| 2026-11-04 | anki | nf_l63_lr02 | nonfirst + leaves 63 + lr 0.02 | 0.36240 | 0.11234 | 0.8064 | 0.0071 | 7.7 | -0.0227 [-0.0327, -0.0159] WIN | 233 |
| 2026-11-04 | anki | l15_md300_lr02 | all, leaves 15, min_leaf 300, lr 0.02 | 0.36318 | 0.11259 | 0.8051 | 0.0070 | 4.9 | -0.0219 [-0.0315, -0.0153] WIN | 356 |
| 2026-11-04 | anki | sm3 | all, user/lexeme prior smoothing m=3 | 0.36208 | 0.11214 | 0.8067 | 0.0070 | 2.6 | -0.0230 [-0.0326, -0.0164] WIN | 117 |
| 2026-11-04 | anki | nf_sm3 | nonfirst + smoothing m=3 | 0.36276 | 0.11253 | 0.8058 | 0.0069 | 4.0 | -0.0224 [-0.0321, -0.0155] WIN | 126 |
| 2026-11-04 | anki | nf_nouserprior | nonfirst, without the userprior group | 0.36137 | 0.11213 | 0.8073 | 0.0072 | 4.8 | -0.0237 [-0.0340, -0.0166] WIN | 197 |
| 2026-11-04 | anki | nf_l15_md300_lr02 | nonfirst + leaves 15 + min_leaf 300 + lr 0.02 | 0.36299 | 0.11263 | 0.8056 | 0.0073 | 9.3 | -0.0221 [-0.0320, -0.0153] WIN | 492 |
| 2026-11-04 | anki | nf_l15_lr02_sm3 | nonfirst + leaves 15 + lr 0.02 + smoothing 3 | 0.36402 | 0.11298 | 0.8041 | 0.0066 | 7.8 | -0.0211 [-0.0307, -0.0143] WIN | 382 |

Notes on the Anki per-row rerun: Anki frozen `nf_l15_lr02_sm3_nup` is 0.36136 again (dLL -0.0238 [-0.0344, -0.0164] WIN), all groups 0.3632, no_user 0.3771; ranking and FROZEN CHOICE unchanged. So the day-grouping cost on Anki (0.3742) was lost legitimate same-day history, not a leak. Current valid numbers: Anki 0.36136, Duolingo 0.2578 (ts grouping). Predictions `pred_gbdt_anki.parquet` regenerated.

Known limitation (accepted, orchestrator ruling): the lexeme rate `lex_*` pools fit rows of other users, and users have different cutoff days, so a fit row of another user can be later in calendar time than a scored row t. No validation or test label is used; this is the same situation as global parameters learned from all users' train rows under the per-user split. The lexeme ablation is small (Duolingo about 0.001 log loss), so the effect on the result is negligible. `user_group="auto"` now defaults to day grouping without a `ts`; Anki is passed `user_group="row"` explicitly (run.py and the validation runner), so the numbers above are unchanged.
