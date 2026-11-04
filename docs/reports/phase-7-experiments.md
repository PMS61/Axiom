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
