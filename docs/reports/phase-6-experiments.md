# Phase 6 experiments log (validation only)

Every ours_v2 variant run, in order. Metrics on validation scored events; lower log loss / Brier / ECE better, higher AUC better.

| date | dataset | variant | description | log_loss | brier | auc | ece | fit+predict s | dLL vs FSRS-opt [95% CI] | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-03 | anki | v1 | candidate 1: hierarchical FSRS-style, 5 user effects, MAP, 6 epochs | 0.38634 | 0.11908 | 0.7702 | 0.0103 | 7.4 | +0.0012 [-0.0059, +0.0068] TIE |  |
| 2026-11-03 | anki | v2 | v1 with 20 epochs | 0.38024 | 0.11738 | 0.7829 | 0.0096 | 19.5 | -0.0049 [-0.0090, -0.0012] WIN |  |
| 2026-11-03 | anki | v3 | v2 with fit_steps=200 (longer card prefix in the fit) | 0.38038 | 0.11741 | 0.7827 | 0.0098 | 29.0 | -0.0047 [-0.0089, -0.0011] WIN |  |
| 2026-11-03 | anki | v4 | v1 with 10 epochs | 0.38284 | 0.11830 | 0.7776 | 0.0087 | 9.3 | -0.0023 [-0.0079, +0.0021] TIE |  |
| 2026-11-03 | duolingo | v1 | candidate 1: hierarchical FSRS-style, 5 user effects, MAP, 6 epochs | 0.27286 | 0.07257 | 0.6239 | 0.0201 | 4.2 | -0.0165 [-0.0202, -0.0130] WIN |  |
| 2026-11-03 | duolingo | v2 | v1 with 20 epochs | 0.27702 | 0.07412 | 0.6395 | 0.0274 | 9.9 | -0.0123 [-0.0186, -0.0054] WIN |  |
| 2026-11-03 | duolingo | v3 | v2 with fit_steps=200 (longer card prefix in the fit) | 0.27696 | 0.07411 | 0.6399 | 0.0274 | 10.8 | -0.0124 [-0.0187, -0.0055] WIN |  |
| 2026-11-03 | duolingo | v4 | v1 with 10 epochs | 0.27173 | 0.07254 | 0.6414 | 0.0200 | 5.1 | -0.0176 [-0.0224, -0.0129] WIN |  |
| 2026-11-03 | anki | v5 | v4 + process noise on log-stability (5-pt Gauss-Hermite marginal, candidate 2) | 0.38280 | 0.11830 | 0.7777 | 0.0084 | 15.7 | -0.0023 [-0.0080, +0.0021] TIE |  |
| 2026-11-03 | anki | v6 | v4 + per-card difficulty random effect (learned tau, prior N(0,1)) | 0.38275 | 0.11830 | 0.7778 | 0.0085 | 9.4 | -0.0024 [-0.0080, +0.0021] TIE |  |
| 2026-11-03 | anki | v7 | v4 + user-level initial-difficulty effect (6 effects) | 0.38280 | 0.11830 | 0.7777 | 0.0085 | 9.3 | -0.0023 [-0.0080, +0.0021] TIE |  |
| 2026-11-03 | duolingo | v5 | v4 + process noise on log-stability (5-pt Gauss-Hermite marginal, candidate 2) | 0.27164 | 0.07252 | 0.6415 | 0.0199 | 8.1 | -0.0177 [-0.0225, -0.0130] WIN |  |
| 2026-11-03 | duolingo | v6 | v4 + per-card difficulty random effect (learned tau, prior N(0,1)) | 0.27173 | 0.07254 | 0.6413 | 0.0200 | 5.3 | -0.0176 [-0.0225, -0.0129] WIN |  |
| 2026-11-03 | duolingo | v7 | v4 + user-level initial-difficulty effect (6 effects) | 0.27173 | 0.07254 | 0.6414 | 0.0200 | 5.1 | -0.0176 [-0.0225, -0.0129] WIN |  |
| 2026-11-03 | anki | v8 | v4 with smaller chunks (3000 cards, ~10x more optimiser steps), 3 epochs | 0.38310 | 0.11860 | 0.7768 | 0.0070 | 11.8 | -0.0020 [-0.0089, +0.0030] TIE |  |
| 2026-11-03 | anki | v9 | v8 + per-card difficulty effect + user d0 effect, z lr 0.1 | 0.38643 | 0.11942 | 0.7729 | 0.0118 | 11.3 | +0.0013 [-0.0045, +0.0064] TIE |  |
| 2026-11-03 | duolingo | v8 | v4 with smaller chunks (3000 cards, ~10x more optimiser steps), 3 epochs | 0.27165 | 0.07243 | 0.6411 | 0.0176 | 5.2 | -0.0177 [-0.0226, -0.0128] WIN |  |
| 2026-11-03 | duolingo | v9 | v8 + per-card difficulty effect + user d0 effect, z lr 0.1 | 0.27368 | 0.07318 | 0.6333 | 0.0210 | 4.3 | -0.0157 [-0.0207, -0.0107] WIN |  |

## Notes (read these with the table)

- dLL = log loss(variant) - log loss(FSRS-opt) on validation; negative = better. CI = user-cluster bootstrap, 1000 reps, seed 0 (`metrics.paired_bootstrap`). WIN only if the whole CI is below 0; TIE if it contains 0.
- The 5000-user Duolingo sample is the DEVELOPMENT set (orchestrator decision); the final test will use all Duolingo data. Anki dev set: 100 users, max_cards 1000.
- FSRS baselines (validation log loss): Anki default 0.39853, opt 0.38511; Duolingo default 0.28977, opt 0.28936.
- The first rows of this file were regenerated after the verifier asked for CIs; the point estimates are identical to the first pass (deterministic seeds). Rows are appended per invocation, so v1-v4 appear first for each dataset, then v5-v9.
- v1-v4: candidate 1, differing only in optimiser length / fit prefix. v5: candidate 2 (process noise). v6, v7, v9: extra random effects (card difficulty, user initial difficulty). v8: more, smaller optimiser steps. v9 is the only variant clearly worse than v4. v6, v7 add nothing measurable: with ~10-20 optimiser steps the extra effects barely move from their prior.
- Fixed selection rule (set before v5-v9): lowest mean validation log loss over the two datasets among variants whose Duolingo CI excludes 0; ties broken by Anki CI. Means: v5 0.32722, v6 0.32724, v7 0.32727, v4 0.32729, v8 0.32737, v2 0.32863, v9 0.33006. FROZEN CHOICE: v5 (epochs 10, process noise, 5 user effects). The differences among v4-v8 are about 1e-4, far inside the noise; v5 is chosen only because the rule is mechanical.
- HONEST VERDICT on validation: Duolingo is a clear WIN (-0.0177 [-0.0225, -0.0130]). Anki for the frozen v5 is a TIE (-0.0023 [-0.0080, +0.0021]); no Anki win may be claimed for it. Only v2 and v3 (20 epochs) had an Anki CI excluding 0 (v2 -0.0049 [-0.0090, -0.0012]) but they have a worse Duolingo mean and were not selected by the rule.
- Not tried (budget of 5 extra variants used): elapsed-seconds for same-day reviews, hierarchical grade-dependent initial stability priors, L2 prior towards FSRS defaults (`prior_w`).

## Revised selection rule (orchestrator decision, supersedes the FROZEN CHOICE above)

Rule: "prefer variants that WIN on both datasets on validation (CI excludes 0); among those, lowest mean val log loss."
This rule change is POST HOC: it was made AFTER seeing the validation CIs (not the test set; no test event was used). Qualifying variants: v2 (Anki -0.0049 [-0.0090, -0.0012], Duolingo -0.0123 [-0.0186, -0.0054]) and v3 (essentially the same as v2; mean 0.32867 vs 0.32863). Lowest mean log loss: v2.
FROZEN CHOICE: v2 (epochs 20, 5 user effects, no process noise), `FROZEN_KWARGS = dict(epochs=20)` in `ours_v2.py`. v5 is no longer the frozen model. Caveat: v2's Duolingo log loss (0.2770) is worse than v4/v5 (0.2717); the choice trades Duolingo margin for a significant Anki win.
