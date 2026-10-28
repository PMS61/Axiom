# Matched sampler comparison

Same datasets, priors, four chains, seeds, iteration/warmup and proposal budgets. Every fit uses the unchanged R-hat <1.01 and bulk/tail ESS >=400 gate. CPU is summed process time; predictive filtering remains approximate.

| Scenario / replicate / model | Sampler | Gate | Max R-hat | Min bulk / tail ESS | Min bulk ESS / CPU s | CPU s | Log loss | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|
| anki / 0 / full | reference | FAIL | 1.5595 | 7.1 / 18.8 | 0.05 | 135.7 | 0.43469 | 0.13509 |
| anki / 0 / no_practice | reference | FAIL | 1.0959 | 39.8 / 65.5 | 0.31 | 127.6 | 0.43457 | 0.13519 |
| anki / 0 / no_hierarchy | reference | FAIL | 1.0451 | 98.2 / 228.6 | 0.81 | 121.3 | 0.50165 | 0.15985 |
| anki / 0 / no_dynamics | reference | FAIL | 1.0693 | 49.0 / 36.4 | 3.48 | 14.1 | 0.45289 | 0.14089 |
| anki / 1 / full | reference | FAIL | 1.0932 | 54.0 / 48.5 | 1.10 | 48.9 | 0.57690 | 0.16174 |
| anki / 1 / no_practice | reference | FAIL | 1.0911 | 36.2 / 51.7 | 0.86 | 42.3 | 0.58567 | 0.16120 |
| anki / 1 / no_hierarchy | reference | FAIL | 1.0353 | 77.6 / 177.5 | 1.55 | 50.0 | 0.52656 | 0.14980 |
| anki / 1 / no_dynamics | reference | FAIL | 1.0115 | 270.6 / 356.9 | 34.30 | 7.9 | 0.61648 | 0.18209 |
| anki / 0 / full | joint | FAIL | 1.3888 | 8.7 / 15.0 | 0.07 | 123.1 | 0.43405 | 0.13465 |
| anki / 0 / no_practice | joint | FAIL | 1.0959 | 39.8 / 65.5 | 0.27 | 150.2 | 0.43457 | 0.13519 |
| anki / 0 / no_hierarchy | joint | FAIL | 1.0346 | 133.7 / 241.6 | 1.04 | 128.6 | 0.50064 | 0.15972 |
| anki / 0 / no_dynamics | joint | FAIL | 1.0693 | 49.0 / 36.4 | 5.07 | 9.7 | 0.45289 | 0.14089 |
| anki / 1 / full | joint | FAIL | 1.1341 | 23.3 / 37.5 | 0.51 | 46.2 | 0.57701 | 0.16119 |
| anki / 1 / no_practice | joint | FAIL | 1.0911 | 36.2 / 51.7 | 0.85 | 42.6 | 0.58567 | 0.16120 |
| anki / 1 / no_hierarchy | joint | FAIL | 1.0422 | 89.4 / 46.7 | 2.37 | 37.7 | 0.52670 | 0.14973 |
| anki / 1 / no_dynamics | joint | FAIL | 1.0115 | 270.6 / 356.9 | 36.96 | 7.3 | 0.61648 | 0.18209 |
| anki / 0 / full | blocked | FAIL | 1.2834 | 11.7 / 26.8 | 0.09 | 136.3 | 0.43481 | 0.13478 |
| anki / 0 / no_practice | blocked | FAIL | 1.0183 | 95.5 / 284.1 | 0.80 | 119.5 | 0.43550 | 0.13543 |
| anki / 0 / no_hierarchy | blocked | FAIL | 1.0330 | 109.0 / 153.8 | 0.93 | 117.0 | 0.50081 | 0.15966 |
| anki / 0 / no_dynamics | blocked | PASS | 1.0009 | 2284.9 / 3803.3 | 75.33 | 30.3 | 0.45325 | 0.14087 |
| anki / 1 / full | blocked | FAIL | 1.0689 | 71.3 / 196.5 | 1.08 | 66.0 | 0.58239 | 0.16336 |
| anki / 1 / no_practice | blocked | FAIL | 1.0435 | 121.6 / 244.1 | 2.29 | 53.0 | 0.58808 | 0.16139 |
| anki / 1 / no_hierarchy | blocked | FAIL | 1.0662 | 66.7 / 107.5 | 1.56 | 42.8 | 0.52520 | 0.14937 |
| anki / 1 / no_dynamics | blocked | PASS | 1.0043 | 1452.3 / 2676.3 | 69.30 | 21.0 | 0.61599 | 0.18204 |
| anki / 0 / full | joint_blocked | FAIL | 1.0536 | 75.8 / 170.7 | 0.58 | 130.0 | 0.43571 | 0.13512 |
| anki / 0 / no_practice | joint_blocked | FAIL | 1.0183 | 95.5 / 284.1 | 0.73 | 130.6 | 0.43550 | 0.13543 |
| anki / 0 / no_hierarchy | joint_blocked | FAIL | 1.0263 | 152.1 / 401.8 | 1.51 | 100.5 | 0.50058 | 0.15966 |
| anki / 0 / no_dynamics | joint_blocked | PASS | 1.0009 | 2284.9 / 3803.3 | 80.69 | 28.3 | 0.45325 | 0.14087 |
| anki / 1 / full | joint_blocked | FAIL | 1.0459 | 84.7 / 246.7 | 1.52 | 55.7 | 0.58228 | 0.16071 |
| anki / 1 / no_practice | joint_blocked | FAIL | 1.0435 | 121.6 / 244.1 | 2.51 | 48.5 | 0.58808 | 0.16139 |
| anki / 1 / no_hierarchy | joint_blocked | FAIL | 1.0649 | 52.9 / 294.1 | 0.89 | 59.3 | 0.52449 | 0.14995 |
| anki / 1 / no_dynamics | joint_blocked | PASS | 1.0043 | 1452.3 / 2676.3 | 69.21 | 21.0 | 0.61599 | 0.18204 |

Source: anki. Real data have no known recovery truth.

## Posterior agreement and recovery

Parameter-comparison.csv records each posterior mean, interval, gate, MCSE, and ESS/CPU ratio against the matched reference. The difference divided by combined MCSE is a descriptive screen, not an independent-chain hypothesis test: seeds are shared and failed diagnostics make MCSE unreliable. No parameter recovery or coverage is computed for these real Anki observations. These are two selected cohorts, not a full-dataset efficacy evaluation.

## Saved reference preservation


![Matched sampler diagnostics and predictions](sampler-comparison.png)

See experiments/TASK3_ANKI_REPORT.md for the conditional audit, interpretation and limitations.
