# Matched sampler comparison

Same datasets, priors, four chains, seeds, iteration/warmup and proposal budgets. Every fit uses the unchanged R-hat <1.01 and bulk/tail ESS >=400 gate. CPU is summed process time; predictive filtering remains approximate.

| Scenario / replicate / model | Sampler | Gate | Max R-hat | Min bulk / tail ESS | Min bulk ESS / CPU s | CPU s | Log loss | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|
| full / 0 / full | reference | FAIL | 1.0401 | 155.2 / 280.4 | 1.83 | 85.0 | 0.61457 | 0.21292 |
| full / 0 / no_dynamics | reference | FAIL | 1.0049 | 385.7 / 589.7 | 28.71 | 13.4 | 0.60420 | 0.20735 |
| full / 1 / full | reference | FAIL | 1.0336 | 152.8 / 238.2 | 1.61 | 94.8 | 0.63674 | 0.22313 |
| full / 1 / no_dynamics | reference | PASS | 1.0059 | 401.7 / 775.8 | 33.21 | 12.1 | 0.64425 | 0.22688 |
| short_gaps / 0 / full | reference | FAIL | 1.3058 | 12.0 / 18.7 | 0.12 | 98.8 | 0.62012 | 0.21639 |
| full / 0 / full | joint | FAIL | 1.0201 | 170.9 / 302.9 | 2.03 | 84.2 | 0.61530 | 0.21328 |
| full / 0 / no_dynamics | joint | FAIL | 1.0049 | 385.7 / 589.7 | 24.58 | 15.7 | 0.60420 | 0.20735 |
| full / 1 / full | joint | FAIL | 1.0276 | 187.7 / 285.9 | 1.96 | 95.8 | 0.63724 | 0.22336 |
| full / 1 / no_dynamics | joint | PASS | 1.0059 | 401.7 / 775.8 | 16.92 | 23.7 | 0.64425 | 0.22688 |
| short_gaps / 0 / full | joint | FAIL | 1.0968 | 37.1 / 40.7 | 0.34 | 110.7 | 0.61948 | 0.21606 |
| full / 0 / full | blocked | FAIL | 1.0577 | 80.3 / 335.9 | 0.74 | 108.6 | 0.61454 | 0.21297 |
| full / 0 / no_dynamics | blocked | PASS | 1.0008 | 11028.6 / 11866.9 | 456.30 | 24.2 | 0.60434 | 0.20745 |
| full / 1 / full | blocked | FAIL | 1.0206 | 180.8 / 372.6 | 1.70 | 106.5 | 0.63637 | 0.22300 |
| full / 1 / no_dynamics | blocked | PASS | 1.0005 | 10353.8 / 12217.1 | 506.83 | 20.4 | 0.64381 | 0.22670 |
| short_gaps / 0 / full | blocked | FAIL | 1.0548 | 92.1 / 216.8 | 1.05 | 88.1 | 0.61954 | 0.21617 |
| full / 0 / full | joint_blocked | FAIL | 1.0225 | 156.6 / 329.2 | 1.68 | 93.3 | 0.61583 | 0.21347 |
| full / 0 / no_dynamics | joint_blocked | PASS | 1.0008 | 11028.6 / 11866.9 | 542.13 | 20.3 | 0.60434 | 0.20745 |
| full / 1 / full | joint_blocked | FAIL | 1.0169 | 162.7 / 397.0 | 1.91 | 85.0 | 0.63803 | 0.22368 |
| full / 1 / no_dynamics | joint_blocked | PASS | 1.0005 | 10353.8 / 12217.1 | 602.70 | 17.2 | 0.64381 | 0.22670 |
| short_gaps / 0 / full | joint_blocked | FAIL | 1.0365 | 107.0 / 112.2 | 1.07 | 100.4 | 0.62053 | 0.21663 |

## Posterior agreement and recovery

Parameter-comparison.csv records each posterior mean, interval, gate, MCSE, and ESS/CPU ratio against the matched reference. The difference divided by combined MCSE is a descriptive screen, not an independent-chain hypothesis test: seeds are shared and failed diagnostics make MCSE unreliable. Recovery RMSE mixes parameter scales and is descriptive only. Inspect parameter-level errors/intervals in each source recovery.csv. Few datasets cannot establish calibrated coverage. Static fits on full-generating data are deliberately not counted as matched recovery.

## Saved reference preservation

- full: exact retained draw equality with saved longer-chain study: True.
- no_dynamics: exact retained draw equality with saved longer-chain study: True.

![Matched sampler diagnostics and predictions](sampler-comparison.png)

See experiments/TASK3_SAMPLER_REPORT.md for the conditional audit, interpretation and limitations.
