# Dynamics investigation: measured results

Four chains per fit; 8,000 iterations / 4,000 warmup. Every monitored scalar must pass rank R-hat <1.01 and bulk/tail ESS >=400. Undefined fails. Fixed controls change the model. Real cohorts retain recorded within-day order. An alternative within-day permutation removes all timing discrepancies but does not authenticate outcome order. Synthetic fits condition on the saved nested gap schedules; these are two fixed-truth replicates, not SBC.

| Dataset / cohort | Control | Kernel | Max R-hat | Min bulk / tail ESS | ESS / CPU s | CPU s | Log loss | Gate |
|---|---|---|---:|---:|---:|---:|---:|---|
| real / 1 | static | centered | 1.0009 | 2284.9 / 3803.3 | 137.04 | 16.7 | 0.45325 | PASS |
| real / 1 | fixed_dynamics | centered | 1.0051 | 539.3 / 1026.7 | 5.57 | 96.8 | 0.48408 | PASS |
| real / 1 | free_diffusion | centered | 1.0177 | 302.2 / 770.8 | 2.12 | 142.8 | 0.44557 | FAIL |
| real / 1 | free_diffusion | interwoven | 1.0071 | 653.0 / 1276.0 | 2.30 | 284.1 | 0.44470 | PASS |
| real / 1 | free_decay_diffusion | centered | 1.0190 | 95.5 / 284.1 | 0.71 | 134.4 | 0.43550 | FAIL |
| real / 1 | free_decay_diffusion | interwoven | 1.0199 | 188.6 / 241.3 | 0.67 | 281.7 | 0.43428 | FAIL |
| real / 1 | full | centered | 1.0536 | 72.4 / 170.7 | 0.49 | 146.4 | 0.43571 | FAIL |
| real / 1 | full | interwoven | 1.0380 | 139.0 / 176.1 | 0.42 | 328.2 | 0.43414 | FAIL |
| nested_synthetic / 1 | static | centered | 1.0019 | 3477.9 / 6821.3 | 111.02 | 31.3 | 0.61726 | PASS |
| nested_synthetic / 1 | fixed_dynamics | centered | 1.0047 | 1201.6 / 2149.9 | 12.45 | 96.5 | 0.61656 | PASS |
| nested_synthetic / 1 | free_diffusion | centered | 1.0202 | 150.9 / 189.4 | 1.21 | 124.5 | 0.61571 | FAIL |
| nested_synthetic / 1 | free_diffusion | interwoven | 1.0040 | 894.8 / 1425.8 | 2.25 | 397.8 | 0.61361 | PASS |
| nested_synthetic / 1 | free_decay_diffusion | centered | 1.0231 | 124.3 / 251.8 | 0.90 | 137.5 | 0.61306 | FAIL |
| nested_synthetic / 1 | free_decay_diffusion | interwoven | 1.0073 | 456.9 / 1257.5 | 1.37 | 332.5 | 0.61108 | PASS |
| nested_synthetic / 1 | full | centered | 1.1212 | 23.3 / 172.4 | 0.18 | 133.2 | 0.60510 | FAIL |
| nested_synthetic / 1 | full | interwoven | 1.0158 | 307.7 / 628.9 | 1.00 | 308.2 | 0.60485 | FAIL |
| real / 2 | static | centered | 1.0043 | 1452.3 / 2676.3 | 68.80 | 21.1 | 0.61599 | PASS |
| real / 2 | fixed_dynamics | centered | 1.0039 | 705.3 / 1013.7 | 17.16 | 41.1 | 0.58968 | PASS |
| real / 2 | free_diffusion | centered | 1.0224 | 246.5 / 400.0 | 5.18 | 47.6 | 0.58862 | FAIL |
| real / 2 | free_diffusion | interwoven | 1.0035 | 558.7 / 848.5 | 6.02 | 92.8 | 0.58871 | PASS |
| real / 2 | free_decay_diffusion | centered | 1.0435 | 121.6 / 244.1 | 2.24 | 54.4 | 0.58808 | FAIL |
| real / 2 | free_decay_diffusion | interwoven | 1.0118 | 338.2 / 718.4 | 2.78 | 121.6 | 0.58754 | FAIL |
| real / 2 | full | centered | 1.0459 | 84.7 / 246.7 | 1.37 | 62.0 | 0.58228 | FAIL |
| real / 2 | full | interwoven | 1.0208 | 329.3 / 627.6 | 2.92 | 112.6 | 0.57640 | FAIL |
| nested_synthetic / 2 | static | centered | 1.0022 | 2075.0 / 4475.4 | 102.11 | 20.3 | 0.63519 | PASS |
| nested_synthetic / 2 | fixed_dynamics | centered | 1.0078 | 1082.3 / 2319.7 | 26.72 | 40.5 | 0.58912 | PASS |
| nested_synthetic / 2 | free_diffusion | centered | 1.0258 | 107.2 / 188.9 | 2.32 | 46.1 | 0.58790 | FAIL |
| nested_synthetic / 2 | free_diffusion | interwoven | 1.0047 | 1028.6 / 918.5 | 10.82 | 95.1 | 0.58515 | PASS |
| nested_synthetic / 2 | free_decay_diffusion | centered | 1.0598 | 64.2 / 64.4 | 1.21 | 53.1 | 0.57069 | FAIL |
| nested_synthetic / 2 | free_decay_diffusion | interwoven | 1.0033 | 590.2 / 1492.0 | 5.68 | 103.9 | 0.57191 | PASS |
| nested_synthetic / 2 | full | centered | 1.0375 | 66.1 / 107.9 | 0.97 | 67.8 | 0.58848 | FAIL |
| nested_synthetic / 2 | full | interwoven | 1.0101 | 491.6 / 902.9 | 4.37 | 112.6 | 0.58047 | FAIL |

## Matched computational comparisons

Ratio = interwoven minimum bulk ESS/CPU divided by centered minimum bulk ESS/CPU. The limiting parameter may differ. Parameter-comparison.csv retains every scalar ratio and MCSE difference screen. A failed whole fit prevents inferential interpretation; an MCSE screen is descriptive and does not prove target equality. Exact transformation tests independently check the target.

| Dataset / cohort / control | Centered / interwoven gate | ESS / CPU ratio | Delta log loss |
|---|---|---:|---:|
| real-0-free_diffusion | False / True | 1.09 | -0.00087 |
| real-0-free_decay_diffusion | False / False | 0.94 | -0.00123 |
| real-0-full | False / False | 0.86 | -0.00157 |
| nested_synthetic-0-free_diffusion | False / True | 1.86 | -0.00210 |
| nested_synthetic-0-free_decay_diffusion | False / True | 1.52 | -0.00197 |
| nested_synthetic-0-full | False / False | 5.70 | -0.00026 |
| real-1-free_diffusion | False / True | 1.16 | +0.00009 |
| real-1-free_decay_diffusion | False / False | 1.24 | -0.00054 |
| real-1-full | False / False | 2.14 | -0.00588 |
| nested_synthetic-1-free_diffusion | False / True | 4.65 | -0.00275 |
| nested_synthetic-1-free_decay_diffusion | False / True | 4.70 | +0.00123 |
| nested_synthetic-1-full | False / False | 4.48 | -0.00801 |

## Recovery

Recovery.csv contains raw and combined parameters, all generating hierarchy variances, and terminal training states. Counts below include failed parameters and dependent scalar summaries. They are not a calibrated coverage rate.

| Fit | Truth-containing intervals / quantities | Quantities passing gate | Whole fit |
|---|---:|---:|---|
| nested_synthetic-0-full-centered | 136 / 142 | 115 | False |
| nested_synthetic-0-full-interwoven | 135 / 142 | 135 | False |
| nested_synthetic-1-full-centered | 141 / 142 | 113 | False |
| nested_synthetic-1-full-interwoven | 141 / 142 | 141 | False |

## Regression and limits

All 24 saved parameter arrays from the two centered real full fits match the preceding joint_blocked arrays exactly: True. These repeats are target/regression controls, not additional independent real datasets. CPU varies between executions; matched ratios are checked to use the same execution segment. New user-level and within-user card summaries supplement the original diagnostic monitor set.

Prediction uses the inherited approximate prequential filter and at most 200 retained draws per chain. Small score differences under failed diagnostics cannot establish predictive superiority or component benefit. No test outcomes tune the fixed controls or proposals. Added innovation proposals increase CPU, so iteration counts alone are not efficiency evidence. No stronger priors, parameter-support changes, dropped chains, threshold relaxation or full-dataset inference are used.

![Dynamics ladder](dynamics-ladder.png)
