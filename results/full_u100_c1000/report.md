# Results: full_u100_c1000

## Data

- Users: 100, cards: 99074, events: 768658 (606364 train, 162294 test)
- Scored events: 95400, recall rate 0.831
- FSRS-6 optimised fallback users (defaults kept): n/a
- Users with zero train rows: 0
- Git commit: 0b6259e5f5bf9fb4cf82017cd01a3b01c9fcfed6 (dirty: True)
- Config: n_users=100, max_cards=1000, seed=0, iters=2000, burn=1000

Per-model timings (seconds):

| Model | Fit | Predict |
|---|---|---|
| FSRS-6 (default) | n/a | n/a |
| FSRS-6 (optimised) | n/a | n/a |
| HLR (Settles & Meeder) | n/a | n/a |
| Ours (dynamic, learned λ) | 27529.624 | 1266.342 |
| Ours (+κ) | 26129.754 | 1339.690 |

n/a: model reused from an interrupted earlier run via --resume; its timing was not recorded.

## Overall

| Model | n | Recall | Log loss | Brier | AUC | ECE | RMSE (bins) | Mean bias |
|---|---|---|---|---|---|---|---|---|
| FSRS-6 (default) | 95400 | 0.831 | 0.396 | 0.121 | 0.745 | 0.012 | 0.017 | 0.011 |
| FSRS-6 (optimised) | 95400 | 0.831 | **0.379** | **0.117** | **0.777** | **0.009** | **0.014** | **-0.002** |
| HLR (Settles & Meeder) | 95400 | 0.831 | 0.680 | 0.170 | 0.555 | 0.146 | 0.176 | 0.060 |
| Ours (dynamic, learned λ) | 95400 | 0.831 | 0.422 | 0.131 | 0.692 | 0.025 | 0.032 | 0.011 |
| Ours (+κ) | 95400 | 0.831 | 0.422 | 0.131 | 0.693 | 0.026 | 0.032 | 0.011 |

Best value per column in bold. Mean bias is closest to zero.

## Gap bins (days since the card's previous review)

### 1d (n = 16983)

| Model | Recall | Log loss | Brier | AUC | ECE | Mean bias |
|---|---|---|---|---|---|---|
| FSRS-6 (default) | 0.825 | 0.409 | 0.126 | 0.748 | 0.020 | 0.017 |
| FSRS-6 (optimised) | 0.825 | **0.396** | **0.122** | **0.769** | **0.017** | 0.010 |
| HLR (Settles & Meeder) | 0.825 | 0.942 | 0.172 | 0.658 | 0.170 | 0.170 |
| Ours (dynamic, learned λ) | 0.825 | 0.423 | 0.133 | 0.708 | 0.024 | -0.006 |
| Ours (+κ) | 0.825 | 0.423 | 0.133 | 0.708 | 0.025 | **-0.005** |

### 2-3d (n = 16080)

| Model | Recall | Log loss | Brier | AUC | ECE | Mean bias |
|---|---|---|---|---|---|---|
| FSRS-6 (default) | 0.835 | 0.402 | 0.124 | 0.730 | 0.021 | 0.016 |
| FSRS-6 (optimised) | 0.835 | **0.389** | **0.121** | **0.755** | **0.013** | **-0.006** |
| HLR (Settles & Meeder) | 0.835 | 0.734 | 0.160 | 0.637 | 0.153 | 0.153 |
| Ours (dynamic, learned λ) | 0.835 | 0.426 | 0.134 | 0.676 | 0.033 | -0.027 |
| Ours (+κ) | 0.835 | 0.426 | 0.134 | 0.677 | 0.035 | -0.027 |

### 4-14d (n = 29204)

| Model | Recall | Log loss | Brier | AUC | ECE | Mean bias |
|---|---|---|---|---|---|---|
| FSRS-6 (default) | 0.849 | 0.369 | 0.110 | 0.737 | **0.017** | **0.006** |
| FSRS-6 (optimised) | 0.849 | **0.361** | **0.109** | **0.760** | 0.017 | -0.015 |
| HLR (Settles & Meeder) | 0.849 | 0.555 | 0.141 | 0.616 | 0.123 | 0.123 |
| Ours (dynamic, learned λ) | 0.849 | 0.398 | 0.121 | 0.680 | 0.020 | -0.020 |
| Ours (+κ) | 0.849 | 0.398 | 0.121 | 0.681 | 0.020 | -0.020 |

### 15-60d (n = 15626)

| Model | Recall | Log loss | Brier | AUC | ECE | Mean bias |
|---|---|---|---|---|---|---|
| FSRS-6 (default) | 0.832 | 0.401 | 0.121 | 0.725 | 0.026 | 0.021 |
| FSRS-6 (optimised) | 0.832 | **0.374** | **0.114** | **0.781** | **0.009** | **0.008** |
| HLR (Settles & Meeder) | 0.832 | 0.473 | 0.143 | 0.599 | 0.076 | 0.075 |
| Ours (dynamic, learned λ) | 0.832 | 0.419 | 0.130 | 0.703 | 0.040 | 0.036 |
| Ours (+κ) | 0.832 | 0.419 | 0.130 | 0.705 | 0.040 | 0.036 |

### 61d+ (n = 17507)

| Model | Recall | Log loss | Brier | AUC | ECE | Mean bias |
|---|---|---|---|---|---|---|
| FSRS-6 (default) | 0.802 | 0.418 | 0.130 | 0.771 | **0.011** | **0.002** |
| FSRS-6 (optimised) | 0.802 | **0.389** | **0.124** | **0.817** | 0.027 | 0.003 |
| HLR (Settles & Meeder) | 0.802 | 0.771 | 0.248 | 0.628 | 0.249 | -0.249 |
| Ours (dynamic, learned λ) | 0.802 | 0.460 | 0.145 | 0.763 | 0.090 | 0.090 |
| Ours (+κ) | 0.802 | 0.460 | 0.145 | 0.764 | 0.090 | 0.090 |

![Log loss by gap bin](gapbins_logloss.png)

## Paired differences (model minus reference), 95% user-cluster bootstrap

Bootstrap: 1000 user-cluster resamples with replacement, seed 0, 95% percentile CI, diff = model minus reference.

| Comparison | Metric | Diff | 95% CI |
|---|---|---|---|
| FSRS-6 (default) - FSRS-6 (optimised) | log_loss | 0.017 | [0.008, 0.025] * |
| FSRS-6 (default) - FSRS-6 (optimised) | brier | 0.004 | [0.002, 0.006] * |
| FSRS-6 (default) - FSRS-6 (optimised) | auc | -0.032 | [-0.047, -0.016] * |
| FSRS-6 (default) - FSRS-6 (optimised) | ece | 0.003 | [-0.010, 0.019] |
| HLR (Settles & Meeder) - FSRS-6 (optimised) | log_loss | 0.301 | [0.246, 0.361] * |
| HLR (Settles & Meeder) - FSRS-6 (optimised) | brier | 0.053 | [0.040, 0.067] * |
| HLR (Settles & Meeder) - FSRS-6 (optimised) | auc | -0.221 | [-0.277, -0.173] * |
| HLR (Settles & Meeder) - FSRS-6 (optimised) | ece | 0.137 | [0.113, 0.156] * |
| Ours (dynamic, learned λ) - FSRS-6 (optimised) | log_loss | 0.043 | [0.033, 0.055] * |
| Ours (dynamic, learned λ) - FSRS-6 (optimised) | brier | 0.014 | [0.011, 0.019] * |
| Ours (dynamic, learned λ) - FSRS-6 (optimised) | auc | -0.084 | [-0.110, -0.064] * |
| Ours (dynamic, learned λ) - FSRS-6 (optimised) | ece | 0.016 | [0.001, 0.027] * |
| Ours (+κ) - FSRS-6 (optimised) | log_loss | 0.043 | [0.033, 0.055] * |
| Ours (+κ) - FSRS-6 (optimised) | brier | 0.014 | [0.011, 0.019] * |
| Ours (+κ) - FSRS-6 (optimised) | auc | -0.084 | [-0.108, -0.064] * |
| Ours (+κ) - FSRS-6 (optimised) | ece | 0.016 | [0.002, 0.027] * |
| FSRS-6 (default) - Ours (dynamic, learned λ) | log_loss | -0.026 | [-0.042, -0.013] * |
| FSRS-6 (default) - Ours (dynamic, learned λ) | brier | -0.011 | [-0.015, -0.007] * |
| FSRS-6 (default) - Ours (dynamic, learned λ) | auc | 0.053 | [0.023, 0.087] * |
| FSRS-6 (default) - Ours (dynamic, learned λ) | ece | -0.013 | [-0.025, 0.002] |
| FSRS-6 (optimised) - Ours (dynamic, learned λ) | log_loss | -0.043 | [-0.055, -0.033] * |
| FSRS-6 (optimised) - Ours (dynamic, learned λ) | brier | -0.014 | [-0.019, -0.011] * |
| FSRS-6 (optimised) - Ours (dynamic, learned λ) | auc | 0.084 | [0.064, 0.110] * |
| FSRS-6 (optimised) - Ours (dynamic, learned λ) | ece | -0.016 | [-0.027, -0.001] * |
| HLR (Settles & Meeder) - Ours (dynamic, learned λ) | log_loss | 0.258 | [0.204, 0.314] * |
| HLR (Settles & Meeder) - Ours (dynamic, learned λ) | brier | 0.038 | [0.026, 0.052] * |
| HLR (Settles & Meeder) - Ours (dynamic, learned λ) | auc | -0.137 | [-0.203, -0.075] * |
| HLR (Settles & Meeder) - Ours (dynamic, learned λ) | ece | 0.121 | [0.098, 0.142] * |
| Ours (+κ) - Ours (dynamic, learned λ) | log_loss | -0.000 | [-0.001, 0.000] |
| Ours (+κ) - Ours (dynamic, learned λ) | brier | -0.000 | [-0.000, 0.000] |
| Ours (+κ) - Ours (dynamic, learned λ) | auc | 0.001 | [-0.000, 0.002] |
| Ours (+κ) - Ours (dynamic, learned λ) | ece | 0.000 | [-0.001, 0.001] |

* CI excludes 0.

## Per-user wins on log loss

| Comparison | Users | Users with lower log loss | Share |
|---|---|---|---|
| FSRS-6 (default) vs FSRS-6 (optimised) | 100 | 28 | 0.280 |
| HLR (Settles & Meeder) vs FSRS-6 (optimised) | 100 | 7 | 0.070 |
| Ours (dynamic, learned λ) vs FSRS-6 (optimised) | 100 | 22 | 0.220 |
| Ours (+κ) vs FSRS-6 (optimised) | 100 | 25 | 0.250 |
| FSRS-6 (default) vs Ours (dynamic, learned λ) | 100 | 68 | 0.680 |
| FSRS-6 (optimised) vs Ours (dynamic, learned λ) | 100 | 78 | 0.780 |
| HLR (Settles & Meeder) vs Ours (dynamic, learned λ) | 100 | 14 | 0.140 |
| Ours (+κ) vs Ours (dynamic, learned λ) | 100 | 52 | 0.520 |

Share = users with lower log loss for the model / users with a finite log loss for both.

## MCMC diagnostics

R-hat here is split-R-hat computed on a single chain (no between-chain comparison). ESS is the effective sample size of the kept draws.

### Ours (dynamic, learned λ)

Kept draws: 250; lambda Metropolis-Hastings acceptance: mean after burn-in 0.240, mean over all sweeps 0.236. lambda acceptance: mean over all Gibbs sweeps (mh_acc_rate) and over sweeps after burn-in (mh_acc_rate_mean_postburn).

| Param | Mean | 95% interval | ESS | R-hat |
|---|---|---|---|---|
| lam | 1.871 | [1.312, 2.646] | 3 | 2.439 |
| sigma2 | 0.204 | [0.189, 0.231] | 3 | 2.071 |
| b0 | 2.332 | [2.314, 2.343] | 4 | 1.854 |
| g0 | 0.066 | [-0.133, 0.270] | 81 | 1.001 |
| tau2_alpha | 2.234 | [1.673, 2.879] | 250 | 0.996 |
| tau2_beta | 0.499 | [0.482, 0.513] | 5 | 1.879 |
| tau2_gamma | 0.993 | [0.745, 1.332] | 143 | 1.064 |

### Ours (+κ)

Kept draws: 250; lambda Metropolis-Hastings acceptance: mean after burn-in 0.271, mean over all sweeps 0.259. lambda acceptance: mean over all Gibbs sweeps (mh_acc_rate) and over sweeps after burn-in (mh_acc_rate_mean_postburn). kappa acceptance is not stored by the sampler, so it is not reported.

| Param | Mean | 95% interval | ESS | R-hat |
|---|---|---|---|---|
| lam | 1.455 | [0.898, 2.228] | 3 | 2.376 |
| kappa | -0.160 | [-0.211, -0.076] | 4 | 1.939 |
| sigma2 | 0.257 | [0.245, 0.277] | 9 | 1.161 |
| b0 | 2.317 | [2.296, 2.331] | 4 | 1.283 |
| g0 | 0.078 | [-0.104, 0.275] | 234 | 0.998 |
| tau2_alpha | 2.250 | [1.687, 2.899] | 250 | 1.005 |
| tau2_beta | 0.480 | [0.465, 0.492] | 9 | 1.257 |
| tau2_gamma | 1.071 | [0.822, 1.366] | 202 | 1.009 |

## Figures

![Reliability](reliability.png)
