# Elo-band experts (validation users 8001-8100, train 2001-2300). Difference vs the global GBDT (+online vs global+online), weighted log loss, 99% CI.
Global: 0.31360; global+online: 0.30945; RWKV-Instant 0.29660.

| Variant | FSRS in experts: all | none | mixed (none in upper bands) |
|---|---|---|---|
| 3 bands, scratch, hard | +0.0004 / +0.0008 | +0.0022 / +0.0013 | +0.0005 / +0.0008 |
| 3 bands, scratch, soft | -0.0002 / -0.0001 | +0.0017 / +0.0005 | -0.0001 / -0.0000 |
| 3 bands, warm, hard | +0.0000 / +0.0000 | -0.0000 / -0.0000 | +0.0000 / +0.0000 |
| 3 bands, warm, soft | +0.0000 / -0.0000 | +0.0000 / -0.0000 | +0.0000 / -0.0000 |
| 5 bands, scratch, hard | +0.0008 / +0.0013 | +0.0024 / +0.0017 | +0.0010 / +0.0014 |
| 5 bands, scratch, soft | +0.0001 / +0.0004 | +0.0021 / +0.0009 | +0.0004 / +0.0005 |
| 5 bands, warm, hard | -0.0000 / -0.0000 | +0.0000 / -0.0000 | -0.0000 / -0.0000 |
| 5 bands, warm, soft | -0.0000 / -0.0000 | -0.0000 / -0.0000 | -0.0000 / -0.0000 |
(each cell: GBDT-only diff / with-online diff; negative = expert better)

Conclusions: no variant helps by more than 0.00005. Dropping FSRS from the experts makes the from-scratch experts WORSE
(+0.002), so the hypothesis "global FSRS adds noise to local bands" is not supported. Soft (overlapping) bands beat hard ones.
