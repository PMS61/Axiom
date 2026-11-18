# GBDT-Curve validation (users 8001-8100, train 2001-2300), Kaggle notebook52e4915672
RWKV-Curve: LL_w 0.31529 (LL_u 0.30180), AUC_w 0.756. RWKV-Instant: LL_w 0.29660.

| Variant | best_iter | Curve LL_w | Curve+online LL_w | Curve+online AUC_w | Curve+online minus RWKV-Curve (99% CI) |
|---|---|---|---|---|---|
| base (v8 run, monotone basic, lr 0.05) | 137 | 0.33210 | 0.33123 | 0.704 | +0.0159 [+0.0126, +0.0212] |
| adv (monotone advanced) | 37 | 0.33913 | 0.33556 | 0.680 | +0.0203 [+0.0161, +0.0258] |
| nomono (no monotone constraint) | 64 | 0.33404 | 0.33308 | 0.694 | +0.0178 [+0.0140, +0.0233] |
| adv_lr02 (advanced, lr 0.02) | 89 | 0.33959 | 0.33555 | 0.680 | +0.0203 [+0.0161, +0.0258] |

Conclusion: every variant loses to RWKV-Curve; removing the monotone constraint does not help and also stops early,
so the constraint is not the bottleneck. The curve feature set (state frozen at the previous review) generalises poorly.
NOT added to slides / black book / guide (user instruction: ask first).
