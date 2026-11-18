# Black book change log

## 2026-11-19: Experiment 6 (extended GBDT on srs-benchmark)

Base: teammate's version `bb6ee05` (new title, IT degree, Sem VII, HOD and Director names, certificate table removed).
All of those front-matter changes are kept unchanged.

### Fixes
- **`references.bib` added to git.** It was never committed, so the book could not be built from the repository.
- **`figures/logo/vjti_logo.png` added to git.** Same reason (the title page needs it).
- `build.sh` added (`latexmk` wrapper).

### Front matter
- **Abstract rewritten** to Vedaang's brief: two paragraphs, plain language, no heavy keywords. It covers why the problem was
  chosen, which models are documented (FSRS, HLR, the Bayesian model and its MAP variants, the tree model, the published
  neural networks), and an honest view of where each is best. Keywords simplified.
- Abbreviations: added EWMA, RNN, RWKV.

### Chapter 1, Introduction
- Seventh objective: extend the tree model and test it once, after pre-registration, on srs-benchmark (Experiment 6).
- New contribution: "a near-leading, low-cost predictor".
- Summary paragraph: Experiment 6 result (weighted log loss 0.2999 vs 0.2828 for RWKV-Instant and 0.3383 for FSRS-6).

### Chapter 2, Literature
- New paragraph on sequence models and online learner ratings: RWKV and Elo in education.
- New references: `peng2023rwkv` (RWKV, EMNLP Findings 2023) and `pelanek2016elo` (Elo in adaptive educational systems,
  Computers & Education 2016).

### Chapter 3, Dataset
- New subsection **"The srs-benchmark protocol"**:
  - the anki-revlogs-10k collection (9,999 users, about 727M reviews);
  - the review fields and side tables;
  - how grades become binary outcomes, and which reviews are scored;
  - the time-ordered 5-fold split, the metrics and the published models;
  - **Instant vs Curve** models (our GBDT is an Instant model).

### Chapter 5, Algorithms
- New section **"The Extended GBDT and the Online Layer (Experiment 6)"**:
  - a table of the 78 feature groups, and the final model's settings and size (1,047 trees, about 196,000 stored numbers);
  - the online layer equations (6 weights per learner, an Elo-style update, η = 0.003, λ = 0.001);
  - the Elo rating equations used for analysis, and their cold-start limitation.

### Chapter 6, Results: new section "Experiment 6: Closing the Gap on srs-benchmark"
- **Protocol and honesty rules:** train on users 2001–2300, choose settings on 8001–8050, report on 8051–8100,
  pre-registration, one look at test users 1–1000.
- **Development table:** the gap to RWKV-Instant fell from +0.0218 to +0.0133 on validation.
- **Final test table** (log loss and AUC, weighted and unweighted, against all published models): the final model has
  0.2999 / 0.795 weighted and is 2nd of all published models.
- **Paired intervals:**
  - vs GBDT alone: −0.0052;
  - vs RWKV-Instant: +0.0171;
  - vs RWKV-Curve: −0.0143;
  - vs FSRS-6: −0.0384.
- **Where the gain comes from:** the FSRS ablation, feature importance, the Elo-band analysis (FSRS +9.2 / −5.7 points of
  bias), and where the remaining gap sits.
- **What did not help:** per-user boosting, Kalman filter, attention model, residual GRU, user-type experts, MoE gates,
  Elo features and Elo-band experts.
- **Size and speed:** about 196k numbers vs 2.76M parameters; 0.09 ms vs about 96 ms per prediction.
- The chapter introduction now says six experiments, and 4 new items were added to the Summary of Findings.

### Chapter 7, System
- New section "srs-benchmark Modules (Experiment 6)", covering the modules under `fsrs_benchmark/srsb/`, their guards and
  their leak tests.

### Chapter 8, Conclusion
- New subsection "What Experiment 6 adds".
- New paragraph "Limits of Experiment 6":
  - it is an Instant model;
  - the online settings were chosen on 100 users;
  - Elo cold start;
  - Anki-only data;
  - the speed-test scope.
- The conclusion and future scope were updated: a curve version, more users, and sequence models at scale.

### Appendix
- New section with the reproduction commands for Experiment 6 (scoring the frozen GBDT, the online layer, the speed test).

### Not included (still running when written)
- The GBDT-Curve (Curve-type version of the model) vs RWKV-Curve.
- The second test set (users 1001–2000).
