# Research paper plan (started 2026-11-15)

## Intent

Turn the FSRS benchmark work (`/home/amal/fyp/Axiom/fsrs_benchmark/`, results in
`docs/EVALUATION.md` and `docs/reports/FINAL.md`) into a research paper. The professor wants
to publish it. The target is **EDM 2027** (Educational Data Mining). Expected dates, which still
need checking against the official call: abstracts about 8 Feb 2027, papers about 15 Feb 2027.
We aim for a full paper and can fall back to a short one.

**Framing (orchestrator default; the user may change it):** an honest, leakage-audited
benchmark of recall-prediction models on two very different datasets (Anki and all of
Duolingo). The claims:
1. Most published models do not beat a constant-recall baseline on Duolingo. Duolingo's
   own HLR is worse than a constant on both datasets.
2. A gradient-boosted tree model that uses FSRS-state features and the user's past
   performance beats FSRS-6 on both datasets, by every metric.
3. The Bayesian MCMC model (v1) is second best on Duolingo and the best calibrated after
   GBDT, but it does poorly on Anki. This makes a Bayesian-vs-ML trade-off study.
4. A same-session leakage pitfall exists in the Duolingo data, and we show how to avoid it.

**Only true numbers go in the paper.** Every number in the paper must come from a results
file, through `paper/FACTS.md`. The disclosures in EVALUATION.md 11.4 must appear in the
paper.

## Fixed decisions

- Repo `/home/amal/fyp/Axiom`, branch `BayesianScheduler`. Python is `.venv/bin/python`.
- The paper lives in `/home/amal/fyp/Axiom/paper/`. Its state is in `paper/docs/`
  (this plan, `reports/phase-N.md`, `BACKLOG.md`).
- Benchmark code changes go in `fsrs_benchmark/`, with tests in `fsrs_benchmark/tests/`.
  Run each test file on its own:
  `for f in fsrs_benchmark/tests/test_*.py; do .venv/bin/python -m pytest -q "$f"; done`.
  Torch and numba in one pytest process can hang.
- **PC limit:** 19 GB RAM. Run at most ONE heavy job at a time locally, where heavy means
  a full-Duolingo job or a long MCMC fit. Ask before any local run longer than about 30 min.
  Heavy Duolingo work goes to Kaggle. The Anki data is gated and stays local.
- Do not re-run v1 MCMC on Anki (it takes 7.5 h). Reuse the saved predictions.
- The test set has already been looked at twice. No new model selection may use the test
  set. New baselines (the constants) are not tuned, so adding them is fine.
- Commit verified milestones locally. Never push unless the user says so. Do not commit
  `blackbook/`.

## Phases

### Phase 1: Close the evidence gaps — DONE (2026-11-15, report: reports/phase-1.md)
Goal: every number the paper needs exists in a results file, including the constant floor.

- **1.1 Constant-recall baselines.** Add two prediction "models" to `fsrs_benchmark/report.py`.
  They are computed from the run's `events.parquet`, not trained by `run.py`:
  - `const_global`: one probability, the recall rate over all TRAIN rows that are not a
    card's first review and have `elapsed_days >= 1`.
  - `const_user`: each user's own recall rate over the same kind of TRAIN rows. A user with
    no such rows falls back to `const_global`.

  They must appear in the overall table, the gap-bin tables, the paired-difference CIs
  (against `fsrs_opt` and `ours`) and the user-wins table, with display names "Constant
  (global)" and "Constant (per user)".
  Done when:
  - tests pass, including: no test row is used for a rate; the fallback works; a
    hand-computed toy example matches;
  - rebuilding the Anki report (`results/full_u100_c1000_final`) reproduces about 0.4544
    (global) and 0.4308 (per user) log loss, and the other models' numbers are unchanged.
- **1.2 Complete result sets for the paper.** Make `results/paper_anki/` and
  `results/paper_duolingo/` hold `events.parquet` and every model's predictions, as symlinks
  or copies from `full_u100_c1000_final`, `duolingo_full` (fsrs_default, fsrs_opt, hlr,
  laglogit, ours_gpu), `duolingo_full_v2` (ours_v2) and `duolingo_final` (ours_v3, gbdt).
  Build both reports, including the constants. Then add `fsrs_benchmark/paper_export.py`,
  which writes `paper/data/*.csv` (the overall metrics, gap bins, paired CIs and user wins
  per dataset) and `paper/FACTS.md` (every headline number, each with its source file).
  Done when:
  - both reports build;
  - the Duolingo numbers match the documented ones (GBDT 0.259, v1 0.268, FSRS-opt 0.300);
  - the export has tests;
  - FACTS.md lists the constant baselines.
- **1.2b Fast paired bootstrap, then the Duolingo report.** The pure-Python bootstrap takes
  513 s per call on Duolingo, which comes to about 5 h for the report. Rewrite
  `metrics.paired_bootstrap` with per-user bootstrap weights: draw the same user indices with
  the same RNG calls, then compute weighted metrics. Log loss, Brier and ECE use per-user
  sums. AUC uses a weighted rank computation with ties handled.
  Done when:
  - the new paired.csv for results/paper_anki matches the old one to 1e-9;
  - tests compare the new implementation with the old one on random data;
  - the Duolingo report builds in under 30 min;
  - the Duolingo export and FACTS section are written, and its headline numbers match.
- **1.3 Anki robustness (needs the user's OK before running; compute-heavy).** Re-run the
  cheap models (fsrs_default, fsrs_opt, hlr, ours_v3, gbdt and the constants) on two more
  100-user Anki samples (seeds 1 and 2), with the frozen settings and no tuning. Report the
  mean and spread of each model's log loss. v1 is not re-run.
  Done when: the runs finish, are reported, and are added to FACTS.md.

### Phase 2: srs-benchmark comparison — DONE (2026-11-17, report: reports/phase-2.md)
Goal: compare the GBDT directly with RWKV, LSTM, GRU and FSRS-6 on the public srs-benchmark
(about 10k Anki users).
- **2.1 Scoping (no heavy runs).** Clone `open-spaced-repetition/srs-benchmark` into
  `external/srs-benchmark` (git-ignored). Write `paper/docs/SRS_BENCHMARK.md` covering:
  - its split and scoring protocol, and how it differs from ours;
  - how a new model is added;
  - whether a cross-user model like our GBDT fits its per-user protocol;
  - the data size, RAM and time estimates, and the smallest honest subset;
  - its published numbers for the baseline models.

  The orchestrator then decides tasks 2.2 and later. DONE: see `paper/docs/SRS_BENCHMARK.md`.
- **2.2 GBDT driver for srs-benchmark, local smoke test.** Write `fsrs_benchmark/srsb/run_gbdt.py`
  following SRS_BENCHMARK.md, with the pooled RWKV-style protocol. Reuse srs-benchmark's loader, split
  and metrics by import, without modifying them. Add tests.
  Done when:
  - on 10 local users, the per-user test row counts equal the `size` in the published
    `FSRS-6-short.jsonl` (or the closest matching file; name it);
  - the driver writes jsonl lines in their format;
  - a script `fsrs_benchmark/srsb/compare.py` reproduces the README weighted mean of FSRS-6 and AVG
    from the published jsonl (on all users), and computes paired comparisons on any user subset.
- **2.3 Kaggle run.** Set A is 500 users and set B is 1,000 users, all disjoint. Bring the jsonl back,
  run compare.py, and add the numbers to FACTS.md (a new srs-benchmark section).

### Phase 3: Write the paper — DONE (2026-11-17, report: reports/phase-3.md)
EDM template, LaTeX in `paper/`. All numbers come from FACTS.md. The submission is
anonymous, double-blind.
- 3.1 Template, skeleton, bib and the build script.
- 3.2 Figures and tables generated from `paper/data/`.
- 3.3 Introduction, related work and methods.
- 3.4 Results, discussion, limitations and abstract.

### Phase 4: Polish — IN PROGRESS (4.1 done)
Page limit, an anonymity check, a check that every number traces back to FACTS.md,
a proofread, and a de-anonymised arXiv version.
- **4.1 Automated checks.** Write `paper/scripts/check_paper.py`.
  - It extracts every number from `sections/*.tex`, the abstract and the generated tables, and
    checks that each one either appears in FACTS.md (after rounding) or sits on a line tagged
    `% PROTOCOL`.
  - It flags any number line that has no `% FACTS:` or `% PROTOCOL` comment.
  - It greps for anonymity leaks (names from blackbook/ or memory, VJTI, Mumbai, FYP, Kaggle,
    the amalverma username, Axiom, PMS61) in the .tex files AND in the PDF text.
  - It reports the page count, splitting body pages from references.
  - It exits non-zero on any problem. Add tests.

  Done when it runs clean on the current paper, or every remaining hit is a reviewed exception
  recorded in the script.
- **4.2 Proofread pass.** Do one careful read of the rendered PDF for grammar, consistency
  (model names, number formats, tense) and repetition. Make only wording changes; no numbers change.
- **4.3 arXiv / camera-ready version.** Write `paper/main_camera.tex`, which reuses the sections with a
  real author block. Leave the professor's name and the author order as placeholders until the
  user gives them. Add an acknowledgements placeholder. Build both PDFs.
