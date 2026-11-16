# Research paper: final report (2026-11-17)

## What exists
- **Anonymous submission PDF**: `paper/build/main.pdf`, 10 pages. The body runs to page 9 and the references fill pages 9-10. The abstract is 199 words, and the paper uses the official EDM template.
- **Camera-ready PDF**: `paper/build/main_camera.pdf`, 10 pages. It has the author block (Amal Verma, Vedaang Pagnis, Prathamesh Sankhe and Vaishvi Khandelwal of VJTI, Mumbai), but the author order, the emails, the supervisor's name, the acknowledgements and the code link are all placeholders (TBD).
- **Working title**: "Beating the Floor: A Leakage-Audited Benchmark of Recall Prediction for Spaced Repetition".
- **Build**: `cd paper && ./build.sh` builds both PDFs. `.venv/bin/python paper/scripts/check_paper.py` then checks that every number traces to `paper/FACTS.md`, that every number line is tagged, that nothing de-anonymises the submission, and the page count.
- **Regenerating everything from results**:
  1. `fsrs_benchmark/paper_export.py` writes `paper/data` and `FACTS.md`.
  2. `paper/scripts/make_tables.py` and `make_figures.py` write the tables and figures.
  3. `build.sh` builds the PDFs.

  `paper/scripts/README.md` gives the full order.

## Main findings (all from FACTS.md)
1. **Floors.** On Duolingo, a smoothed per-user constant (log loss 0.2723) beats FSRS-6 (optimised, 0.3003), v2, v3, LagLogit and HLR. Only GBDT (0.2588) and the Bayesian MCMC model v1 (0.2682) beat it, and v1 only by −0.0040 [−0.0045, −0.0035]. HLR does worse than the constant on both datasets.
2. **GBDT.** It beats FSRS-6 (optimised) on Anki (−0.0210, with a CI that excludes 0), and does so again on two more 100-user Anki samples (−0.0226 and −0.0210). On Duolingo the margin is −0.0415.
3. **srs-benchmark.** These are 1,000 held-out users, and all 1,000 match the benchmark's own test rows. The pooled GBDT scores 0.3211 (unweighted per-user mean), against 0.3580 for FSRS-6, 0.3441 for GRU and 0.2860 for RWKV-Instant. The size-weighted difference against FSRS-6 is −0.0235, with a 99% CI of [−0.0286, −0.0183]. GBDT ties RWKV-Curve and trails RWKV-Instant. Its boosting hit the round cap, so this is a lower bound.
4. **Leakage.** In the Duolingo traces, rows from the same session share a timestamp. Letting them see each other inflated validation log loss (0.2473 vs 0.2578). This pitfall is documented and fixed.

## How it was checked
- Every task had a Haiku builder and a separate Haiku verifier, with fix loops until the verifier reported NO FINDINGS.
- The verifiers recomputed the headline numbers independently.
- The paper checker was itself attacked with planted errors, and it caught every one.
- The full test suite was run file by file. The only failures are one known old test (`test_task52_54_verify.py::test_resume_config_records_thin_for_gpu_draws`) and one expected xfail.

## Commits (local, BayesianScheduler, NOT pushed)
d29dd07 … through the Phase 4 commit. See `git log --oneline` from d29dd07 onwards.

## What is missing / only the user can do
- **The EDM 2027 call for papers.** Check the page limit, the deadlines (expected around 8 Feb 2027 for abstracts and 15 Feb 2027 for papers) and the anonymity rules once the call is published. The paper currently has 9 body pages.
- **Authors.** Give the supervisor's name and the author order. Fill in the emails and the acknowledgements in `paper/main_camera.tex`.
- **Code release.** Decide whether to release the code. If so, push the branch or make a cleaned repo, then fill in the `\url{TBD}` in intro.tex (camera-ready only). Nothing has been pushed.
- **Kaggle.** The notebook `notebook8f676349c8` holds the `HF_TOKEN` secret, which is a read-only HF token named `kaggle-anki-paper`. Revoke it on Hugging Face (Settings → Tokens) when it is no longer needed. The private datasets `axiom-paper-code` and `axiom-paper-code-srsb` hold code only. The Anki data is downloaded at run time and never uploaded.
- **Optional, to strengthen the paper:**
  - re-run the srs-benchmark GBDT with more boosting rounds, since it hit the cap;
  - evaluate on more of the ~10k users;
  - run a scheduling simulation;
  - get a converged v1 chain.
- **Backlog.** See `paper/docs/BACKLOG.md`. The LOW items include the remaining overfull boxes under 10 pt and the rounding of one user-win share.
