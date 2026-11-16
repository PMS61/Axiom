# Proofread pass (PLAN 4.2)

Wording only. No number or claim was changed, and every `% FACTS:` / `% PROTOCOL` tag is kept.

## Changes
- **Model and constant names.** `FSRS-6 (optimised)` is now `FSRS-6 (opt.)` in prose (the tables' name). `FSRS-6 (default)` is now `FSRS-6 (def.)`. The three constants are `constant (global)`, `constant (per user)` and `constant (per user, smoothed)`, replacing "global constant", "smoothed floor" and similar. v1 is `v1 (MCMC)` at each section's first mention, then `v1`.
- **Acronyms defined at first use.** GBDT (abstract and intro), HLR (intro), MAP (intro), R-hat and ESS (intro, now with the statistic named). MCMC is defined in the abstract.
- **Number formats.** Thousands separators in prose (e.g. 95,400 scored events; 1,000 users; 2,000 sweeps; 17,060,060 rows). Decimals are unchanged; log loss and ECE are already 4 decimals in prose.
- **Repetition.** Removed the duplicate paired-share sentence (82 of 100, 78%) from the Anki/Duolingo robustness paragraph, with its FACTS tag. Removed the duplicate srs-benchmark sample sizes from the outlier-rule item in Limitations.
- **Clarity.** The results ranking now says where the smoothed constant (0.2723) falls, between v1 and v3. "Ours" in Limitations is now v1, v2, v3. "per-user floor" is now "constant (per user)". Grammar fixes in v3 item (1), the Duolingo label sentence ("recorded probability"), the calibration sentence, and the srs-benchmark RWKV-Instant sentence.
- **Abstract.** 199 words (pdftotext count), down from 202.
- **Code-release line** (intro): the anonymous text is unchanged. The `\ifcamera` branch holds the camera placeholder.

## Flagged, not changed
- Limitations/Discussion give the validation log loss 0.363 and 0.377 with 3 decimals, while the rest of the paper uses 4. The FACTS line has the same 3-decimal precision, so writing 0.3630 would imply more precision than the source has.
- Paired-share parentheticals such as "(0.7401)" are kept as raw shares next to the percentages.
