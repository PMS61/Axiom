# KAR³L learner demo sample

Source: Matthew Shu, Nishant Balepur, Shi Feng, and Jordan Boyd-Graber.
**KARL: Knowledge-Aware Retrieval and Representations aid Retention and Learning
in Students**, EMNLP 2024. <https://aclanthology.org/2024.emnlp-main.784/>

- Study logs: <https://huggingface.co/datasets/nbalepur/KARL>, revision
  `198aa632c0e8dca71c579f0647041d27d479a9e7`.
- Questions and answers: `facts.csv` in
  <https://github.com/Pinafore/fact-repetition>, revision
  `2bdcb801232dab5362610fa8297119dc5e9876f2`.
- `karl-demo.json` records both complete-source SHA-256 hashes, extraction
  criteria, the selected learner ID, and original review timestamps/outcomes.

## Reproduce

From repository root, with the existing diagnostics environment:

```bash
.venv-diagnostics/bin/python client/scripts/import_karl_demo.py
```

Downloads are cached in `/tmp/axiom-karl-data` (override with `--cache-dir`).
Only the small extracted sample belongs in the frontend bundle. Neither full
source dataset is committed.

The importer joins `card_id` to `fact_id` and checks normalized question text
for **every selected review**. No answers are inferred or generated. It selects
the learner with the largest number of eligible cards, then eight cards by
review count, using numeric IDs to break ties. Eligibility requires at least
four reviews, question length at most 130 characters, a nonempty answer, unique
review timestamps, and matching question text. Selection never filters on
recall outcome. This convenience sample is not a representative evaluation set.

The current sample contains eight cards and 173 released reviews from learner
38. `response` is used as the recorded recall outcome. The separate
`correct_on_first_try` field is preserved but is not used as the recall label.
The dataset's log counts/features can reflect events beyond the released rows;
the UI counts only the released rows actually included here.

## Demo semantics

The snapshot is one day after the last selected historical event. Day 1 offers
all eight cards for practice; it does not reproduce the original due queue.
Initial stability is the median consecutive review gap in days, floored at
one day. This is a descriptive input to the illustrative scheduler, not an
estimated memory parameter. Recall estimates and future scheduling remain
heuristics, not KAR³L predictions, FSRS, or the Bayesian research model.

Original reviews are immutable source data. New feedback is stored separately
in browser storage under `axiom-learning-karl-v1`; the old synthetic demo's
storage key is retained. Reset restores this source snapshot and clears only
the current demo session. User-added cards have no borrowed historical reviews.

## Source terms

The source repository includes the MIT notice retained verbatim in
`KARL-LICENSE.txt` (Copyright 2020 Shi Feng). The Hugging Face dataset card does
not state a separate data license. This local research demonstration does not
assert that the repository's software license resolves all rights in the trivia
content. Recheck the source terms before publishing or distributing the sample.
