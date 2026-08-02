# Plan 3 — Trends: Accurate Predictions and Recommendations

## Branch strategy

Build this plan on its own branch (e.g. `plan/03-trends-accuracy`), off current `master` — Plan 4 (personalization) is already merged there, so `toTrendPersonalizationPayload()` (`lib/personalization.ts`) is available from the start of this branch. Independent of Plans 1 and 2; no need to wait on either. Plan 5 depends on this plan merging first (see Plan 5's own branch note).

## Context: what exists today

The trend engine (`client/src/lib/trend-engine/`) is pure math with no learning or feedback:

- **Topic mapping** (`topicMapper.ts`): ~60 hand-written regex patterns, first-match-wins, ordered by specificity. Anything not matching a pattern is silently dropped (`processTrends`) or falls back to title-casing the raw query (`normalizeQueryToTopic`, used only for user-typed search). No semantic matching — "state management library" or a topic phrased unusually never maps to anything.
- **Scoring** (`scoring.ts`): `trend_score = 0.4·frequency + 0.4·engagement + 0.2·recency`, fixed weights, min-max normalized across whatever topics happened to get signals that run. `inferDirection`/`inferMomentum` compare first-half vs. second-half of whatever raw signals exist — no real time-series model, and both are sensitive to how few signals a topic has (e.g. `inferDirection` needs only 4 signals to render a verdict).
- **Sources**: exactly three (HN, Reddit, NewsAPI), fetched every call, no per-domain source selection.
- **"Insight"/"future_outlook" text**: one batch LLM call per `fetchTopTrends()` run, given only the computed score/direction/momentum as text — the LLM is not shown any actual headlines, so its "insight" is a plausible-sounding guess dressed up as analysis, not a grounded claim.
- **`searchTopic()` / `getTopicOverview()`**: single-shot Gemini calls where "current trend data" is a few numbers pasted into a prompt (or entirely absent, falling back to "use your knowledge"); `relevance_score`, `roi_estimate`, `market_relevance` etc. are model guesses with no source-checkable basis and no confidence indicator.
- **No feedback loop**: nothing checks whether a "rising" trend actually kept rising, whether a high relevance_score topic correlated with anything real, or whether the fixed 0.4/0.4/0.2 weighting is any good. The project's original research brief (superseded by this `plans/` directory, no longer in the repo) already named the intended fix for personalization ("Mastery-Weighted Trend Ranking," feeding scheduler completion-rate calibration back into topic ranking) but that link doesn't exist yet either.

This is the honest baseline: "prediction" today means a single ungrounded LLM guess per query, and "trend score" is an unvalidated fixed-weight heuristic over whatever three sources happened to mention something in the last 48 hours.

## What "more accurate" has to mean here

Two different problems are bundled under "accuracy" and need separate treatment:

1. **Trend *scoring* accuracy** — is `trend_score`/`direction`/`momentum` actually predictive of what happens next? This is checkable computationally, the same way the scheduler research plan checks scheduling quality.
2. **Recommendation *groundedness*** — is the LLM's "should you learn this" text actually based on real evidence, or is it hallucinated plausibility? This is a prompt/architecture fix, not a stats problem.

## Enhancements

### A. Topic mapping — reduce silent drops and misses

- Add an embedding-similarity fallback: when no regex pattern matches a title, embed it and compare against embeddings of the existing canonical topic list (the `KEYWORD_MAP` targets are already a fixed vocabulary — embed those once, cache them). Route to the nearest topic above a similarity threshold instead of dropping the signal. Falls back to "unmapped" (excluded, as today) only below threshold, rather than requiring exact keyword coverage.
- Keep the regex map as the fast path (it's cheap and precise for well-known terms) — the embedding step only runs on the leftover unmatched titles per fetch cycle, so cost stays bounded.

### B. Signal quality — more sources, source-aware weighting

- Add at least one more source for tech specifically: GitHub Trending (repo star velocity) and/or Stack Overflow tag activity are natural fits given the existing domain taxonomy (`DOMAINS` in `types.ts`). Each new source plugs into the same `RawSignal` shape (`fetchAllX Signals()` pattern already established by `fetchNews.ts`/`fetchReddit.ts`/`fetchHackerNews.ts`).
- Move from fixed 0.4/0.4/0.2 weights to per-source reliability weighting informed by backtesting (see D) rather than another arbitrary constant.

### C. Grounded recommendations — stop letting the LLM guess in a vacuum

- `fetchTopTrends()`'s insight prompt currently passes only `topic`, `trend_score`, `direction` — extend it to include 2–3 actual representative headlines/snippets per topic (already collected in `TopicSignals.raw`, just not passed through). The LLM should be citing real signal, not inventing plausible-sounding trend commentary.
- `searchTopic()`/`getTopicOverview()` should do the same: pass real fetched signal data whenever `processSingleTopic()` returns something, and explicitly flag in the response when no live data existed and the model fell back to background knowledge (`trend_data: undefined` today silently hides this — surface a `groundedInLiveData: boolean` field to the UI instead of hiding the distinction).
- Add a `confidence` field to `TopicSearchResult`, driven by how much real signal backed the answer (data-Grounded + high signal count → high confidence; no live data → explicitly low confidence), rather than presenting every LLM guess with equal authority.

### D. Backtesting / calibration — make the score claims checkable

- `trend_cache` already timestamps every scored run (`setCached`/`getCached` in `app/actions/trends.ts`). Add a small evaluation job that, for topics scored N days ago as "rising"/"accelerating", re-scores them now and checks whether engagement/mentions actually grew — this is the same "computational, no human subjects" evaluation posture used in the scheduler research plan (Plan 1), reused here for a different subsystem.
- Log predicted vs. realized outcomes over time; use that to periodically refit the 0.4/0.4/0.2 frequency/engagement/recency weights (and the new per-source weights from B) instead of leaving them as launch-day guesses.
- This doesn't need to be a full ML model — even a simple grid search over weight combinations minimizing backtest error is a legitimate, honestly-scoped improvement over "someone picked 0.4/0.4/0.2 once."

### E. Personalized ranking — implement the loop-closing bridge from the brief

- Wire `computeCalibratedMultipliers()` (already in `engine.ts`, tracks per-task-type completion rates) and roadmap `masteryScore` values into trend ranking: topics matching a `TaskType`/subject the user is struggling with (low completion rate, low mastery) get a ranking boost, per the original brief's "Mastery-Weighted Trend Ranking (When → What)" item. Plan 4 is now merged: `getLearnerProfile()` and `toTrendPersonalizationPayload()` (`lib/personalization.ts`) already exist and return `strugglingTopics` (mastery < 0.5, sorted lowest-first) plus `interestDomains`/`profileType` — this item can be built directly against them now, no further waiting on Plan 4.
- Also actually use the existing-but-inert `user_interests` table (`domains`, `profile_type`) to filter/boost `fetchTopTrends()` results — right now it's saved (`saveUserInterests`) but never read back into the ranking path at all.

## Suggested sequencing

1. Grounding fixes (C) — cheapest, highest trust impact, no new infra.
2. Embedding fallback for topic mapping (A) — moderate effort, fixes silent data loss.
3. Additional sources + reliability weighting (B).
4. Backtesting harness (D) — reuse patterns from Plan 1's research harness where sensible (both are "log predictions, check them later, refit weights" problems).
5. Mastery/interest-weighted personalization (E) — Plan 4 shipped already, no longer a blocker; sequence this last within the branch anyway since it should layer on top of (C)/(D)'s scoring fixes, not compound with an unfixed score.

## Open question for the user

Backtesting (D) needs a place to run periodically (cron/scheduled job) — confirm whether this is expected to run as a Vercel cron function, an external scheduler, or is out of scope for now and should just ship as an on-demand script.
