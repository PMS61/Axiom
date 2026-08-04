# Plan 5 — Closing the What/How/When Adaptive Loop

## Branch strategy

Build this plan on its own branch (e.g. `plan/05-loop-integration`), but **do not create it until Plan 2a and Plan 3 are both merged to `master`** — this plan wires their interfaces together (Bridge 2 needs Plan 2a's course pipeline profile slot; Bridge 1 needs Plan 3's recalibrated trend scoring). Branching early and rebasing repeatedly against two moving targets costs more than waiting. Plan 4 is already merged, so Bridge 3's write side is already done (see below) by the time this branch opens. (Plan 2 was split into 2a/content-pipeline and 2b/narration-video-chat after this doc was originally written — this plan's dependency is on 2a only; Plan 2b can merge independently and doesn't gate this plan.)

## Context

The project's original research brief (superseded by this `plans/` directory, no longer in the repo) already named Axiom's architecture correctly: it is "a closed loop, not a bag of independent features" — What (trend engine) → How (material generators) → When (scheduler). This framing is also recorded as a standing note for this project: the three pillars are one intentional loop, not separate features to be improved in isolation.

Today, exactly **one** of the three loop edges actually exists in code:

```
What ──(roadmap-agent.ts)──> How        ✅ implemented
How  ──────────?───────────> When       ❌ tasks flow to the scheduler, but nothing
                                            about how content was generated (mastery,
                                            performance) flows back
When ──────────?───────────> What       ❌ scheduler completion data never reaches
                                            trend ranking
```

The roadmap agent already bridges What→How: it takes a goal + profile + `existingMastery` and produces a DAG of course/assessment nodes with prerequisite locking (`recomputeNodeStates()`). That edge works and should be left alone.

The other two edges are exactly the two items the original brief names explicitly — "Mastery-Weighted Trend Ranking (When → What)" and "Performance-Aware Generation (When → How)" — and this plan is their concrete implementation plan, sequenced against Plans 2a/3/4.

## Why this has to be last

This plan does not introduce new capability by itself — it wires capability that Plans 2, 3, and 4 create:

- The When→What edge needs a readable mastery/interest signal, and needs Plan 3's trend-ranking function to have a slot for a personalization boost.
- The How→When feedback (performance data flowing from assessments/course completion back into mastery) needs a mastery table to exist as the write target.
- The When→How edge (performance-aware generation) needs Plan 2a's course pipeline to accept a profile parameter in the first place.

**Plan 4 shipped first (commit `a23c4a6`) and already satisfies the first two of these** — `user_mastery` exists, both write points are live, and `toTrendPersonalizationPayload()`/`toCoursePersonalizationPayload()` are built. What's actually still blocking this plan is narrower than originally scoped: only Plan 2a's pipeline profile-slot and Plan 3's trend-ranking hook. Attempting this plan before those two land means wiring against interfaces that don't exist yet or will change shape. This is explicitly the integration milestone, not a parallel workstream. (Plan 2b — narration/video/avatar/chat — is not a dependency here at all.)

## Concrete data flow to implement

```
                    ┌─────────────────────────────────────────────┐
                    │         user_mastery table (Plan 4 — LIVE)    │
                    │   user_id · topic · score · source · updated  │
                    └───────────────┬───────────────────┬──────────┘
                                     │ read                │ write
                     ┌───────────────▼──────┐   ┌──────────▼───────────────┐
   Trend ranking ◄───┤ toTrendPersonalization│   │ Scheduler completion      │
   (Plan 3, item E)  │ Payload()             │   │ (computeCalibratedMulti-  │
                     └───────────────────────┘   │  pliers, engine.ts)       │
                                                  │ Assessment results        │
                                                  │ (evaluateAssessment)      │
                                                  │ Roadmap mastery updates   │
                     ┌───────────────────────┐   │ (recomputeNodeStates)     │
   Course generation ◄┤ toCoursePersonaliza- │   └───────────────────────────┘
   (Plan 2a, staged   │ tionPayload()         │
    pipeline)         └───────────────────────┘
```

### Bridge 1 — When → What (Mastery-Weighted Trend Ranking)

- Trigger point: every `fetchTopTrends()` call (`app/actions/trends.ts`), after the existing score/insight pipeline runs.
- Read `toTrendPersonalizationPayload(profile)` (`lib/personalization.ts`, live since Plan 4) to get the user's lowest-mastery topics and declared interest domains.
- Apply a ranking boost (not a hard filter — a struggling-but-relevant topic should surface higher, not replace the global trend signal entirely) before returning results to the dashboard.
- This is additive to Plan 3's own scoring-accuracy work — Plan 3 fixes whether `trend_score` itself means anything; this bridge decides how personal ranking sits on top of that already-fixed score. Sequence Plan 3's core scoring fixes before this bridge, so personalization isn't layered on top of a score still being recalibrated.

### Bridge 2 — When → How (Performance-Aware Generation)

- Trigger point: whenever a course/assessment/flashcard/short-bit/etc. is generated for a specific topic (Plan 2a's Stage 1/2 — Cheatsheet was removed and Story folded into Course under Plan 2a's scope decision), pull `masteryByTopic[topic]` via `toCoursePersonalizationPayload()`.
- Low mastery on a topic → generation prompt asks for more foundational framing, more worked examples, slower pacing. High mastery → generation skips basics, assumes more prior knowledge, can move faster. This is a prompt-construction change in Plan 2a's pipeline, not a new subsystem — the hook already exists in Plan 2a's design (profile parameter reserved at Stage 1/2), this bridge is what actually populates and uses it in production instead of leaving it null.

### Bridge 3 — How/When → mastery table (the write side, easy to overlook)

**Done — shipped as part of Plan 4 (commit `a23c4a6`), no work remaining here.** Recorded for completeness since this is where the end-to-end flow gets verified:

1. ✅ `evaluateAssessmentAction` (`app/actions/assessment.ts`) writes to `user_mastery` with `source: "assessment"` — this write always overwrites (it's a graded measurement).
2. ✅ `completeNodeAction` (`app/actions/roadmap.ts`) writes to `user_mastery` with `source: "roadmap"` — the merge policy was decided explicitly rather than left as undefined behavior: a `"roadmap"`-sourced write never overwrites an existing `"assessment"`-sourced score for the same topic, since the roadmap signal is a coarse "task completed" flag (always 1), not a graded score.
3. `engine.ts`'s `computeCalibratedMultipliers()` stays a *type*-level signal (learning/problem_solving/writing/etc.), not topic-level — confirmed correct to keep it out of `user_mastery` and live-computed rather than persisted, per Plan 4's own note. Use it specifically for pacing/format decisions (e.g. "this user under-completes writing-type tasks" → generate lighter writing-heavy content), not for topic-selection ranking.

## What "done" looks like

A concrete end-to-end scenario that should work once this plan lands:

1. User completes an assessment on "React Hooks" scoring 40%.
2. `mastery["React Hooks"] = 0.4` is written.
3. Next trends fetch: "React" (mapped topic, per `topicMapper.ts`) gets a ranking boost even if its raw `trend_score` is middling, because the user has low mastery there.
4. User generates a new course on "React Hooks" (or the roadmap agent auto-suggests it, already prioritized higher via `existingMastery` in its own prompt): the course pipeline reads `mastery["React Hooks"] = 0.4` and generates more foundational content than it would for a topic scored 0.9.
5. User completes the new course's assessment at 75% — mastery updates, next generation/ranking cycle reflects it.

This scenario is the acceptance test for the whole plan — if it doesn't hold end-to-end, the loop isn't actually closed regardless of how much of Plans 2–4 shipped individually.

## Suggested sequencing

1. ~~Confirm Plan 4's mastery table and projection functions are live~~ — **done**, shipped in commit `a23c4a6`.
2. ~~Implement the three write points (Bridge 3)~~ — **done**, shipped alongside Plan 4.
3. Confirm Plan 3's core scoring/grounding fixes are merged before layering Bridge 1 on top (soft dependency — avoids compounding an unfixed score with unvalidated personalization).
4. Confirm Plan 2a's staged pipeline has the profile-parameter slot merged (hard dependency for Bridge 2; Plan 2b is not a dependency).
5. Implement Bridge 1 (When→What).
6. Implement Bridge 2 (When→How).
7. Run the end-to-end acceptance scenario above manually, then as a repeatable test.
