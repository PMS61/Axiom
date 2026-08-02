# Plan 5 — Closing the What/How/When Adaptive Loop

## Context

`AXIOM_PROJECT_BRIEF.md` already names Axiom's architecture correctly: it is "a closed loop, not a bag of independent features" — What (trend engine) → How (material generators) → When (scheduler). This framing is also recorded as a standing note for this project: the three pillars are one intentional loop, not separate features to be improved in isolation.

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

The other two edges are exactly the two items `AXIOM_PROJECT_BRIEF.md` §06 names explicitly — "Mastery-Weighted Trend Ranking (When → What)" and "Performance-Aware Generation (When → How)" — and this plan is their concrete implementation plan, sequenced against Plans 2/3/4.

## Why this has to be last

This plan does not introduce new capability by itself — it wires capability that Plans 2, 3, and 4 create:

- The When→What edge needs Plan 4's `masteryByTopic` + `typeCompletionRates` to exist and be readable, and needs Plan 3's trend-ranking function to have a slot for a personalization boost.
- The How→When feedback (performance data flowing from assessments/course completion back into mastery) needs Plan 4's `mastery` table to exist as the write target.
- The When→How edge (performance-aware generation) needs Plan 2's course pipeline to accept a profile parameter in the first place.

Attempting this plan before 2/3/4 land means wiring against interfaces that don't exist yet or will change shape. This is explicitly the integration milestone, not a parallel workstream.

## Concrete data flow to implement

```
                    ┌─────────────────────────────────────────────┐
                    │              mastery table (Plan 4)           │
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
   (Plan 2, staged    │ tionPayload()         │
    pipeline)         └───────────────────────┘
```

### Bridge 1 — When → What (Mastery-Weighted Trend Ranking)

- Trigger point: every `fetchTopTrends()` call (`app/actions/trends.ts`), after the existing score/insight pipeline runs.
- Read `toTrendPersonalizationPayload(profile)` (Plan 4) to get the user's lowest-mastery topics and declared interest domains.
- Apply a ranking boost (not a hard filter — a struggling-but-relevant topic should surface higher, not replace the global trend signal entirely) before returning results to the dashboard.
- This is additive to Plan 3's own scoring-accuracy work — Plan 3 fixes whether `trend_score` itself means anything; this bridge decides how personal ranking sits on top of that already-fixed score. Sequence Plan 3's core scoring fixes before this bridge, so personalization isn't layered on top of a score still being recalibrated.

### Bridge 2 — When → How (Performance-Aware Generation)

- Trigger point: whenever a course/assessment/cheatsheet/etc. is generated for a specific topic (Plan 2's Stage 1/2), pull `masteryByTopic[topic]` via `toCoursePersonalizationPayload()`.
- Low mastery on a topic → generation prompt asks for more foundational framing, more worked examples, slower pacing. High mastery → generation skips basics, assumes more prior knowledge, can move faster. This is a prompt-construction change in Plan 2's pipeline, not a new subsystem — the hook already exists in Plan 2's design (profile parameter reserved at Stage 1/2), this bridge is what actually populates and uses it in production instead of leaving it null.

### Bridge 3 — How/When → mastery table (the write side, easy to overlook)

For bridges 1 and 2 to have real data instead of always reading zeros, three write points need to land (part of Plan 4's scope, called out here because the integration plan is where the *end-to-end* flow gets verified):

1. `assessment-agent.ts`'s `evaluateAssessment()` result → write to `mastery` table (`source: "assessment"`).
2. `roadmap-agent.ts`'s `recomputeNodeStates()` mastery updates → write to `mastery` table (`source: "roadmap"`), reconciled/merged with assessment-sourced scores for the same topic rather than overwritten blindly (e.g. weighted average or "most recent wins" — decide explicitly, don't leave it as undefined behavior).
3. `engine.ts`'s `computeCalibratedMultipliers()` → this is a *type*-level signal (learning/problem_solving/writing/etc.), not topic-level — do not force it into the topic-keyed `mastery` table. Keep it as the separate `typeCompletionRates` field Plan 4 already reserves, and use it specifically for pacing/format decisions (e.g. "this user under-completes writing-type tasks" → generate lighter writing-heavy content), not for topic-selection ranking.

## What "done" looks like

A concrete end-to-end scenario that should work once this plan lands:

1. User completes an assessment on "React Hooks" scoring 40%.
2. `mastery["React Hooks"] = 0.4` is written.
3. Next trends fetch: "React" (mapped topic, per `topicMapper.ts`) gets a ranking boost even if its raw `trend_score` is middling, because the user has low mastery there.
4. User generates a new course on "React Hooks" (or the roadmap agent auto-suggests it, already prioritized higher via `existingMastery` in its own prompt): the course pipeline reads `mastery["React Hooks"] = 0.4` and generates more foundational content than it would for a topic scored 0.9.
5. User completes the new course's assessment at 75% — mastery updates, next generation/ranking cycle reflects it.

This scenario is the acceptance test for the whole plan — if it doesn't hold end-to-end, the loop isn't actually closed regardless of how much of Plans 2–4 shipped individually.

## Suggested sequencing

1. Confirm Plan 4's `mastery` table and projection functions are live (hard dependency).
2. Confirm Plan 3's core scoring/grounding fixes are live before layering Bridge 1 on top (soft dependency — avoids compounding an unfixed score with unvalidated personalization).
3. Confirm Plan 2's staged pipeline has the profile-parameter slot open (hard dependency for Bridge 2).
4. Implement the three write points (Bridge 3).
5. Implement Bridge 1 (When→What).
6. Implement Bridge 2 (When→How).
7. Run the end-to-end acceptance scenario above manually, then as a repeatable test.
