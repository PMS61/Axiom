# Plan 4 — Unify Personalization Across Trends, Course, and Scheduler

## Context: personalization today is three disconnected islands

| Pillar | Profile data it actually uses | Source | Persistence |
|---|---|---|---|
| **Scheduler** (When) | `StoredUserProfile`: wake/sleep, peak/low-energy windows, fixed commitments, hard exclusions, session style, deadline style, recovery activities. Plus derived signal: `computeCalibratedMultipliers()` in `engine.ts` (per-task-type completion rate, adjusts type multipliers ±20%). | `userProfileStorage.ts` (localStorage-first, server-hydrated) | Vercel Postgres (via `app/actions/`), most mature of the three |
| **Roadmap agent** (What→How bridge) | A *different* shape (`RoadmapGoal.userProfile`: `experienceLevel`, `dailyMinutes`, `learningStyle`, plus `existingMastery` array) — not derived from `StoredUserProfile` at all, appears to be assembled ad hoc per call site. | Unclear/ad hoc | Unclear |
| **Course/Assessment/Cheatsheet/Flashcard/Short-bit/Story agents** (How) | **None.** `generateCourseSkeleton`/`generateCourseSlide` take only `title`, `description`, `userPrompt`. Same content generated regardless of who's asking, their mastery, or their learning style. | — | — |
| **Trends** (What) | A *third*, unrelated shape: `user_interests` table (`domains: string[]`, `profile_type`). Saved via `saveUserInterests()` but confirmed (Plan 3) never read back into `fetchTopTrends()`'s ranking — currently write-only. | `trends.ts` server actions | Vercel Postgres, separate table, no FK relationship to scheduler profile beyond `user_id` |

Three different profile shapes, three different degrees of completeness, and no shared mastery signal anywhere except inside the scheduler's own calibration and the roadmap agent's optional `existingMastery` input (which itself isn't shown to be fed by anything concrete — assessment results, scheduler completion, and roadmap mastery all look like they should be the same underlying number but currently have no wiring connecting them).

## Goal

One canonical learner profile, with typed projection functions (the pattern already exists and works well — `userProfileStorage.ts`'s `toDashboardProfilePayload()` and `toReportProfile()` each slice `StoredUserProfile` into exactly what one consumer needs). Extend that pattern instead of replacing it.

## Design

### 1. Canonical model

Extend `StoredUserProfile` (keep its existing schedule-related fields untouched — they work and nothing here should regress the scheduler) with:

```
learningStyle: string            // currently only lives in roadmap's ad-hoc profile shape
experienceLevel: string          // currently only lives in roadmap's ad-hoc profile shape
interestDomains: string[]        // currently only in the disconnected user_interests table
profileType: string              // ditto
masteryByTopic: Record<string, number>   // NEW — the actually-missing piece
typeCompletionRates: Record<TaskType, number>  // derived, not stored — see below
```

`masteryByTopic` is the load-bearing addition: right now mastery exists in at least three uncoordinated places —
- `RoadmapDAGNode.masteryScore` (updated by `recomputeNodeStates()` in `roadmap-agent.ts`, driven by assessment results per its own doc comment),
- `assessment-agent.ts`'s `evaluateAssessment()` output score (0.0–1.0), which has no visible write-back into anything,
- `engine.ts`'s `computeCalibratedMultipliers()`, which tracks completion *rate by task type*, not mastery by topic — a related but distinct signal.

These three need to converge on one persisted `mastery` table (`user_id`, `topic`, `score`, `updated_at`, `source` — so it's clear whether a score came from an assessment, roadmap inference, or scheduler completion) rather than living in three different in-memory/ad-hoc shapes.

### 2. Projection functions (extend the existing pattern)

Add alongside `toDashboardProfilePayload()`/`toReportProfile()`:

- `toRoadmapPersonalizationPayload(profile)` — replaces the ad-hoc `RoadmapGoal.userProfile` construction with a real projection off the canonical model, so roadmap generation and scheduler personalization can never drift out of sync again.
- `toCoursePersonalizationPayload(profile, topic)` — new; slices `experienceLevel`, `learningStyle`, and `masteryByTopic[topic]` into whatever shape `course-agent.ts` ends up taking once Plan 2's pipeline accepts a profile parameter.
- `toTrendPersonalizationPayload(profile)` — new; slices `interestDomains`, `profileType`, and the topics with the lowest `masteryByTopic` scores (the struggling-topics signal Plan 3 needs for mastery-weighted ranking).

### 3. Persistence consolidation

- Migrate `user_interests` (domains, profile_type) into the same profile table/row as the scheduler profile, or at minimum add the FK relationship and a single fetch path — stop treating trends personalization as a separate account.
- Add the new `mastery` table described above.
- Keep `StoredUserProfile`'s localStorage-first read pattern for the schedule-only fields (it's a deliberate fast-path for the dashboard/matrix UI) — the new fields (mastery, interests, learning style) should be server-fetched only, since they're read far less often (course/trend generation, not every scheduler render) and don't need the same latency treatment.

### 4. Wiring — what actually changes in each pillar

- **Course/Assessment/Cheatsheet/Flashcard/Short-bit/Story agents**: accept an optional profile parameter (see Plan 2's pipeline design, which already reserves a slot for this) — adjust vocabulary/depth/pacing by `experienceLevel` and `learningStyle`, and skip-ahead or add-remediation by `masteryByTopic[topic]`. This is the brief's "Performance-Aware Generation" item made concrete.
- **Trend ranking**: read `interestDomains`/`profileType` (finally — currently write-only) and boost topics tied to low-mastery subjects, per Plan 3's item E.
- **Roadmap agent**: swap its ad-hoc profile construction for `toRoadmapPersonalizationPayload()` — no behavior change intended here beyond removing the drift risk, since it already mostly does the right thing.
- **Scheduler**: no changes required by this plan — it's already the most complete implementation and is the reference pattern the others are being brought up to.

## Suggested sequencing

1. Design and migrate the `mastery` table + write-back points (assessment completion, roadmap mastery updates, scheduler calibration) — this is the piece that doesn't exist anywhere yet and everything else depends on it existing before it can be read.
2. Consolidate `user_interests` into the canonical profile.
3. Add the three new projection functions.
4. Wire roadmap agent to the new projection (low-risk, mostly a refactor of existing behavior).
5. Wire course agents to accept and use the profile (coordinate with Plan 2's pipeline refactor — do this at the same time, not before Plan 2's pipeline stages exist, to avoid threading a parameter through code that's about to be restructured anyway).
6. Wire trend ranking to read interests + mastery (coordinate with Plan 3's item E — same reasoning).

## Relationship to the other plans

This plan is infrastructure, not a user-facing feature — it exists because Plans 2, 3, and 5 all independently need "the same mastery/interest signal" and would otherwise each invent their own shape (which is exactly the problem that already happened three times). **Do this before or alongside Plans 2/3's personalization-dependent items**, not after — retrofitting a shared model under three already-shipped bespoke ones is strictly more work than building it first.
