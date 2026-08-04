# Plan 4 — Unify Personalization Across Trends, Course, and Scheduler

## Status: core implemented (commit `a23c4a6`)

What shipped, and how it differs from the original design below (kept for history — this file is still the reference for what's left):

| Sequencing step | Status | Notes |
|---|---|---|
| 1. `mastery` table + write-back points | **Done** | Landed as `user_mastery` (not `mastery`) in `app/actions/personalization.ts`. Write-back wired at both points identified: `evaluateAssessmentAction` (assessment.ts) and `completeNodeAction` (roadmap.ts). Merge policy implemented exactly as specified — `source="assessment"` always overwrites, `source="roadmap"` only writes when no assessment-sourced row exists yet. Scheduler's `computeCalibratedMultipliers()` was deliberately **not** wired into this table — confirmed correct per this plan's own §4 note that it's a type-level signal, not topic-level; it stays live-computed from `tasks`, no persistence needed. |
| 2. Consolidate `user_interests` | **Done, differently than specified** | Did **not** physically merge `user_interests` into the `users` row — kept it as its own table (it already had a clean FK) and instead built the "single fetch path" the design allowed as an alternative: `getLearnerProfile(userId)` joins `users` + `user_interests` + `section_weights` + `user_mastery` into one object. Lower migration risk, same practical effect (nothing reads `user_interests` directly anymore outside this join). |
| 3. Three projection functions | **Done** | Landed in `lib/personalization.ts`: `toRoadmapPersonalizationPayload` (wired), `toCoursePersonalizationPayload` and `toTrendPersonalizationPayload` (built, intentionally unwired — see step 5/6). Also added `normalizeTopicKey()`, not in the original design — routes every mastery key through the trend engine's `normalizeQueryToTopic()` so mastery topics, roadmap node titles, and trend topic labels converge on one vocabulary instead of three. |
| 4. Wire roadmap agent | **Done, bigger than expected** | The plan assumed this was "mostly a refactor" of an existing ad-hoc construction. Investigation found `RoadmapGoal.userProfile` and `existingMastery` were typed fields **never populated by any call site** — `RoadmapGoalModal` only ever submitted `{goal, syllabus, deadline}`. This was a from-scratch wire, not a refactor: `generateRoadmapAction` now fetches `getLearnerProfile()` and fills both fields when the caller didn't set them explicitly. |
| 5. Wire course agents | **Not started — deferred, as planned** | Waiting on Plan 2a's pipeline refactor (Plan 2 was split into 2a/2b — this dependency is on 2a specifically, not 2b's narration/video/chat work), per this file's own §"Suggested sequencing" note. `toCoursePersonalizationPayload` is ready to consume. |
| 6. Wire trend ranking | **Not started — deferred, as planned** | Waiting on Plan 3's scoring/grounding fixes landing first. `toTrendPersonalizationPayload` is ready to consume. |

Also shipped but not in the original design: a canonical `LearnerProfile` type (`lib/types.ts`) — rather than extending `StoredUserProfile` directly as §1 originally proposed, personalization fields live in a separate type, fetched server-side only, exactly matching this plan's own §3 guidance to keep them out of the localStorage-first fast path. And a `users.experience_level`/`users.learning_style` column pair plus a minimal edit UI in `ProfileClient.tsx`, since those fields had no capture path anywhere (not scoped in the original design, added because the alternative was another write-only dead field like `user_interests` was before this plan).

**Next consumers**: Plan 2a and Plan 3 pick up `toCoursePersonalizationPayload`/`toTrendPersonalizationPayload` respectively — see the branch note in each plan's file. (Plan 2 was subsequently split into 2a/content-pipeline and 2b/narration-video-chat; this consumer relationship belongs to 2a only.)

---

## Context: personalization today is three disconnected islands

| Pillar | Profile data it actually uses | Source | Persistence |
|---|---|---|---|
| **Scheduler** (When) | `StoredUserProfile`: wake/sleep, peak/low-energy windows, fixed commitments, hard exclusions, session style, deadline style, recovery activities. Plus derived signal: `computeCalibratedMultipliers()` in `engine.ts` (per-task-type completion rate, adjusts type multipliers ±20%). | `userProfileStorage.ts` (localStorage-first, server-hydrated) | Vercel Postgres (via `app/actions/`), most mature of the three |
| **Roadmap agent** (What→How bridge) | A *different* shape (`RoadmapGoal.userProfile`: `experienceLevel`, `dailyMinutes`, `learningStyle`, plus `existingMastery` array) — not derived from `StoredUserProfile` at all, appears to be assembled ad hoc per call site. | Unclear/ad hoc | Unclear |
| **Course/Assessment/Cheatsheet/Flashcard/Short-bit/Story agents** (How) — *note: Plan 2a later removed Cheatsheet and folded Story into Course as a mode; table kept as-is for history* | **None.** `generateCourseSkeleton`/`generateCourseSlide` take only `title`, `description`, `userPrompt`. Same content generated regardless of who's asking, their mastery, or their learning style. | — | — |
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
- `toCoursePersonalizationPayload(profile, topic)` — new; slices `experienceLevel`, `learningStyle`, and `masteryByTopic[topic]` into whatever shape `course-agent.ts` ends up taking once Plan 2a's pipeline accepts a profile parameter.
- `toTrendPersonalizationPayload(profile)` — new; slices `interestDomains`, `profileType`, and the topics with the lowest `masteryByTopic` scores (the struggling-topics signal Plan 3 needs for mastery-weighted ranking).

### 3. Persistence consolidation

- Migrate `user_interests` (domains, profile_type) into the same profile table/row as the scheduler profile, or at minimum add the FK relationship and a single fetch path — stop treating trends personalization as a separate account.
- Add the new `mastery` table described above.
- Keep `StoredUserProfile`'s localStorage-first read pattern for the schedule-only fields (it's a deliberate fast-path for the dashboard/matrix UI) — the new fields (mastery, interests, learning style) should be server-fetched only, since they're read far less often (course/trend generation, not every scheduler render) and don't need the same latency treatment.

### 4. Wiring — what actually changes in each pillar

- **Course/Assessment/Flashcard/Short-bit agents** (Cheatsheet since removed, Story since folded into Course as a mode — per Plan 2a's scope decision): accept an optional profile parameter (see Plan 2a's pipeline design, which already reserves a slot for this) — adjust vocabulary/depth/pacing by `experienceLevel` and `learningStyle`, and skip-ahead or add-remediation by `masteryByTopic[topic]`. This is the brief's "Performance-Aware Generation" item made concrete.
- **Trend ranking**: read `interestDomains`/`profileType` (finally — currently write-only) and boost topics tied to low-mastery subjects, per Plan 3's item E.
- **Roadmap agent**: swap its ad-hoc profile construction for `toRoadmapPersonalizationPayload()` — no behavior change intended here beyond removing the drift risk, since it already mostly does the right thing.
- **Scheduler**: no changes required by this plan — it's already the most complete implementation and is the reference pattern the others are being brought up to.

## Suggested sequencing

1. Design and migrate the `mastery` table + write-back points (assessment completion, roadmap mastery updates, scheduler calibration) — this is the piece that doesn't exist anywhere yet and everything else depends on it existing before it can be read.
2. Consolidate `user_interests` into the canonical profile.
3. Add the three new projection functions.
4. Wire roadmap agent to the new projection (low-risk, mostly a refactor of existing behavior).
5. Wire course agents to accept and use the profile (coordinate with Plan 2a's pipeline refactor — do this at the same time, not before Plan 2a's pipeline stages exist, to avoid threading a parameter through code that's about to be restructured anyway).
6. Wire trend ranking to read interests + mastery (coordinate with Plan 3's item E — same reasoning).

## Relationship to the other plans

This plan is infrastructure, not a user-facing feature — it exists because Plans 2a, 3, and 5 all independently need "the same mastery/interest signal" and would otherwise each invent their own shape (which is exactly the problem that already happened three times). **Do this before or alongside Plans 2a/3's personalization-dependent items**, not after — retrofitting a shared model under three already-shipped bespoke ones is strictly more work than building it first. (Plan 2b, the narration/video/chat half of the original Plan 2, has no dependency on this plan.)
