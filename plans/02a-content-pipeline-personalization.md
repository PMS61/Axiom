# Plan 2a — Content Pipeline Refactor: Course/Flashcard/Short-bit/Assessment + Personalization

Split off from the original Plan 2 (`02-course-generation-agentic-video.md`, now an index pointing to this file and 2b) so the content-pipeline work and the narration/video/avatar/chat work can be assigned to two people in parallel with minimal file overlap. See "Interference boundary with Plan 2b" below before touching any file this doc doesn't explicitly own.

## Branch strategy

Build on its own branch (e.g. `plan/02a-content-pipeline`), off current `master` — Plan 4 (personalization) is already merged, so `toCoursePersonalizationPayload()` (`lib/personalization.ts`) is available from the start. Independent of Plans 1 and 3. **Plan 2b does not need to wait for this plan to merge** — only Plan 2b's final integration step (wiring its Narration Writer as Stage 4 after this plan's Stage 3 critic) needs this plan's Stage 1–3 typed output contract to exist; the rest of 2b is independent (see 2b's own doc). Plan 5 needs this plan (2a) merged — specifically the profile-parameter slot on `course-agent.ts` — before it can build Bridge 2 (When→How). Plan 5 does **not** need Plan 2b.

## Interference boundary with Plan 2b

To keep the two branches mergeable without fighting over the same files:

- **This plan (2a) owns**: `course-agent.ts` (Stages 1–3 + narrative-mode flag only, not Stage 4+), `flashcard-agent.ts`, `short-bit-agent.ts`, `assessment-agent.ts`, `assessment-type.ts`, `story-agent.ts` (deletion), `cheatsheet-agent.ts` (deletion), `FlashcardStack.jsx`, `ShortBitStack.jsx` (all engagement features except the audio/listen button — see below), `SlideViewer.tsx` (narrative-mode rendering path), `app/actions/assessment.ts`, personalization wiring into all four generators, mastery write-back from flashcard.
- **Do not touch**: anything under Plan 2b's ownership (narration/TTS/Rhubarb/`CourseNarrator`/Remotion/doubt-chat, or the course player page's narration/chat/video UI composition).
- **One deliberate shared touchpoint**: Flashcard/Short-bit's "Listen" (Piper audio) buttons are Plan 2b's to add (2b owns the Piper wrapper), landing as a small fast-follow PR to `FlashcardStack.jsx`/`ShortBitStack.jsx` *after* this plan's version of those components has merged — sequenced, not concurrent, to avoid a live merge conflict on the same component files.
- **Contract for 2b**: freeze and export the typed Stage 1–3 output shape (`CourseSkeleton`/`CourseSlide`/critic result) early in this branch's work so 2b can build its Stage 4 Narration Writer against real types without waiting for the rest of this plan.

## Context: what exists today

`course-agent.ts` is two flat, sequential Gemini calls with no orchestration:

1. `generateCourseSkeleton()` — one LLM call, title/description/userPrompt in, a flat array of `{title, description, type}` slide specs out (max 3 slides per topic-title, types drawn from `slide-prompts.ts`'s 12 registered slide kinds: title, section, content, image, mermaid, quote, code, content-and-image, content-and-code, table, list, conclusion).
2. `generateCourseSlide()` — one LLM call per skeleton entry (run in `Promise.all`, so "parallel" but not orchestrated — no shared state, no critique, no retry-with-feedback), rendered by `SlideViewer.tsx`.

No personalization input anywhere (confirmed absent from `course-agent.ts`'s signature, unlike `roadmap-agent.ts` which does take a profile).

The project's original research brief (superseded by this `plans/` directory, no longer in the repo) scoped "How" pillar work as an engineering showcase and named the exact techniques to demonstrate: self-refinement loop (Reflexion-style critic), model routing by task complexity, adaptive in-context learning from user history, and performance-aware generation. This plan operationalizes those across all four generators this plan owns.

**Audit of the other generators (`lib/*-agent.ts`), done to inform scope:**

- `flashcard-agent.ts` — one Gemini call, no personalization input. Has real defensive parsing though: `normalizeSet()` validates each card has non-empty `front`/`back`, drops malformed entries, falls back to a hardcoded 5-card set if the LLM output doesn't parse into ≥3 valid cards. Best-defended of the non-Course agents.
- `short-bit-agent.ts` — same shape: one Gemini call, no personalization, but has `normalizeSet()` + `fallbackSet()` with defensive validation (drops items missing `blogContent`, simplifies/whitelists `imageKeyword` against a fixed word list, falls back to hardcoded items if <3 valid). Comparably well-defended to flashcard.
- `story-agent.ts` — one Gemini call (`generateNextChapterMarkdown`), returns raw markdown. No schema to validate against (freeform prose), no fallback content, no personalization. Thinnest of the group. Being retired (see below).
- `assessment-agent.ts` — see "Assessment refactor" section; this one has actual correctness bugs, not just missing personalization.

None of the four have a critic/self-refinement stage, model routing, or the personalization payload from Plan 4 wired in.

## Scope decision (final)

Content-type surface is three core types — **Course, Flashcard, Short-bit** — each getting equal-effort treatment (staged pipeline, critic, personalization). **Story** is retired as a standalone generator/route and folded into Course as a narrative generation mode. **Cheatsheet** is removed entirely. **Assessment** is not one of the three content types (graded/functional, not a consumption format) but is pulled into this plan's refactor scope because of real defects found below.

- **Course** (this plan owns Stages 1–3 + narrative mode only — narration/video/avatar/chat are Plan 2b's): staged pipeline, personalization, narrative mode.
- **Flashcard** — staged pipeline + personalization + engagement UI:
  - Self-rated recall ("Got it" / "Struggled") after flip, writes to `user_mastery` with `source: "flashcard"`, using the same overwrite/merge policy `evaluateAssessmentAction` already establishes (Plan 5) — biggest gap: flashcards currently don't feed the mastery loop at all.
  - Spaced-repetition queue — missed cards resurface instead of a single linear pass.
  - Shuffle mode — only sequential order exists today.
  - Session summary screen — X/Y recalled, shown at end of a pass.
  - (Piper audio per card is Plan 2b's fast-follow addition to this component, not this plan's.)
- **Short-bit** — staged pipeline + personalization + engagement UI:
  - Render an actual image keyed off `imageKeyword` instead of just printing it as a text label (`VISUAL CUE: DESK`) — the field exists for this purpose but nothing renders it as an image today.
  - Swipe gesture support — the component's own copy says "Swipe your focus one bit at a time" but only click buttons are wired; no touch/swipe handler exists today. Copy/functionality mismatch, not just a missing nice-to-have.
  - Streak/completion tracking tied into the personalization/mastery loop.
  - (Piper listen mode is Plan 2b's fast-follow addition to this component, not this plan's.)
- **Story → Course narrative mode**: `generateNextChapterMarkdown`'s narrative-generation capability folds into Course as a generation *mode* — a `style: 'standard' | 'narrative'` (naming TBD at implementation) flag alongside the personalization payload in `generateCourseSkeleton`/`generateCourseSlide`. Retire `/dashboard/story` route and `story-agent.ts`'s call site once the Course-mode path covers it.

  **Narrative mode spec (approved):** changes both layout and tone. Content shifts from bullet/structured slide layout to free-flowing prose meant to be relaxed-read or listened to — "something one can relax and read/listen to." Concretely:
  - **Stage 1 (skeleton)**: fewer rigid structural constraints than standard mode's `{title, description, type}` bullet-driven slide specs — sections read as chapters/beats rather than a strict topic outline.
  - **Stage 2 (slide content)**: prose-driven paragraphs instead of bullet lists; check which of the 12 registered slide kinds (`slide-prompts.ts`) still apply vs. need a narrative-specific rendering path in `SlideViewer.tsx`.
  - **Critic rubric (Stage 3)** needs a narrative-specific variant — "matches description" / "appropriate slide-type usage" don't transfer directly; needs a "reads as coherent, relaxed narrative prose" check instead.
  - Note for whoever picks up Plan 2b: narrative-mode courses should get a narration script that tracks close to on-slide prose (near-verbatim) rather than the more expansive spoken-script approach standard-mode courses get — flag this in 2b's Stage 4 design, this plan doesn't build narration itself.
- **Cheatsheet** — removed entirely. Delete `cheatsheet-agent.ts`, the `app/dashboard/cheatsheet` route, and its nav entry.

## Architecture: staged pipeline (Stages 1–3, applies to all four generators this plan owns)

Replace each generator's flat one-or-two-call structure with an explicit staged pipeline (hand-rolled TypeScript, each stage a typed input/output contract):

```
Stage 1 — Planner            generate*Skeleton() / generate*Set()   [existing per-agent, keep]
Stage 2 — Generator          generate*Slide() / per-item generation [existing per-agent, keep]
Stage 3 — Critic / Refiner   scoreAgainstRubric() →                 [new — Reflexion-style]
                              regenerate if below threshold
```

Stage 3 (critic) and model routing are the direct implementation of the brief's "Self-refinement loop" and "Model routing" showcase items — a cheap/fast model call scores each generated unit (slide, card, item, question) against a short type-appropriate rubric and triggers one regeneration if below threshold, capped at 1 retry to bound cost. This is an LLM-judge quality gate, explicitly not human-testing evidence (keep that distinction in any research-adjacent writeup, per the brief's own caveat).

Rubric shape varies per type: Course = clarity/factual plausibility/matches description/appropriate slide-type usage (or the narrative variant above); Flashcard = front/back clarity, no duplicate concepts across the set, appropriate difficulty spread; Short-bit = punchy/self-contained/matches ~30-second-read framing; Assessment = question clarity, appropriate difficulty mix, no ambiguous correct answers.

**Course's Stage 1–3 output contract must be frozen and exported early** — this is what Plan 2b's Stage 4 (Narration Writer) builds against.

## Assessment refactor (`assessment-agent.ts`, `app/actions/assessment.ts`)

Pulled into this plan after an audit found it behind Course's pre-refactor baseline, not just missing personalization. Concrete defects found by direct read of the current code:

- **Coding question type is dropped entirely (approved).** Assessments are now MCQ + short answer + long answer only. `CodingQuestion` (assessment-type.ts:27-39), its `testCases`/`starterCode`/`language` fields, and the "at least 1 Coding Question if applicable" generation rule (assessment-agent.ts:24) are removed. This also removes any need for sandboxed code execution — no longer a concern for this plan.
- **MCQ grading is delegated to the LLM despite being deterministic.** `evaluateAssessment`'s prompt (assessment-agent.ts:91) asks Gemini to "check against correct options" even though `correctOptionIndex` is already embedded in the assessment JSON at generation time — a plain equality check. Fix: grade MCQ programmatically in `evaluateAssessment`/`evaluateAssessmentAction`, only send short/long answers to the LLM for evaluation.
- **`evaluateAssessment` returns `Promise<any>`** (assessment-agent.ts:84) — no return type, defeats the type system for every caller.
- **Type/data mismatch**: `AssessmentResult.nodeId: string` (assessment-type.ts:50) is declared but never populated anywhere. Either the field is dead and should be removed, or something upstream is supposed to inject it and doesn't — needs a decision, not just a type-level patch.
- **No schema validation on parsed LLM output** — both `generateAssessment` and `evaluateAssessment` do `JSON.parse(raw)` and blind-cast (agent.ts:68,74,111), unlike flashcard/short-bit's `normalizeSet()`/`fallbackSet()` pattern. Bring assessment up to that same defensive-parsing bar.
- **No personalization input** — same gap as the other three.
- **No critic/self-refinement stage** — one-shot generation, no quality gate.

Scope: drop the coding question type, apply the Stage 1–3 pattern, add the personalization hook, fix the MCQ grading and `nodeId` type mismatch above, bring parsing up to the flashcard/short-bit defensive-validation standard.

## Personalization hook (ties to Plan 4 — already shipped)

Plan 4 is merged: `getLearnerProfile(userId)` and `toCoursePersonalizationPayload(profile, topic)` (`lib/personalization.ts`) already exist and return `{experienceLevel, learningStyle, masteryScore}` for a given topic. Wire this into `generateCourseSkeleton`/`generateCourseSlide`, `generateFlashcardSet`, `generateShortBitSet`, and `generateAssessment` — all four currently take only `(title, description, ...)` with no profile input, confirmed by direct read. Adjust depth/pacing/vocabulary by `experienceLevel`/`learningStyle`, skip-ahead or add-remediation by `masteryByTopic[topic]`. This is real wiring now, not a follow-up — do it as part of each generator's Stage 1/2 build.

This is also the hard dependency Plan 5 is waiting on (its Bridge 2, When→How) — the profile-parameter slot on `course-agent.ts` specifically.

## Suggested sequencing

1. Refactor `course-agent.ts` into the explicit staged pipeline (Stages 1–3). Freeze and export the typed output contract early — this unblocks Plan 2b's Stage 4 work.
2. Apply the same Stage 1–3 pattern + personalization hook to `flashcard-agent.ts` and `short-bit-agent.ts`. Ship the engagement UI: flashcard self-rated recall + mastery write-back + spaced-repetition queue + shuffle + session summary; short-bit real image rendering + swipe gesture + streak tracking.
3. Refactor `assessment-agent.ts`: drop coding questions, deterministic MCQ grading, fix `nodeId` mismatch, add schema validation, add Stage 1–3 pattern + personalization hook.
4. Fold `story-agent.ts`'s narrative generation into Course as a generation mode per the narrative-mode spec; retire `/dashboard/story`. Delete `cheatsheet-agent.ts`, its route, and nav entry in this same pass.

## Approved decisions

- Content-type surface: Course, Flashcard, Short-bit as the three core types — approved.
- Assessment question types: MCQ + short answer + long answer only, coding dropped — approved.
- Cheatsheet: removed entirely — approved.
- Narrative mode: layout + tone shift to free-flowing, relax-and-read/listen prose (spec above) — approved.

## Open questions

None remaining.
