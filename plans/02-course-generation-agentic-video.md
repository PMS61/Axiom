# Plan 2 — Course Generation: PPT + 3D Narration + Video Export + Doubt Chat

## Branch strategy

Build this plan on its own branch (e.g. `plan/02-course-agentic-video`), off current `master` — Plan 4 (personalization) is already merged there, so `toCoursePersonalizationPayload()` (`lib/personalization.ts`) is available from the start of this branch. Independent of Plans 1 and 3; no need to wait on either. Plan 5 depends on this plan merging first (see Plan 5's own branch note).

## Context: what exists today

`course-agent.ts` is two flat, sequential Gemini calls with no orchestration:

1. `generateCourseSkeleton()` — one LLM call, title/description/userPrompt in, a flat array of `{title, description, type}` slide specs out (max 3 slides per topic-title, types drawn from `slide-prompts.ts`'s 12 registered slide kinds: title, section, content, image, mermaid, quote, code, content-and-image, content-and-code, table, list, conclusion).
2. `generateCourseSlide()` — one LLM call per skeleton entry (run in `Promise.all`, so "parallel" but not orchestrated — no shared state, no critique, no retry-with-feedback), rendered by `SlideViewer.tsx`.

There is no personalization input (no user profile, no mastery, no learning style — confirmed absent from `course-agent.ts`'s signature, unlike `roadmap-agent.ts` which does take a profile). There is no narration script, no TTS, no avatar, no video export, and no chat. This is a from-scratch feature build, not an enhancement of an existing partial system.

The project's original research brief (superseded by this `plans/` directory, no longer in the repo) already scoped "How" pillar work as an engineering showcase (not the research track) and named the exact techniques to demonstrate: self-refinement loop (Reflexion-style critic), model routing by task complexity, adaptive in-context learning from user history, and performance-aware generation. This plan operationalizes those for the course feature specifically, since it's the one getting the video/avatar/chat treatment.

## Scope clarification

"Only be PPTs" is read as: **the deliverable format for Course stays a slide deck** (it already is, via `CourseSkeleton`/`CourseSlide`/`SlideViewer`) — this is not a request to remove the other content generators (flashcard, cheatsheet, short-bit, story stay as-is, untouched by this plan). What's new is bolted onto the existing slide deck: narration, video export, and a chat sidebar. Confirm this reading with the user before implementation if there's any ambiguity about deprecating the other agents.

## Architecture: formalize the agent pipeline

Replace the two-call flat structure with an explicit staged pipeline (still hand-rolled TypeScript — no need for a heavyweight framework like LangGraph at this scale, but each stage gets a typed input/output contract instead of being an inline function call):

```
Stage 1 — Planner            generateCourseSkeleton()          [existing, keep]
Stage 2 — Slide Generator    generateCourseSlide() per slide   [existing, keep, but add stage 3 loop]
Stage 3 — Critic / Refiner   scoreSlideAgainstRubric() →       [new — Reflexion-style]
                              regenerate if below threshold
Stage 4 — Narration Writer   generateNarrationScript(slide)    [new — separate from on-slide text]
Stage 5 — TTS Synthesis      synthesizeNarrationAudio(script)  [new]
Stage 6 — Viseme Extraction  extractVisemeTimeline(audio)      [new]
Stage 7 — Video Render       renderCourseVideo(slides, audio,  [new — background job]
                              visemes)
```

Stage 3 (critic) and model routing are the direct implementation of the brief's "Self-refinement loop" and "Model routing" showcase items — a cheap/fast model call scores each generated slide against a short rubric (clarity, factual plausibility, matches description, appropriate slide-type usage) and triggers one regeneration if below threshold, capped at 1 retry to bound cost. This is an LLM-judge quality gate, explicitly not human-testing evidence (keep that distinction in any research-adjacent writeup, per the brief's own caveat).

## 3D bot narration — technology recommendation

Requirements: free TTS, an avatar that visibly speaks in sync, must work inside the existing Next.js/React app.

- **TTS**: use a self-hosted open-source engine (Piper TTS is the concrete recommendation — fast, free, runs as a local/server process, good voice quality, no per-call API cost) rather than the browser's Web Speech API. Web Speech API is free but has no reliable cross-browser way to get phoneme/word timing, which is needed for lip sync — this would force a much cruder amplitude-only mouth animation. Piper generates a WAV file server-side that can be post-processed for timing.
- **Lip sync**: use **Rhubarb Lip Sync** (free, open-source, MIT-licensed CLI) — it takes any narration audio + the exact text and outputs a viseme timeline (mouth-shape keyframes) without needing the TTS engine itself to expose phoneme timestamps. This decouples the TTS choice from the lip-sync choice and is the standard approach for this exact problem (narrated-avatar video tools use this pairing).
- **Avatar**: a free Ready Player Me glTF avatar (or an equivalent free-to-use rigged glTF with ARKit-style blendshapes) rendered via **React Three Fiber** (the `3d-web-experience` skill is exactly for this — load it when implementing this stage). Blendshape morph targets are driven frame-by-frame from the Rhubarb viseme timeline during playback.
- Land this as a `<CourseNarrator>` component: mounts a `<Canvas>` (R3F) with the avatar, takes `{ audioUrl, visemeTimeline }`, drives morph targets off `audio.currentTime` during playback, sits alongside `SlideViewer` in the course player.

## Video export — technology recommendation

Two options were weighed:

| Approach | Verdict |
|---|---|
| `MediaRecorder` capturing the live DOM/canvas playback in-browser | **Reject as primary path** — flaky across browsers, quality/frame-rate depends on the user's machine, can't run as a background job, breaks if the tab is backgrounded. |
| **Remotion** (server-/CLI-rendered, React-component-driven video) | **Recommended** — the `remotion-best-practices` skill is already available for this. Each slide becomes a timed React composition (reusing `SlideViewer`'s render logic), narration audio and the avatar's viseme-driven animation are composited per-frame, and the whole course renders to MP4 as a background job (queued server action → polling or webhook → download link). Deterministic output, works headless, matches "download as video" cleanly. |

Land as: a "Download as Video" button on the course page that enqueues a render job (course id + generated narration + viseme data as input), polls status, and serves the resulting MP4 once done. This is a genuinely async, potentially slow operation — do not attempt to render synchronously inside a server action request/response cycle.

## Doubt chatbox — technology recommendation

Given a single course's total slide text is small (a crash course capped at ~3 slides/topic-title), the entire generated course JSON fits comfortably in a single LLM context window. **No vector DB / embedding retrieval is needed at this scale** — over-engineering it would contradict the "don't design for hypothetical future requirements" principle. Implementation:

- New server action `askCourseDoubt(courseId, conversationHistory, question)`.
- System prompt = the full course skeleton + all slide content (title, bullets, code, etc.) as grounding context, explicit instruction to answer only within the course's scope and say so when a question is out of scope.
- Conversation history persisted per course session (reuse the existing Postgres/JWT auth pattern already used elsewhere in `app/actions/`) so the chat has memory across turns — this is the actual "LLM orchestration" requirement: multi-turn state, not a stateless Q&A.
- If a course is later allowed to be much larger (full curriculum, not a crash course), *then* revisit with retrieval — flag this as a documented non-goal for now, not a silent gap.

## Personalization hook (ties to Plan 4 — already shipped)

Plan 4 is merged: `getLearnerProfile(userId)` and `toCoursePersonalizationPayload(profile, topic)` (`lib/personalization.ts`) already exist and return `{experienceLevel, learningStyle, masteryScore}` for a given topic. `generateCourseSkeleton`/`generateCourseSlide` should accept this payload and adjust depth/pacing/vocabulary per the brief's "Performance-Aware Generation" item — this plan's pipeline stages are designed so that hook slots in at Stage 1/2 without restructuring (the payload becomes another input alongside title/description/userPrompt). This is real wiring now, not a future dependency — do it as part of Stage 1/2's build, not as a follow-up.

## Suggested sequencing

1. Refactor `course-agent.ts` into the explicit staged pipeline (Stages 1–3: planner, generator, critic) — no video/avatar yet, ships value alone as better slide quality.
2. Narration Writer (Stage 4) — per-slide spoken-script generation, distinct field from on-slide bullet text.
3. TTS + viseme pipeline (Stages 5–6) — Piper + Rhubarb, server-side, cached per slide.
4. `<CourseNarrator>` R3F avatar component wired to narration playback (load `3d-web-experience` skill here).
5. Doubt chatbox — independent of video work, can ship in parallel with step 3–4.
6. Remotion video render job + download flow (load `remotion-best-practices` skill here) — last, since it depends on narration audio + avatar animation both being finished.

## Open questions for the user

- Confirm "only be PPTs" doesn't mean deprecating flashcard/cheatsheet/short-bit/story agents.
- Confirm Piper + Rhubarb (self-hosted, free, some server setup) is acceptable vs. a paid API (e.g., ElevenLabs + a hosted lip-sync service) that would be less setup but not free.
- Confirm Remotion background-rendering is acceptable given it needs a job queue/worker, not just a server action.
