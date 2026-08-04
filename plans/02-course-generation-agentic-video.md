# Plan 2 — Course Generation: PPT + 3D Narration + Video Export + Doubt Chat (index)

**This plan has been split into two independently-assignable plans**, so the work can go to two different people in parallel with minimal file overlap:

- **[Plan 2a — Content Pipeline Refactor](02a-content-pipeline-personalization.md)**: Course/Flashcard/Short-bit/Assessment staged pipeline (planner/generator/critic), personalization wiring, Flashcard/Short-bit engagement UI (mastery write-back, spaced repetition, shuffle, session summary, real images, swipe, streak), Story→Course narrative-mode fold-in, Cheatsheet removal. This is the plan Plan 5 depends on (its profile-parameter-slot dependency).
- **[Plan 2b — Course Narration + Avatar + Video + Chat](02b-course-narration-video-chat.md)**: Narration Writer (Stage 4), Piper TTS + Rhubarb viseme pipeline (Stages 5–6), `<CourseNarrator>` avatar (adapting the existing `CogniBot.tsx`), Remotion video export (Stage 7), doubt chatbox. Independent of Plan 2a except for one integration step (wiring Stage 4 in after 2a's Stage 3 critic) — see 2b's own doc for how that dependency is sequenced to avoid blocking.

Read each doc's own "Interference boundary" section before starting — it defines exactly which files each plan owns, and the one deliberate shared touchpoint (Flashcard/Short-bit's Piper "Listen" button, added by 2b as a fast-follow after 2a's component work merges).

Original content-type audit, scope decisions (3 core types: Course/Flashcard/Short-bit, Story folded into Course, Cheatsheet removed, Assessment refactored, coding questions dropped), and all approved technology choices (Piper, Rhubarb, Remotion, no-retrieval doubt chat) now live in the two split docs above, split by which workstream they belong to.
