# Plan 2b — Course Narration + 3D Avatar + Video Export + Doubt Chat

Split off from the original Plan 2 (`02-course-generation-agentic-video.md`, now an index pointing to this file and 2a) so this narration/video/avatar/chat work can be assigned to a different person than the content-pipeline refactor (Plan 2a), with minimal file overlap. See "Interference boundary with Plan 2a" below before touching any file this doc doesn't explicitly own.

## Branch strategy

Build on its own branch (e.g. `plan/02b-course-narration-video-chat`), off current `master`. **Most of this plan does not depend on Plan 2a merging** — the Piper TTS wrapper, Rhubarb wrapper, `<CourseNarrator>` avatar component build-out, and the doubt chatbox are all independent of Plan 2a's pipeline refactor (the chatbox only needs the *final* course JSON, not pipeline internals; the avatar/TTS/lip-sync work doesn't touch `course-agent.ts` at all until the final integration step below). **Only one step is a hard dependency**: wiring the Narration Writer (Stage 4) in as the step after Plan 2a's Stage 3 critic needs Plan 2a's typed Stage 1–3 output contract to exist. Sequence this plan's own work so that step is last (see Suggested sequencing), and coordinate with whoever has Plan 2a to get that contract frozen/exported early rather than waiting for all of 2a to merge.

Plan 5 does not depend on this plan (2b) — only on Plan 2a's profile-parameter slot. This plan can merge on its own timeline.

## Interference boundary with Plan 2a

- **This plan (2b) owns**: new files only — Piper service wrapper, Rhubarb service wrapper, `<CourseNarrator>` component (built from the existing `CogniBot.tsx` procedural-robot base — see notes below), narration-writer/TTS/viseme server actions, doubt-chat server action (`askCourseDoubt`), Remotion compositions + render job + download flow, and the course player page's narration/chat/video UI composition (mounting `<CourseNarrator>`, the chat sidebar, and the "Download as Video" button alongside `SlideViewer`).
- **Do not touch**: `course-agent.ts`'s Stage 1–3 internals, `flashcard-agent.ts`, `short-bit-agent.ts`, `assessment-agent.ts`, `SlideViewer.tsx`'s internals (read/reuse its render logic for Remotion compositions, don't modify it — narrative-mode rendering changes there are Plan 2a's).
- **One deliberate shared touchpoint**: this plan adds the "Listen" (Piper audio) buttons to `FlashcardStack.jsx`/`ShortBitStack.jsx` as a small fast-follow PR, landing *after* Plan 2a's version of those components has merged — sequenced, not concurrent, to avoid a live merge conflict on the same files.
- **Contract consumed from 2a**: the typed `CourseSkeleton`/`CourseSlide`/critic-result shapes that Plan 2a freezes and exports early in its own work. Build the Narration Writer step against those types; if 2a hasn't landed yet, stub the type locally and swap the import at integration time rather than blocking on 2a's full merge.

## Context: what exists today

`course-agent.ts` currently has no narration script, no TTS, no avatar, no video export, and no chat — this is a from-scratch feature build, not an enhancement of an existing partial system. `CourseSkeleton`/`CourseSlide` are rendered by `SlideViewer.tsx`.

There's an existing procedural 3D robot component, `CogniBot.tsx`, evaluated as a starting point for `<CourseNarrator>`:

- Built with React Three Fiber, procedural THREE primitives (RoundedBox/Sphere/Cylinder — no glTF, no morph targets/blendshapes). Has a working gesture state machine (`Teacher_*` animation cases: standing, explaining, thinking, pointing, emphasizing, listening, etc.) and UI chrome (dialogue box, pause button, tech-grid overlay) — both reusable.
- Current "speech" is the browser's Web Speech API (`speakText`, module-level `currentUtterance`) with a fake sine-wave mouth flap (`Math.sin(time * 20)`) — not driven by real audio or viseme data. This layer needs full replacement, not adaptation: swap Web Speech for real `<audio>` playback of Piper output, and drive the existing mouth-plane mesh's scale off a real Rhubarb viseme timeline instead of the sine fake. This is cheaper than building a glTF/ARKit-blendshape avatar from scratch — the mouth is a single scaled plane, not a rigged face — so map Rhubarb's viseme categories (A–H, X) to a handful of mouth-plane scale/shape variants rather than doing full blendshape morphing.
- Known bugs to fix regardless: module-level `currentUtterance` breaks with multiple instances mounted (move to component state); no `timerRef` cleanup on unmount (setState-after-unmount leak); `getVoices()` called without an `onvoiceschanged` handler (silently empty on first render — moot once Web Speech is replaced, but note if any fallback path keeps it); fragile `children[0] as any` material access on the eyes; dead `<Float>` with no children in `Environment`.

## Technology decisions (approved)

- **TTS**: Piper — self-hosted, free, open-source (MIT), fast (ONNX runtime, real-time on CPU, no GPU needed), runs as a local/server process, no per-call API cost. Rejected the browser's Web Speech API as primary path because it has no reliable cross-browser phoneme/word timing, which lip sync needs.
- **Lip sync**: Rhubarb Lip Sync — free, open-source, MIT-licensed CLI. Takes narration audio + the exact spoken text, outputs a viseme timeline (mouth-shape keyframes) without needing the TTS engine to expose phoneme timestamps itself. Decouples TTS choice from lip-sync choice.
- **Avatar**: keep `CogniBot.tsx`'s procedural-robot approach rather than switching to a glTF/Ready Player Me avatar — cheaper integration (viseme→mouth-plane-scale mapping vs. full blendshape rigging), and it's already built with a working gesture state machine. Land as `<CourseNarrator>`: takes `{ audioUrl, visemeTimeline }`, drives the mouth mesh off `audio.currentTime` during playback, mounts alongside `SlideViewer` in the course player.
- **Video export**: Remotion (server-/CLI-rendered, React-component-driven), not in-browser `MediaRecorder` (rejected — flaky across browsers, can't run as a background job, breaks if the tab is backgrounded). Each slide becomes a timed React composition reusing `SlideViewer`'s render logic (read-only reuse, per the interference boundary above); narration audio and the avatar's viseme-driven animation are composited per-frame; whole course renders to MP4 as a background job (queued server action → polling or webhook → download link). This is a genuinely async, potentially slow operation — do not render synchronously inside a server action request/response cycle.
- **Doubt chatbox**: no vector DB / embedding retrieval — a crash course capped at ~3 slides/topic-title fits comfortably in a single LLM context window, so retrieval would be over-engineering at this scale. New server action `askCourseDoubt(courseId, conversationHistory, question)`; system prompt = full course skeleton + all slide content as grounding context, explicit instruction to answer only within the course's scope; conversation history persisted per course session (reuse the existing Postgres/JWT auth pattern used elsewhere in `app/actions/`) so the chat has real multi-turn memory — this is the actual "LLM orchestration" requirement, not a stateless Q&A. If courses later grow to full curricula, revisit retrieval then — documented non-goal for now, not a silent gap.

## Architecture: Stages 4–7 (Course-exclusive)

```
Stage 4 — Narration Writer   generateNarrationScript(slide)    [new — separate from on-slide text]
Stage 5 — TTS Synthesis      synthesizeNarrationAudio(script)  [new — Piper]
Stage 6 — Viseme Extraction  extractVisemeTimeline(audio)      [new — Rhubarb]
Stage 7 — Video Render       renderCourseVideo(slides, audio,  [new — background job, Remotion]
                              visemes)
```

These stages are keyed to a slide-deck timeline, which is why they stay Course-exclusive — Flashcard/Short-bit don't have one (their Piper audio is a much simpler flat per-item playback, added as a fast-follow to Plan 2a's UI work, not a pipeline stage).

Note from Plan 2a: narrative-mode courses (Story folded into Course) should get a narration script that tracks close to on-slide prose (near-verbatim) rather than the more expansive spoken-script approach standard-mode courses get — branch the Stage 4 prompt on the course's `style` flag.

## Suggested sequencing

Ordered so the one hard dependency on Plan 2a (step 5) comes last — everything before it can start immediately, independent of Plan 2a's progress.

1. Piper TTS service wrapper (text → WAV, server-side) and Rhubarb wrapper (audio + text → viseme timeline). No dependency on Plan 2a.
2. Replace `CogniBot.tsx`'s Web Speech layer with real `<audio>` playback + viseme-driven mouth mapping; fix the known bugs listed above; land as `<CourseNarrator>` taking `{ audioUrl, visemeTimeline }`. No dependency on Plan 2a.
3. Doubt chatbox (`askCourseDoubt`, conversation persistence) — independent of both video work and Plan 2a, can ship any time.
4. Piper "Listen" button fast-follow to `FlashcardStack.jsx`/`ShortBitStack.jsx` — after Plan 2a's version of those components has merged (see Interference boundary).
5. Narration Writer (Stage 4), wired in as the step after Plan 2a's Stage 3 critic — **this is the step that needs Plan 2a's frozen Stage 1–3 output contract.** Build against a stubbed type first if 2a hasn't landed yet, swap the import at integration time.
6. TTS + viseme pipeline (Stages 5–6) wired end-to-end against real narration scripts, cached per slide.
7. `<CourseNarrator>` wired to real narration playback in the course player (load `3d-web-experience` skill here if further R3F work is needed beyond the existing CogniBot base).
8. Remotion video render job + download flow (load `remotion-best-practices` skill here) — last, since it depends on narration audio + avatar animation both being finished.

## Approved decisions

- TTS: Piper — approved.
- Lip sync: Rhubarb Lip Sync, feeding a viseme timeline into `<CourseNarrator>` — approved.
- Avatar base: keep `CogniBot.tsx`'s procedural-robot approach, adapt its mouth mesh to real visemes rather than switching to a glTF/blendshape avatar — approved by extension of the CogniBot review, not yet explicitly re-confirmed by the user for this exact split doc — flag if a different avatar direction is wanted.
- Video export: Remotion background-render job, audio track = Piper narration composited per-frame with slides + avatar animation — approved.
- Doubt chatbox: no retrieval/vector DB, full-course-JSON-as-context, persisted multi-turn history — approved.

## Open questions

None blocking start. One re-confirm suggested: the avatar-base decision above (keep CogniBot vs. build a glTF/Ready Player Me avatar) was reasoned through in the CogniBot review but should get an explicit yes/no before step 2 starts, since it's a bigger effort swing than the other approved items.
