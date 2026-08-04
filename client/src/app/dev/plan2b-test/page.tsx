"use client";

import { useState } from "react";
import CourseDoubtChat from "@/components/CourseDoubtChat";
import CourseNarrator, { type VisemeCue } from "@/components/CourseNarrator";

// Throwaway QA harness for Plan 2b steps 1-3 — not part of the plan's
// committed deliverables, just a way to see the new components render
// before they're wired into the real course page (that's step 7).
// Delete this route once manual testing is done.

const PLACEHOLDER_AUDIO_URL = "/test-fixtures/placeholder-narration.wav";

// Fabricated timeline (no Rhubarb output available in this sandbox) —
// cycles through every viseme category every ~0.4s across the 4s
// placeholder tone, purely to exercise the mouth-shape mapping + the
// audio.currentTime-driven lookup in CourseNarrator's useFrame loop.
const FIXTURE_VISEMES: VisemeCue["viseme"][] = [
  "X",
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "H",
];
const FIXTURE_VISEME_TIMELINE: VisemeCue[] = Array.from(
  { length: 10 },
  (_, i) => ({
    start: i * 0.4,
    end: (i + 1) * 0.4,
    viseme: FIXTURE_VISEMES[i % FIXTURE_VISEMES.length],
  }),
);

export default function Plan2bTestPage() {
  const [courseId, setCourseId] = useState("");
  const [activeCourseId, setActiveCourseId] = useState<string | null>(null);

  return (
    <div
      style={{
        padding: 32,
        display: "flex",
        flexDirection: "column",
        gap: 40,
        maxWidth: 900,
        margin: "0 auto",
      }}
    >
      <div>
        <h1 style={{ fontSize: 20, marginBottom: 8 }}>Plan 2b QA harness</h1>
        <p style={{ fontSize: 13, color: "#666" }}>
          The bot below uses a placeholder warble tone + a hand-fabricated
          viseme timeline, not real Piper/Rhubarb output — for real narration,
          test via the actual course page instead. This only exercises the
          playback + mouth-sync plumbing.
        </p>
      </div>

      <section>
        <h2 style={{ fontSize: 16, marginBottom: 12 }}>CourseNarrator</h2>
        <div style={{ height: 450 }}>
          <CourseNarrator
            audioUrl={PLACEHOLDER_AUDIO_URL}
            visemeTimeline={FIXTURE_VISEME_TIMELINE}
            autoPlay={false}
          />
        </div>
      </section>

      <section>
        <h2 style={{ fontSize: 16, marginBottom: 12 }}>CourseDoubtChat</h2>
        <p style={{ fontSize: 13, color: "#666", marginBottom: 12 }}>
          Requires being logged into the app (JWT cookie) and a real courseId
          (task_id) that already has a saved course — generate one via the
          normal course flow, then paste its task id below. It reads/writes real
          Postgres rows and calls Gemini for real, live requests.
        </p>
        <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
          <input
            type="text"
            value={courseId}
            onChange={(e) => setCourseId(e.target.value)}
            placeholder="Paste a real course task_id (UUID)"
            style={{ flex: 1, padding: 8, border: "1px solid #ccc" }}
          />
          <button
            type="button"
            onClick={() => setActiveCourseId(courseId.trim() || null)}
            style={{
              padding: "8px 16px",
              border: "1px solid #333",
              background: "#333",
              color: "#fff",
            }}
          >
            Load
          </button>
        </div>
        {activeCourseId && <CourseDoubtChat courseId={activeCourseId} />}
      </section>
    </div>
  );
}
