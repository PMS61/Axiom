/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Profile Client
   Displays all onboarding data with inline editing for
   sleep/wake, energy windows, and availability blocks.
   ═══════════════════════════════════════════════════════════ */

"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useState } from "react";
import { updateUserProfile } from "@/app/actions/auth";
import Header from "@/components/Header";
import { AppProvider } from "@/lib/store";

/* ── Label maps ─────────────────────────────────────────── */

const ROLE_OPTIONS = [
  "Student",
  "Researcher",
  "Professional",
  "Self-learner",
  "Other",
];

const SESSION_STYLE_OPTIONS = [
  "Long deep blocks (90+ min)",
  "Medium sprints (45-60 min)",
  "Short bursts (20-30 min)",
  "Mix it up",
];

const SWITCH_BUFFER_OPTIONS = [
  "Jump right in (0-5 min)",
  "Quick shift (10-15 min)",
  "Proper break (20-30 min)",
  "Finish one first",
];

const DEADLINE_STYLE_OPTIONS = [
  "Finish early",
  "Work steadily",
  "Work under pressure",
];

const EXPERIENCE_LEVEL_OPTIONS: Array<{
  value: "beginner" | "intermediate" | "advanced";
  label: string;
}> = [
  { value: "beginner", label: "Beginner" },
  { value: "intermediate", label: "Intermediate" },
  { value: "advanced", label: "Advanced" },
];

const LEARNING_STYLE_OPTIONS: Array<{
  value: "visual" | "reading" | "practice" | "balanced";
  label: string;
}> = [
  { value: "visual", label: "Visual (diagrams, video)" },
  { value: "reading", label: "Reading (text, docs)" },
  { value: "practice", label: "Practice (hands-on)" },
  { value: "balanced", label: "Balanced mix" },
];

const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/* ── Helpers ────────────────────────────────────────────── */

type TimeWindow = { start_min: number; end_min: number };
type FixedBlock = {
  title: string;
  start_min: number;
  end_min: number;
  days: number[];
  recurring: boolean;
};

function formatTime(minutes: number | null | undefined): string {
  if (minutes === null || minutes === undefined) return "N/A";
  const normalized = ((Math.floor(minutes) % 1440) + 1440) % 1440;
  const h = Math.floor(normalized / 60)
    .toString()
    .padStart(2, "0");
  const m = (normalized % 60).toString().padStart(2, "0");
  return `${h}:${m}`;
}

function minutesFromTimeString(s: string): number {
  const [h, m] = s.split(":").map(Number);
  return (h || 0) * 60 + (m || 0);
}

function timeInputValue(minutes: number | null | undefined): string {
  if (minutes === null || minutes === undefined) return "07:00";
  const normalized = ((Math.floor(minutes) % 1440) + 1440) % 1440;
  const h = Math.floor(normalized / 60)
    .toString()
    .padStart(2, "0");
  const m = (normalized % 60).toString().padStart(2, "0");
  return `${h}:${m}`;
}

/* ── Styles ─────────────────────────────────────────────── */

const cardStyle: React.CSSProperties = {
  background: "var(--card-bg)",
  border: "0.5px solid var(--rule)",
  padding: 32,
  display: "grid",
  gap: 24,
};

const sectionHeadingStyle: React.CSSProperties = {
  marginBottom: 24,
  paddingBottom: 8,
  borderBottom: "0.5px solid var(--rule)",
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
};

const fieldLabelStyle: React.CSSProperties = {
  fontSize: 12,
  color: "var(--muted)",
  marginBottom: 4,
};
const fieldValueStyle: React.CSSProperties = { fontWeight: 600 };

const chipStyle: React.CSSProperties = {
  display: "inline-block",
  fontSize: 10,
  fontFamily: "var(--mono)",
  letterSpacing: "0.08em",
  textTransform: "uppercase" as const,
  borderWidth: "0.5px",
  borderStyle: "solid",
  borderColor: "var(--rule)",
  padding: "3px 10px",
  marginRight: 6,
  marginBottom: 6,
};

const chipActiveStyle: React.CSSProperties = {
  ...chipStyle,
  background: "var(--ink)",
  color: "var(--bg)",
  borderColor: "var(--ink)",
  cursor: "pointer",
};

const chipInactiveStyle: React.CSSProperties = {
  ...chipStyle,
  cursor: "pointer",
};

const inputBoxStyle: React.CSSProperties = {
  fontFamily: "var(--mono)",
  fontSize: 13,
  fontVariantNumeric: "tabular-nums",
  background: "var(--card-bg)",
  border: "0.5px solid var(--rule)",
  padding: "8px 12px",
  outline: "none",
  width: "100%",
};

/* ── Component ──────────────────────────────────────────── */

export default function ProfileClient({ user }: { user: any }) {
  const router = useRouter();

  // --- Wake / Sleep editing ---
  const [editingSleep, setEditingSleep] = useState(false);
  const [wakeTime, setWakeTime] = useState<number>(user.wake_time ?? 420);
  const [sleepTime, setSleepTime] = useState<number>(user.sleep_time ?? 1380);

  // --- Peak / Low windows ---
  const [editingPeak, setEditingPeak] = useState(false);
  const rawPeak =
    typeof user.peak_focus_windows === "string"
      ? JSON.parse(user.peak_focus_windows)
      : (user.peak_focus_windows ?? []);
  const [peakWindows, setPeakWindows] = useState<TimeWindow[]>(rawPeak);

  const [editingLow, setEditingLow] = useState(false);
  const rawLow =
    typeof user.low_energy_windows === "string"
      ? JSON.parse(user.low_energy_windows)
      : (user.low_energy_windows ?? []);
  const [lowWindows, setLowWindows] = useState<TimeWindow[]>(rawLow);

  // --- Learning preferences (experience level, learning style) ---
  const [editingLearning, setEditingLearning] = useState(false);
  const [experienceLevel, setExperienceLevel] = useState<
    "beginner" | "intermediate" | "advanced"
  >(user.experience_level ?? "beginner");
  const [learningStyle, setLearningStyle] = useState<
    "visual" | "reading" | "practice" | "balanced"
  >(user.learning_style ?? "balanced");

  // --- Fixed commitments ---
  const [editingBlocks, setEditingBlocks] = useState(false);
  const rawBlocks =
    typeof user.fixed_commitments === "string"
      ? JSON.parse(user.fixed_commitments)
      : (user.fixed_commitments ?? []);
  const [fixedBlocks, setFixedBlocks] = useState<FixedBlock[]>(rawBlocks);

  // --- Saving state ---
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState("");

  /* ── Save helpers ─────────────────────────────────────── */

  const saveSleepWake = useCallback(async () => {
    setSaving(true);
    setSaveMsg("");
    const res = await updateUserProfile({
      wake_time: wakeTime,
      sleep_time: sleepTime,
    });
    setSaving(false);
    if (res.error) {
      setSaveMsg(res.error);
      return;
    }
    setEditingSleep(false);
    router.refresh();
  }, [wakeTime, sleepTime, router]);

  const saveLearningPrefs = useCallback(async () => {
    setSaving(true);
    setSaveMsg("");
    const res = await updateUserProfile({
      experience_level: experienceLevel,
      learning_style: learningStyle,
    });
    setSaving(false);
    if (res.error) {
      setSaveMsg(res.error);
      return;
    }
    setEditingLearning(false);
    router.refresh();
  }, [experienceLevel, learningStyle, router]);

  const savePeakWindows = useCallback(async () => {
    setSaving(true);
    setSaveMsg("");
    const res = await updateUserProfile({ peak_focus_windows: peakWindows });
    setSaving(false);
    if (res.error) {
      setSaveMsg(res.error);
      return;
    }
    setEditingPeak(false);
    router.refresh();
  }, [peakWindows, router]);

  const saveLowWindows = useCallback(async () => {
    setSaving(true);
    setSaveMsg("");
    const res = await updateUserProfile({ low_energy_windows: lowWindows });
    setSaving(false);
    if (res.error) {
      setSaveMsg(res.error);
      return;
    }
    setEditingLow(false);
    router.refresh();
  }, [lowWindows, router]);

  const saveFixedBlocks = useCallback(async () => {
    setSaving(true);
    setSaveMsg("");
    const res = await updateUserProfile({ fixed_commitments: fixedBlocks });
    setSaving(false);
    if (res.error) {
      setSaveMsg(res.error);
      return;
    }
    setEditingBlocks(false);
    router.refresh();
  }, [fixedBlocks, router]);

  /* ── Render ───────────────────────────────────────────── */

  return (
    <div style={{ minHeight: "100vh" }}>
      <AppProvider>
        <Header />
      </AppProvider>

      <main className="container" style={{ paddingTop: 60, paddingBottom: 80 }}>
        <div style={{ maxWidth: 800, margin: "0 auto" }}>
          {/* Hero */}
          <div className="meta-text" style={{ marginBottom: 16 }}>
            USER PROFILE
          </div>
          <h1 style={{ fontSize: 42, marginBottom: 8 }}>{user.name}</h1>
          <p style={{ color: "var(--muted)", marginBottom: 40, fontSize: 16 }}>
            {user.email}
          </p>

          {saveMsg && (
            <div
              style={{
                background: "var(--vermillion)",
                color: "var(--bg)",
                padding: "10px 16px",
                marginBottom: 24,
                fontFamily: "var(--mono)",
                fontSize: 12,
              }}
            >
              {saveMsg}
            </div>
          )}

          <div style={{ display: "grid", gap: 32 }}>
            {/* ─── General Information ─── */}
            <section style={cardStyle}>
              <div>
                <h3 className="meta-text" style={sectionHeadingStyle}>
                  General Information
                </h3>
                <div
                  className="responsive-2col"
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 24,
                  }}
                >
                  <div>
                    <div style={fieldLabelStyle}>Role</div>
                    <div style={fieldValueStyle}>
                      {ROLE_OPTIONS[user.role] || "Unknown"}
                    </div>
                  </div>
                  <div>
                    <div style={fieldLabelStyle}>Timezone</div>
                    <div style={fieldValueStyle}>
                      {user.timezone || "Not set"}
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* ─── Learning Preferences (editable) ─── */}
            <section style={cardStyle}>
              <div>
                <h3 className="meta-text" style={sectionHeadingStyle}>
                  <span>Learning Preferences</span>
                  {!editingLearning ? (
                    <button
                      className="btn btn-sm"
                      onClick={() => setEditingLearning(true)}
                    >
                      Edit
                    </button>
                  ) : (
                    <div style={{ display: "flex", gap: 8 }}>
                      <button
                        className="btn btn-sm btn-primary"
                        onClick={saveLearningPrefs}
                        disabled={saving}
                      >
                        {saving ? "Saving…" : "Save"}
                      </button>
                      <button
                        className="btn btn-sm"
                        onClick={() => {
                          setEditingLearning(false);
                          setExperienceLevel(
                            user.experience_level ?? "beginner",
                          );
                          setLearningStyle(user.learning_style ?? "balanced");
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  )}
                </h3>
                <div
                  className="responsive-2col"
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 24,
                  }}
                >
                  <div>
                    <div style={fieldLabelStyle}>Experience Level</div>
                    {editingLearning ? (
                      <select
                        value={experienceLevel}
                        onChange={(e) =>
                          setExperienceLevel(
                            e.target.value as typeof experienceLevel,
                          )
                        }
                        style={inputBoxStyle}
                      >
                        {EXPERIENCE_LEVEL_OPTIONS.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <div style={fieldValueStyle}>
                        {
                          EXPERIENCE_LEVEL_OPTIONS.find(
                            (o) =>
                              o.value === (user.experience_level ?? "beginner"),
                          )?.label
                        }
                      </div>
                    )}
                  </div>
                  <div>
                    <div style={fieldLabelStyle}>Learning Style</div>
                    {editingLearning ? (
                      <select
                        value={learningStyle}
                        onChange={(e) =>
                          setLearningStyle(
                            e.target.value as typeof learningStyle,
                          )
                        }
                        style={inputBoxStyle}
                      >
                        {LEARNING_STYLE_OPTIONS.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <div style={fieldValueStyle}>
                        {
                          LEARNING_STYLE_OPTIONS.find(
                            (o) =>
                              o.value === (user.learning_style ?? "balanced"),
                          )?.label
                        }
                      </div>
                    )}
                  </div>
                </div>
                <div
                  style={{ marginTop: 12, fontSize: 12, color: "var(--muted)" }}
                >
                  Used to personalize roadmap generation, and — as content
                  generators and trend ranking pick this up — course
                  depth/pacing and topic recommendations.
                </div>
              </div>
            </section>

            {/* ─── Wake / Sleep (editable) ─── */}
            <section style={cardStyle}>
              <div>
                <h3 className="meta-text" style={sectionHeadingStyle}>
                  <span>Sleep / Wake Cycle</span>
                  {!editingSleep ? (
                    <button
                      className="btn btn-sm"
                      onClick={() => setEditingSleep(true)}
                    >
                      Edit
                    </button>
                  ) : (
                    <div style={{ display: "flex", gap: 8 }}>
                      <button
                        className="btn btn-sm btn-primary"
                        onClick={saveSleepWake}
                        disabled={saving}
                      >
                        {saving ? "Saving…" : "Save"}
                      </button>
                      <button
                        className="btn btn-sm"
                        onClick={() => {
                          setEditingSleep(false);
                          setWakeTime(user.wake_time ?? 420);
                          setSleepTime(user.sleep_time ?? 1380);
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  )}
                </h3>
                <div
                  className="responsive-2col"
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 24,
                  }}
                >
                  <div>
                    <div style={fieldLabelStyle}>Wake Time</div>
                    {editingSleep ? (
                      <input
                        type="time"
                        value={timeInputValue(wakeTime)}
                        onChange={(e) =>
                          setWakeTime(minutesFromTimeString(e.target.value))
                        }
                        style={inputBoxStyle}
                      />
                    ) : (
                      <div style={fieldValueStyle}>
                        {formatTime(user.wake_time)}
                      </div>
                    )}
                  </div>
                  <div>
                    <div style={fieldLabelStyle}>Sleep Time</div>
                    {editingSleep ? (
                      <input
                        type="time"
                        value={timeInputValue(sleepTime)}
                        onChange={(e) =>
                          setSleepTime(minutesFromTimeString(e.target.value))
                        }
                        style={inputBoxStyle}
                      />
                    ) : (
                      <div style={fieldValueStyle}>
                        {formatTime(user.sleep_time)}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </section>

            {/* ─── Peak Focus Windows (editable) ─── */}
            <section style={cardStyle}>
              <div>
                <h3 className="meta-text" style={sectionHeadingStyle}>
                  <span>Peak Focus Windows</span>
                  {!editingPeak ? (
                    <button
                      className="btn btn-sm"
                      onClick={() => setEditingPeak(true)}
                    >
                      Edit
                    </button>
                  ) : (
                    <div style={{ display: "flex", gap: 8 }}>
                      <button
                        className="btn btn-sm btn-primary"
                        onClick={savePeakWindows}
                        disabled={saving}
                      >
                        {saving ? "Saving…" : "Save"}
                      </button>
                      <button
                        className="btn btn-sm"
                        onClick={() => {
                          setEditingPeak(false);
                          setPeakWindows(rawPeak);
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  )}
                </h3>

                {peakWindows.length === 0 && !editingPeak && (
                  <p style={{ color: "var(--muted)", fontSize: 12 }}>
                    No peak focus windows configured.
                  </p>
                )}

                {peakWindows.map((w, i) => (
                  <div
                    key={i}
                    style={{
                      display: "grid",
                      gridTemplateColumns: editingPeak
                        ? "1fr 1fr auto"
                        : "1fr 1fr",
                      gap: 16,
                      marginBottom: 12,
                      alignItems: "end",
                    }}
                  >
                    <div>
                      <div style={fieldLabelStyle}>Start</div>
                      {editingPeak ? (
                        <input
                          type="time"
                          value={timeInputValue(w.start_min)}
                          onChange={(e) => {
                            const next = [...peakWindows];
                            next[i] = {
                              ...next[i],
                              start_min: minutesFromTimeString(e.target.value),
                            };
                            setPeakWindows(next);
                          }}
                          style={inputBoxStyle}
                        />
                      ) : (
                        <div style={fieldValueStyle}>
                          {formatTime(w.start_min)}
                        </div>
                      )}
                    </div>
                    <div>
                      <div style={fieldLabelStyle}>End</div>
                      {editingPeak ? (
                        <input
                          type="time"
                          value={timeInputValue(w.end_min)}
                          onChange={(e) => {
                            const next = [...peakWindows];
                            next[i] = {
                              ...next[i],
                              end_min: minutesFromTimeString(e.target.value),
                            };
                            setPeakWindows(next);
                          }}
                          style={inputBoxStyle}
                        />
                      ) : (
                        <div style={fieldValueStyle}>
                          {formatTime(w.end_min)}
                        </div>
                      )}
                    </div>
                    {editingPeak && (
                      <button
                        className="btn btn-sm btn-danger"
                        onClick={() =>
                          setPeakWindows(peakWindows.filter((_, j) => j !== i))
                        }
                      >
                        ×
                      </button>
                    )}
                  </div>
                ))}

                {editingPeak && peakWindows.length < 2 && (
                  <button
                    className="btn btn-sm"
                    style={{ marginTop: 8 }}
                    onClick={() =>
                      setPeakWindows([
                        ...peakWindows,
                        { start_min: 540, end_min: 660 },
                      ])
                    }
                  >
                    + Add Window
                  </button>
                )}
              </div>
            </section>

            {/* ─── Low Energy Windows (editable) ─── */}
            <section style={cardStyle}>
              <div>
                <h3 className="meta-text" style={sectionHeadingStyle}>
                  <span>Low Energy Windows</span>
                  {!editingLow ? (
                    <button
                      className="btn btn-sm"
                      onClick={() => setEditingLow(true)}
                    >
                      Edit
                    </button>
                  ) : (
                    <div style={{ display: "flex", gap: 8 }}>
                      <button
                        className="btn btn-sm btn-primary"
                        onClick={saveLowWindows}
                        disabled={saving}
                      >
                        {saving ? "Saving…" : "Save"}
                      </button>
                      <button
                        className="btn btn-sm"
                        onClick={() => {
                          setEditingLow(false);
                          setLowWindows(rawLow);
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  )}
                </h3>

                {lowWindows.length === 0 && !editingLow && (
                  <p style={{ color: "var(--muted)", fontSize: 12 }}>
                    No low energy windows configured.
                  </p>
                )}

                {lowWindows.map((w, i) => (
                  <div
                    key={i}
                    style={{
                      display: "grid",
                      gridTemplateColumns: editingLow
                        ? "1fr 1fr auto"
                        : "1fr 1fr",
                      gap: 16,
                      marginBottom: 12,
                      alignItems: "end",
                    }}
                  >
                    <div>
                      <div style={fieldLabelStyle}>Start</div>
                      {editingLow ? (
                        <input
                          type="time"
                          value={timeInputValue(w.start_min)}
                          onChange={(e) => {
                            const next = [...lowWindows];
                            next[i] = {
                              ...next[i],
                              start_min: minutesFromTimeString(e.target.value),
                            };
                            setLowWindows(next);
                          }}
                          style={inputBoxStyle}
                        />
                      ) : (
                        <div style={fieldValueStyle}>
                          {formatTime(w.start_min)}
                        </div>
                      )}
                    </div>
                    <div>
                      <div style={fieldLabelStyle}>End</div>
                      {editingLow ? (
                        <input
                          type="time"
                          value={timeInputValue(w.end_min)}
                          onChange={(e) => {
                            const next = [...lowWindows];
                            next[i] = {
                              ...next[i],
                              end_min: minutesFromTimeString(e.target.value),
                            };
                            setLowWindows(next);
                          }}
                          style={inputBoxStyle}
                        />
                      ) : (
                        <div style={fieldValueStyle}>
                          {formatTime(w.end_min)}
                        </div>
                      )}
                    </div>
                    {editingLow && (
                      <button
                        className="btn btn-sm btn-danger"
                        onClick={() =>
                          setLowWindows(lowWindows.filter((_, j) => j !== i))
                        }
                      >
                        ×
                      </button>
                    )}
                  </div>
                ))}

                {editingLow && lowWindows.length < 2 && (
                  <button
                    className="btn btn-sm"
                    style={{ marginTop: 8 }}
                    onClick={() =>
                      setLowWindows([
                        ...lowWindows,
                        { start_min: 840, end_min: 900 },
                      ])
                    }
                  >
                    + Add Window
                  </button>
                )}
              </div>
            </section>

            {/* ─── Fixed Commitments / Availability Blocks (editable) ─── */}
            <section style={cardStyle}>
              <div>
                <h3 className="meta-text" style={sectionHeadingStyle}>
                  <span>Availability Blocks</span>
                  {!editingBlocks ? (
                    <button
                      className="btn btn-sm"
                      onClick={() => setEditingBlocks(true)}
                    >
                      Edit
                    </button>
                  ) : (
                    <div style={{ display: "flex", gap: 8 }}>
                      <button
                        className="btn btn-sm btn-primary"
                        onClick={saveFixedBlocks}
                        disabled={saving}
                      >
                        {saving ? "Saving…" : "Save"}
                      </button>
                      <button
                        className="btn btn-sm"
                        onClick={() => {
                          setEditingBlocks(false);
                          setFixedBlocks(rawBlocks);
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  )}
                </h3>

                {fixedBlocks.length === 0 && !editingBlocks && (
                  <p style={{ color: "var(--muted)", fontSize: 12 }}>
                    No availability blocks configured.
                  </p>
                )}

                {fixedBlocks.map((block, i) => (
                  <div
                    key={i}
                    style={{
                      border: "0.5px solid var(--rule)",
                      padding: 20,
                      marginBottom: 16,
                    }}
                  >
                    {editingBlocks ? (
                      <div style={{ display: "grid", gap: 16 }}>
                        <div>
                          <div style={fieldLabelStyle}>Block Name</div>
                          <input
                            type="text"
                            value={block.title}
                            onChange={(e) => {
                              const next = [...fixedBlocks];
                              next[i] = { ...next[i], title: e.target.value };
                              setFixedBlocks(next);
                            }}
                            placeholder="e.g. Morning Commute"
                            style={{
                              ...inputBoxStyle,
                              borderBottom: "0.5px solid var(--rule)",
                            }}
                          />
                        </div>
                        <div
                          className="responsive-2col"
                          style={{
                            display: "grid",
                            gridTemplateColumns: "1fr 1fr",
                            gap: 16,
                          }}
                        >
                          <div>
                            <div style={fieldLabelStyle}>Start</div>
                            <input
                              type="time"
                              value={timeInputValue(block.start_min)}
                              onChange={(e) => {
                                const next = [...fixedBlocks];
                                next[i] = {
                                  ...next[i],
                                  start_min: minutesFromTimeString(
                                    e.target.value,
                                  ),
                                };
                                setFixedBlocks(next);
                              }}
                              style={inputBoxStyle}
                            />
                          </div>
                          <div>
                            <div style={fieldLabelStyle}>End</div>
                            <input
                              type="time"
                              value={timeInputValue(block.end_min)}
                              onChange={(e) => {
                                const next = [...fixedBlocks];
                                next[i] = {
                                  ...next[i],
                                  end_min: minutesFromTimeString(
                                    e.target.value,
                                  ),
                                };
                                setFixedBlocks(next);
                              }}
                              style={inputBoxStyle}
                            />
                          </div>
                        </div>
                        <div>
                          <div style={fieldLabelStyle}>Days</div>
                          <div
                            style={{
                              display: "flex",
                              flexWrap: "wrap",
                              gap: 4,
                              marginTop: 4,
                            }}
                          >
                            {DAYS.map((day, dayIdx) => (
                              <span
                                key={dayIdx}
                                style={
                                  block.days.includes(dayIdx)
                                    ? chipActiveStyle
                                    : chipInactiveStyle
                                }
                                onClick={() => {
                                  const next = [...fixedBlocks];
                                  const days = block.days.includes(dayIdx)
                                    ? block.days.filter((d) => d !== dayIdx)
                                    : [...block.days, dayIdx].sort(
                                        (a, b) => a - b,
                                      );
                                  next[i] = { ...next[i], days };
                                  setFixedBlocks(next);
                                }}
                              >
                                {day}
                              </span>
                            ))}
                          </div>
                        </div>
                        <button
                          className="btn btn-sm btn-danger"
                          onClick={() =>
                            setFixedBlocks(
                              fixedBlocks.filter((_, j) => j !== i),
                            )
                          }
                        >
                          Remove Block
                        </button>
                      </div>
                    ) : (
                      <div>
                        <div
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            marginBottom: 8,
                          }}
                        >
                          <div style={{ fontWeight: 700 }}>
                            {block.title || "Untitled"}
                          </div>
                          <span className="meta-text">
                            {formatTime(block.start_min)} –{" "}
                            {formatTime(block.end_min)}
                          </span>
                        </div>
                        <div
                          style={{ display: "flex", flexWrap: "wrap", gap: 4 }}
                        >
                          {(block.days || []).map((dayIdx: number) => (
                            <span key={dayIdx} style={chipStyle}>
                              {DAYS[dayIdx]}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}

                {editingBlocks && (
                  <button
                    className="btn btn-sm"
                    style={{ marginTop: 4 }}
                    onClick={() =>
                      setFixedBlocks([
                        ...fixedBlocks,
                        {
                          title: "",
                          start_min: 540,
                          end_min: 600,
                          days: [1, 2, 3, 4, 5],
                          recurring: true,
                        },
                      ])
                    }
                  >
                    + Add Block
                  </button>
                )}
              </div>
            </section>

            {/* ─── Cognitive Preferences (read-only) ─── */}
            <section style={cardStyle}>
              <div>
                <h3 className="meta-text" style={sectionHeadingStyle}>
                  Cognitive &amp; Session Preferences
                </h3>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr",
                    gap: 24,
                  }}
                >
                  <div>
                    <div style={fieldLabelStyle}>Session Style</div>
                    <div style={fieldValueStyle}>
                      {SESSION_STYLE_OPTIONS[user.session_style] || "Unknown"}
                    </div>
                  </div>
                  <div>
                    <div style={fieldLabelStyle}>Task Switching</div>
                    <div style={fieldValueStyle}>
                      {SWITCH_BUFFER_OPTIONS[user.switch_buffer] || "Unknown"}
                    </div>
                  </div>
                  <div>
                    <div style={fieldLabelStyle}>Deadline Management</div>
                    <div style={fieldValueStyle}>
                      {DEADLINE_STYLE_OPTIONS[user.deadline_style] || "Unknown"}
                    </div>
                  </div>
                </div>
              </div>
            </section>
          </div>

          <div style={{ marginTop: 40, textAlign: "center" }}>
            <Link href="/dashboard" className="btn btn-primary">
              Back to Dashboard
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
