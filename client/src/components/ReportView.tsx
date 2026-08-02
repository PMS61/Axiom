/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Behaviour Deviation Report
   Reads user onboarding baseline from localStorage and
   renders a human-readable drift report from saved daily
   observations.
   ═══════════════════════════════════════════════════════════ */

"use client";

import { useEffect, useState } from "react";
import { getUserProfile } from "@/app/actions/auth";
import { getReportDataForDate } from "@/app/actions/report";
import Header from "@/components/Header";
import { AppProvider } from "@/lib/store";
import {
  type ReportUserProfile,
  readStoredUserProfile,
  toReportProfile,
  toStoredUserProfileFromServerUser,
  writeStoredUserProfile,
} from "@/lib/userProfileStorage";

// ── Types ─────────────────────────────────────────────────

type UserProfile = ReportUserProfile;

type DeviationDir = "early" | "late" | "shorter" | "longer" | "on_track";

interface Deviation {
  category: string;
  label: string;
  baseline: string;
  current: string;
  delta: string;
  direction: DeviationDir;
  severity: "low" | "medium" | "high";
  note: string;
}

interface ObservedValues {
  actualWakeMin: number;
  actualSleepMin: number;
  avgSleepDurationMin: number;
  peak1ActualStart: number;
  low1ActualStart: number;
  avgSessionDurationPct: number;
  recoveryAdherencePct: number;
  contextSwitchesPerDay: number;
  deadlineBufferDays: number;
}

// ── Helpers ───────────────────────────────────────────────

function fmt(minutes: number): string {
  const h = Math.floor((((minutes % 1440) + 1440) % 1440) / 60)
    .toString()
    .padStart(2, "0");
  const m = ((((minutes % 1440) + 1440) % 1440) % 60)
    .toString()
    .padStart(2, "0");
  return `${h}:${m}`;
}

function fmtDelta(minutes: number): string {
  const abs = Math.abs(minutes);
  const h = Math.floor(abs / 60);
  const m = abs % 60;
  const parts: string[] = [];
  if (h > 0) parts.push(`${h}h`);
  if (m > 0) parts.push(`${m}m`);
  return parts.join(" ") || "0m";
}

const ROLE_LABELS = [
  "Student",
  "Researcher",
  "Professional",
  "Self-learner",
  "Other",
];
const SESSION_LABELS = [
  "Long deep blocks (90+ min)",
  "Medium sprints (45–60 min)",
  "Short bursts (20–30 min)",
  "Mix it up",
];
const DEADLINE_LABELS = [
  "Finish early",
  "Work steadily",
  "Work under pressure",
];

// colour helpers
function severityColour(s: "low" | "medium" | "high") {
  if (s === "high") return "var(--vermillion)";
  if (s === "medium") return "var(--watch)";
  return "var(--safe)";
}

function dirIcon(dir: DeviationDir) {
  if (dir === "early") return "↑";
  if (dir === "late") return "↓";
  if (dir === "shorter") return "←";
  if (dir === "longer") return "→";
  return "—";
}

// ── Observed deviation data ───────────────────────────────

function buildDeviations(
  profile: UserProfile,
  observed: ObservedValues | null,
): Deviation[] {
  const { wake_sleep } = profile;

  if (!observed) return [];

  const actualWakeMin = observed.actualWakeMin;
  const actualSleepMin = observed.actualSleepMin;
  const peak1ActualStart = observed.peak1ActualStart;
  const low1ActualStart = observed.low1ActualStart;
  const avgSleepDurationMin = observed.avgSleepDurationMin;
  const avgSessionDurationPct = observed.avgSessionDurationPct;
  const recoveryAdherencePct = observed.recoveryAdherencePct;
  const contextSwitchesPerDay = observed.contextSwitchesPerDay;
  const deadlineBufferDays = observed.deadlineBufferDays;

  const deviations: Deviation[] = [];

  // ── Wake time ─────────────────────────────────────────
  const wakeDelta = actualWakeMin - wake_sleep.wake;
  if (Math.abs(wakeDelta) >= 5) {
    deviations.push({
      category: "Sleep Rhythm",
      label: "Wake-up time",
      baseline: fmt(wake_sleep.wake),
      current: fmt(actualWakeMin),
      delta: fmtDelta(wakeDelta),
      direction: wakeDelta > 0 ? "late" : "early",
      severity:
        Math.abs(wakeDelta) > 45
          ? "high"
          : Math.abs(wakeDelta) > 20
            ? "medium"
            : "low",
      note:
        wakeDelta > 0
          ? `You're waking up ${fmtDelta(wakeDelta)} later than your baseline. This compresses your productive morning window.`
          : `You've been rising ${fmtDelta(Math.abs(wakeDelta))} earlier than planned — monitor energy levels in the afternoon.`,
    });
  }

  // ── Sleep time ────────────────────────────────────────
  const sleepDelta = actualSleepMin - wake_sleep.sleep;
  if (Math.abs(sleepDelta) >= 5) {
    deviations.push({
      category: "Sleep Rhythm",
      label: "Sleep time",
      baseline: fmt(wake_sleep.sleep),
      current: fmt(actualSleepMin),
      delta: fmtDelta(sleepDelta),
      direction: sleepDelta > 0 ? "late" : "early",
      severity:
        Math.abs(sleepDelta) > 60
          ? "high"
          : Math.abs(sleepDelta) > 30
            ? "medium"
            : "low",
      note:
        sleepDelta > 0
          ? `You're sleeping ${fmtDelta(sleepDelta)} later than intended. Late sleep accumulates a circadian debt that degrades next-day cognition.`
          : `Sleeping ${fmtDelta(Math.abs(sleepDelta))} earlier than planned. Could indicate fatigue front-loading or early recovery.`,
    });
  }

  // ── Sleep duration ────────────────────────────────────
  const targetSleepSpan =
    wake_sleep.wake > wake_sleep.sleep
      ? wake_sleep.wake - wake_sleep.sleep
      : 1440 - wake_sleep.sleep + wake_sleep.wake;
  const sleepDurDelta = avgSleepDurationMin - targetSleepSpan;
  if (Math.abs(sleepDurDelta) >= 10) {
    deviations.push({
      category: "Sleep Rhythm",
      label: "Sleep duration",
      baseline: fmtDelta(targetSleepSpan),
      current: fmtDelta(avgSleepDurationMin),
      delta: fmtDelta(Math.abs(sleepDurDelta)),
      direction: sleepDurDelta < 0 ? "shorter" : "longer",
      severity:
        Math.abs(sleepDurDelta) > 90
          ? "high"
          : Math.abs(sleepDurDelta) > 45
            ? "medium"
            : "low",
      note:
        sleepDurDelta < 0
          ? `Getting ${fmtDelta(Math.abs(sleepDurDelta))} less sleep than baseline. Chronic short sleep reduces decision quality and increases CL sensitivity.`
          : `Sleeping ${fmtDelta(sleepDurDelta)} beyond baseline. Could indicate recovery debt or under-stimulation during the day.`,
    });
  }

  // ── Peak-focus window drift ───────────────────────────
  if (profile.peak_focus_windows.length > 0) {
    const p = profile.peak_focus_windows[0];
    const startDelta = peak1ActualStart - p.start_min;
    if (Math.abs(startDelta) >= 10) {
      deviations.push({
        category: "Focus Windows",
        label: "Peak-focus start",
        baseline: fmt(p.start_min),
        current: fmt(peak1ActualStart),
        delta: fmtDelta(startDelta),
        direction: startDelta > 0 ? "late" : "early",
        severity: Math.abs(startDelta) > 45 ? "high" : "medium",
        note:
          startDelta > 0
            ? `Your deep-work block is starting ${fmtDelta(startDelta)} late, likely eating into your peak cognitive bandwidth.`
            : `Starting focus sessions ${fmtDelta(Math.abs(startDelta))} earlier than baseline — align with your natural alertness curve.`,
      });
    }
  }

  // ── Low-energy window drift ───────────────────────────
  if (profile.low_energy_windows.length > 0) {
    const l = profile.low_energy_windows[0];
    const lDelta = low1ActualStart - l.start_min;
    if (Math.abs(lDelta) >= 10) {
      deviations.push({
        category: "Energy Pattern",
        label: "Low-energy onset",
        baseline: fmt(l.start_min),
        current: fmt(low1ActualStart),
        delta: fmtDelta(lDelta),
        direction: lDelta < 0 ? "early" : "late",
        severity: Math.abs(lDelta) > 45 ? "high" : "medium",
        note:
          lDelta < 0
            ? `Energy dip is arriving ${fmtDelta(Math.abs(lDelta))} earlier than expected. Consider lighter tasks after lunch to avoid forced errors.`
            : `Low-energy phase shifted ${fmtDelta(lDelta)} later — you may be sustaining effort longer than optimal.`,
      });
    }
  }

  // ── Session completion ────────────────────────────────
  if (avgSessionDurationPct < 85) {
    deviations.push({
      category: "Session Quality",
      label: "Session completion rate",
      baseline: "100%",
      current: `${avgSessionDurationPct}%`,
      delta: `${100 - avgSessionDurationPct}% gap`,
      direction: "shorter",
      severity: avgSessionDurationPct < 65 ? "high" : "medium",
      note: `Only ${avgSessionDurationPct}% of planned session time is being used. Interruptions or overestimation of capacity are likely causes.`,
    });
  }

  // ── Recovery adherence ────────────────────────────────
  if (recoveryAdherencePct < 75) {
    deviations.push({
      category: "Recovery",
      label: "Recovery activity adherence",
      baseline: "100%",
      current: `${recoveryAdherencePct}%`,
      delta: `${100 - recoveryAdherencePct}% missed`,
      direction: "shorter",
      severity: recoveryAdherencePct < 50 ? "high" : "medium",
      note: `Recovery activities (${profile.recovery_activities[0]?.name ?? "e.g. Walk"}) are being skipped ${100 - recoveryAdherencePct}% of the time. This compounds fatigue over multi-day periods.`,
    });
  }

  // ── Context switching ─────────────────────────────────
  if (contextSwitchesPerDay > 4) {
    deviations.push({
      category: "Focus Quality",
      label: "Context switches / day",
      baseline: "≤ 4",
      current: `${contextSwitchesPerDay}`,
      delta: `+${contextSwitchesPerDay - 4}`,
      direction: "late",
      severity: contextSwitchesPerDay > 7 ? "high" : "medium",
      note: `Averaging ${contextSwitchesPerDay} task switches per day vs. your preferred style. Each unplanned switch adds ~${SESSION_LABELS[profile.session_config.switch_buffer].split("(")[1]?.replace(")", "") ?? "15 min"} of re-entry cost.`,
    });
  }

  // ── Deadline buffer ───────────────────────────────────
  if (deadlineBufferDays !== 0) {
    deviations.push({
      category: "Deadline Rhythm",
      label: "Average deadline buffer",
      baseline: DEADLINE_LABELS[profile.deadline_style],
      current:
        deadlineBufferDays < 0
          ? `${Math.abs(deadlineBufferDays)}d late on avg`
          : `${deadlineBufferDays}d ahead on avg`,
      delta: `${Math.abs(deadlineBufferDays)}d`,
      direction: deadlineBufferDays < 0 ? "late" : "early",
      severity: Math.abs(deadlineBufferDays) > 2 ? "high" : "medium",
      note:
        deadlineBufferDays < 0
          ? `Tasks are completing ${Math.abs(deadlineBufferDays)} day(s) past deadline on average — misaligned with your "${DEADLINE_LABELS[profile.deadline_style]}" preference.`
          : `Finishing ahead of deadline. Great alignment with your "${DEADLINE_LABELS[profile.deadline_style]}" style.`,
    });
  }

  return deviations;
}

// ── Sub-components ────────────────────────────────────────

function DeviationCard({ d }: { d: Deviation }) {
  const col = severityColour(d.severity);
  const isOnTrack = d.direction === "on_track";
  return (
    <div
      style={{
        borderTop: `2px solid ${col}`,
        borderLeft: "0.5px solid var(--rule)",
        borderRight: "0.5px solid var(--rule)",
        borderBottom: "0.5px solid var(--rule)",
        padding: "20px 24px",
        background: "var(--card-bg)",
        display: "grid",
        gap: 14,
      }}
    >
      {/* Header row */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: 12,
        }}
      >
        <div>
          <div className="meta-text" style={{ marginBottom: 4 }}>
            {d.category}
          </div>
          <div style={{ fontWeight: 700, fontSize: 15 }}>{d.label}</div>
        </div>
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: 10,
            textTransform: "uppercase",
            letterSpacing: "0.12em",
            color: col,
            border: `0.5px solid ${col}`,
            padding: "3px 10px",
            whiteSpace: "nowrap",
          }}
        >
          {d.severity === "high"
            ? "High drift"
            : d.severity === "medium"
              ? "Moderate"
              : "Slight"}
        </span>
      </div>

      {/* Baseline → Current */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr auto 1fr",
          alignItems: "center",
          gap: 12,
          padding: "12px 16px",
          background: "var(--bg)",
          border: "0.5px solid var(--rule)",
        }}
      >
        <div>
          <div className="meta-text" style={{ marginBottom: 4 }}>
            Baseline
          </div>
          <div
            style={{ fontFamily: "var(--mono)", fontWeight: 700, fontSize: 16 }}
          >
            {d.baseline}
          </div>
        </div>
        <div
          style={{
            textAlign: "center",
            color: col,
            fontSize: 20,
            fontWeight: 700,
          }}
        >
          {isOnTrack ? "✓" : dirIcon(d.direction)}
        </div>
        <div style={{ textAlign: "right" }}>
          <div className="meta-text" style={{ marginBottom: 4 }}>
            Observed
          </div>
          <div
            style={{
              fontFamily: "var(--mono)",
              fontWeight: 700,
              fontSize: 16,
              color: isOnTrack ? "var(--safe)" : col,
            }}
          >
            {d.current}
          </div>
        </div>
      </div>

      {/* Delta pill */}
      {!isOnTrack && (
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            style={{
              fontFamily: "var(--mono)",
              fontSize: 11,
              color: col,
              fontWeight: 700,
            }}
          >
            Δ {d.delta}
          </span>
          <span className="meta-text">deviation from baseline</span>
        </div>
      )}

      {/* Note */}
      <div
        style={{
          fontSize: 12,
          lineHeight: 1.8,
          color: "var(--muted)",
          borderLeft: `2px solid ${col}`,
          paddingLeft: 12,
        }}
      >
        {d.note}
      </div>
    </div>
  );
}

function SummaryBar({ deviations }: { deviations: Deviation[] }) {
  const high = deviations.filter((d) => d.severity === "high").length;
  const medium = deviations.filter((d) => d.severity === "medium").length;
  const low = deviations.filter((d) => d.severity === "low").length;
  const score = Math.max(0, Math.round(100 - high * 15 - medium * 7 - low * 3));

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
        gap: 0,
        borderTop: "0.5px solid var(--rule)",
        borderLeft: "0.5px solid var(--rule)",
        marginBottom: 40,
      }}
    >
      {[
        {
          label: "Alignment score",
          value: `${score}`,
          colour:
            score >= 80
              ? "var(--safe)"
              : score >= 60
                ? "var(--watch)"
                : "var(--vermillion)",
        },
        {
          label: "High drift",
          value: String(high),
          colour: "var(--vermillion)",
        },
        {
          label: "Moderate drift",
          value: String(medium),
          colour: "var(--watch)",
        },
        { label: "Low drift", value: String(low), colour: "var(--safe)" },
      ].map((item) => (
        <div
          key={item.label}
          style={{
            padding: "20px 24px",
            borderRight: "0.5px solid var(--rule)",
            borderBottom: "0.5px solid var(--rule)",
          }}
        >
          <div className="meta-text" style={{ marginBottom: 8 }}>
            {item.label}
          </div>
          <div
            style={{
              fontFamily: "var(--mono)",
              fontWeight: 700,
              fontSize: 32,
              lineHeight: 1,
              color: item.colour,
            }}
          >
            {item.value}
          </div>
        </div>
      ))}
    </div>
  );
}

function CategoryGroup({
  category,
  items,
}: {
  category: string;
  items: Deviation[];
}) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          marginBottom: 16,
        }}
      >
        <span className="meta-text">{category}</span>
        <div style={{ flex: 1, height: "0.5px", background: "var(--rule)" }} />
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          gap: 12,
        }}
      >
        {items.map((d) => (
          <DeviationCard key={`${d.category}-${d.label}`} d={d} />
        ))}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div
      style={{
        textAlign: "center",
        padding: "80px 0",
        border: "0.5px solid var(--rule)",
        background: "var(--card-bg)",
      }}
    >
      <div style={{ fontSize: 48, marginBottom: 16 }}>✓</div>
      <div style={{ fontWeight: 700, fontSize: 18, marginBottom: 8 }}>
        All behaviour aligned
      </div>
      <p style={{ color: "var(--muted)", fontSize: 12 }}>
        No significant deviations detected from your baseline profile. Keep
        going — we'll flag any drift as it appears.
      </p>
    </div>
  );
}

function Timeline({ profile }: { profile: UserProfile }) {
  // Simple visual timeline for the day (midnight = 0, 23:59 = 1439)
  const segments = [
    {
      label: "Sleep",
      start: profile.wake_sleep.sleep,
      end: 1440,
      colour: "#E8E0D4",
    },
    {
      label: "Sleep",
      start: 0,
      end: profile.wake_sleep.wake,
      colour: "#E8E0D4",
    },
    ...profile.peak_focus_windows.map((w) => ({
      label: "Peak",
      start: w.start_min,
      end: w.end_min,
      colour: "var(--ink)",
    })),
    ...profile.low_energy_windows.map((w) => ({
      label: "Low",
      start: w.start_min,
      end: w.end_min,
      colour: "var(--watch)",
    })),
  ];

  function pct(min: number) {
    return `${(min / 1440) * 100}%`;
  }

  return (
    <div style={{ marginBottom: 40 }}>
      <div className="meta-text" style={{ marginBottom: 10 }}>
        Your baseline day — 00:00 → 23:59
      </div>
      <div
        style={{
          position: "relative",
          height: 28,
          background: "var(--bg)",
          border: "0.5px solid var(--rule)",
        }}
      >
        {segments.map((s) => (
          <div
            key={`${s.label}-${s.start}`}
            title={`${s.label}: ${fmt(s.start)} – ${fmt(s.end)}`}
            style={{
              position: "absolute",
              left: pct(s.start),
              width: pct(Math.max(s.end - s.start, 0)),
              top: 0,
              bottom: 0,
              background: s.colour,
              opacity: 0.85,
            }}
          />
        ))}
        {/* Wake marker */}
        <div
          style={{
            position: "absolute",
            left: pct(profile.wake_sleep.wake),
            top: -6,
            bottom: -6,
            width: 1,
            background: "var(--ink)",
          }}
        />
        {/* Sleep marker */}
        <div
          style={{
            position: "absolute",
            left: pct(profile.wake_sleep.sleep),
            top: -6,
            bottom: -6,
            width: 1,
            background: "var(--ink)",
          }}
        />
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: 6,
        }}
      >
        <span className="meta-text">00:00</span>
        <span className="meta-text">06:00</span>
        <span className="meta-text">12:00</span>
        <span className="meta-text">18:00</span>
        <span className="meta-text">23:59</span>
      </div>
      <div style={{ display: "flex", gap: 16, marginTop: 10 }}>
        {[
          { label: "Sleep window", colour: "#E8E0D4" },
          { label: "Peak focus", colour: "var(--ink)" },
          { label: "Low energy", colour: "var(--watch)" },
        ].map((l) => (
          <div
            key={l.label}
            style={{ display: "flex", alignItems: "center", gap: 6 }}
          >
            <div
              style={{
                width: 10,
                height: 10,
                background: l.colour,
                flexShrink: 0,
              }}
            />
            <span className="meta-text">{l.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main report content ───────────────────────────────────

function ReportContent() {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [observed, setObserved] = useState<ObservedValues | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let isMounted = true;

    async function loadData() {
      const localProfile = readStoredUserProfile();
      if (localProfile) {
        if (isMounted) {
          setProfile(toReportProfile(localProfile));
        }
      } else {
        try {
          const { user, error } = await getUserProfile();
          if (!error && user) {
            const normalizedProfile = toStoredUserProfileFromServerUser(user);
            if (normalizedProfile) {
              if (isMounted) {
                writeStoredUserProfile(normalizedProfile);
                setProfile(toReportProfile(normalizedProfile));
              }
            }
          }
        } catch {
          // Ignore fetch errors
        }
      }

      try {
        const result = await getReportDataForDate();
        if (!isMounted) return;

        if (result.observation) {
          setObserved({
            actualWakeMin: result.observation.actualWakeMin,
            actualSleepMin: result.observation.actualSleepMin,
            avgSleepDurationMin: result.observation.avgSleepDurationMin,
            peak1ActualStart: result.observation.peak1ActualStart,
            low1ActualStart: result.observation.low1ActualStart,
            avgSessionDurationPct: result.observation.avgSessionDurationPct,
            recoveryAdherencePct: result.observation.recoveryAdherencePct,
            contextSwitchesPerDay: result.observation.contextSwitchesPerDay,
            deadlineBufferDays: result.observation.deadlineBufferDays,
          });
        }
      } catch {
        // Ignore observation errors
      } finally {
        if (isMounted) setLoaded(true);
      }
    }

    loadData();

    return () => {
      isMounted = false;
    };
  }, []);

  const deviations = profile ? buildDeviations(profile, observed) : [];

  // group by category
  const grouped = deviations.reduce<Record<string, Deviation[]>>((acc, d) => {
    if (!acc[d.category]) acc[d.category] = [];
    acc[d.category].push(d);
    return acc;
  }, {});

  return (
    <div style={{ minHeight: "100vh" }}>
      <Header />

      {/* Hero ─────────────────────────────────────────── */}
      <section
        className="container section-rule"
        style={{ paddingTop: 80, paddingBottom: 40 }}
      >
        <div className="meta-text" style={{ marginBottom: 10 }}>
          Behaviour Report
        </div>
        <h1 style={{ marginBottom: 12 }}>How your habits have shifted</h1>
        {profile ? (
          <p
            style={{
              color: "var(--muted)",
              maxWidth: "100%",
              fontSize: 13,
              lineHeight: 1.8,
            }}
          >
            Comparing <strong>{profile.name}</strong>'s observed behaviour
            against the baseline set during onboarding on{" "}
            {new Date(profile.onboarded_at).toLocaleDateString("en-GB", {
              day: "numeric",
              month: "long",
              year: "numeric",
            })}
            . Deviations marked{" "}
            <span style={{ color: "var(--vermillion)", fontWeight: 700 }}>
              High
            </span>{" "}
            need immediate attention.
          </p>
        ) : (
          <p style={{ color: "var(--muted)", fontSize: 13 }}>
            {loaded
              ? "No onboarding profile found. Complete onboarding to see your personal drift report."
              : "Loading…"}
          </p>
        )}
      </section>

      <main
        className="container"
        style={{
          paddingTop: 40,
          paddingBottom: 80,
          maxWidth: 1280,
          width: "100%",
        }}
      >
        {!loaded && (
          <div
            style={{
              textAlign: "center",
              padding: "80px 0",
              color: "var(--muted)",
            }}
          >
            Loading profile…
          </div>
        )}

        {loaded && !profile && (
          <div
            style={{
              padding: "40px 32px",
              border: "0.5px solid var(--rule)",
              background: "var(--card-bg)",
              textAlign: "center",
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 8 }}>
              No profile data
            </div>
            <p style={{ color: "var(--muted)", fontSize: 12 }}>
              Finish onboarding to generate your behaviour deviation report.
            </p>
            <a
              href="/onboarding"
              className="btn btn-primary"
              style={{ display: "inline-block", marginTop: 20 }}
            >
              Start onboarding →
            </a>
          </div>
        )}

        {loaded && profile && (
          <>
            {/* Baseline profile pill strip */}
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: 8,
                marginBottom: 32,
                padding: "16px 0",
                borderBottom: "0.5px solid var(--rule)",
              }}
            >
              {[
                { k: "Role", v: ROLE_LABELS[profile.role] ?? "–" },
                { k: "Wake", v: fmt(profile.wake_sleep.wake) },
                { k: "Sleep", v: fmt(profile.wake_sleep.sleep) },
                {
                  k: "Session style",
                  v:
                    SESSION_LABELS[profile.session_config.session_style]
                      ?.split("(")[0]
                      .trim() ?? "–",
                },
                {
                  k: "Deadline",
                  v: DEADLINE_LABELS[profile.deadline_style] ?? "–",
                },
                { k: "Timezone", v: profile.timezone },
              ].map(({ k, v }) => (
                <div
                  key={k}
                  style={{
                    padding: "6px 14px",
                    border: "0.5px solid var(--rule)",
                    display: "flex",
                    gap: 8,
                    alignItems: "center",
                  }}
                >
                  <span className="meta-text">{k}</span>
                  <span style={{ fontWeight: 700, fontSize: 12 }}>{v}</span>
                </div>
              ))}
            </div>

            {/* Timeline */}
            <Timeline profile={profile} />

            {/* Summary */}
            <SummaryBar deviations={deviations} />

            {/* Note about data source */}
            <div
              className="trace-log"
              style={{ marginBottom: 40, padding: "16px 24px" }}
            >
              <div className="meta-text" style={{ marginBottom: 6 }}>
                Data source
              </div>
              <div
                style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.7 }}
              >
                {observed
                  ? "Observed values are pulled from your saved daily feedback. This includes actual wake/sleep times, session completion rates, and other metrics you track in the Feedback tab."
                  : "No observed data available yet. Enter your daily values in the Feedback tab to see how your behaviour deviates from baseline."}
              </div>
            </div>

            {/* Per-category deviations */}
            {deviations.length === 0 ? (
              <EmptyState />
            ) : (
              <section
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(420px, 1fr))",
                  gap: 24,
                }}
              >
                {Object.entries(grouped).map(([cat, items]) => (
                  <CategoryGroup key={cat} category={cat} items={items} />
                ))}
              </section>
            )}

            {/* Footer note */}
            <div
              style={{
                marginTop: 48,
                paddingTop: 24,
                borderTop: "0.5px solid var(--rule)",
                fontSize: 11,
                color: "var(--muted)",
                lineHeight: 1.8,
              }}
            >
              <span className="meta-text">Methodology — </span>
              Baseline is your onboarding snapshot. Observed values are averaged
              over the current reporting window. Severity thresholds:{" "}
              <strong>High</strong> = &gt; 45 min drift or &gt; 25% adherence
              gap; <strong>Moderate</strong> = 20–45 min or 15–25%;{" "}
              <strong>Low</strong> = 5–20 min or &lt; 15%.
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default function ReportView() {
  return (
    <AppProvider>
      <ReportContent />
    </AppProvider>
  );
}
