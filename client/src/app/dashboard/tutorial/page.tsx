/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Tutorial & Documentation
   Brutalist, rule-based manual matching the project aesthetics.
   ═══════════════════════════════════════════════════════════ */

import Header from "@/components/Header";
import { AppProvider } from "@/lib/store";

function TutorialContent() {
  return (
    <div style={{ minHeight: "100vh", paddingBottom: 100 }}>
      <Header />

      <main
        className="container section-rule"
        style={{ paddingTop: 60, maxWidth: 640, margin: "0 auto" }}
      >
        <div className="meta-text" style={{ marginBottom: 16 }}>
          SYSTEM REFERENCE MANUAL.
        </div>
        <h1 style={{ fontSize: 42, marginBottom: 40, lineHeight: 1.1 }}>
          Axiom User Manual <br />
          <em style={{ fontWeight: 400, color: "var(--muted)", fontSize: 32 }}>
            v0.2 Alpha
          </em>
        </h1>

        <div
          style={{
            padding: "24px",
            background: "var(--card-bg)",
            border: "0.5px solid var(--ink)",
            marginBottom: 60,
          }}
        >
          <p
            style={{
              margin: 0,
              fontSize: 16,
              lineHeight: 1.5,
              fontWeight: 500,
            }}
          >
            Standard schedulers ask &quot;When do you have time?&quot;. Axiom
            asks &quot;When do you have the cognitive capacity?&quot;. Time is
            fixed; capacity is variable. This system computes schedule fitness
            based purely on tracked mental bandwidth.
          </p>
        </div>

        {/* ── 1. Getting Started ── */}
        <section style={{ marginBottom: 48 }}>
          <h2
            style={{
              fontSize: 24,
              marginBottom: 16,
              borderBottom: "0.5px solid var(--rule)",
              paddingBottom: 8,
            }}
          >
            1. Getting Started
          </h2>
          <p style={{ marginBottom: 16, lineHeight: 1.6 }}>
            Axiom begins with a 5-step onboarding calibration wizard that
            captures your biological rhythms, cognitive preferences, and
            recurring commitments.
          </p>

          <div
            className="trace-log"
            style={{ padding: "20px", marginBottom: 24 }}
          >
            <div
              className="log-line rule"
              style={{ fontSize: 13, textTransform: "none", marginBottom: 8 }}
            >
              Onboarding Sequence:
            </div>
            <div
              style={{
                fontFamily: "var(--mono)",
                fontSize: 12,
                lineHeight: 2.2,
              }}
            >
              <div className="log-line">
                Step 1 → Account creation (name, email, password)
              </div>
              <div className="log-line">
                Step 2 → Timezone, role selection, wake/sleep cycle
              </div>
              <div className="log-line">
                Step 3 → Fixed commitments (recurring availability blocks)
              </div>
              <div className="log-line">
                Step 4 → Peak focus windows & low energy windows
              </div>
              <div className="log-line">
                Step 5 → Session style, task switching, deadline preference,
                recovery activities
              </div>
            </div>
          </div>
          <p style={{ lineHeight: 1.6, color: "var(--muted)" }}>
            All onboarding data is editable later from the{" "}
            <strong style={{ color: "var(--ink)" }}>Profile</strong> page,
            accessible via the navigation bar.
          </p>
        </section>

        {/* ── 2. The Scoring System ── */}
        <section style={{ marginBottom: 48 }}>
          <h2
            style={{
              fontSize: 24,
              marginBottom: 16,
              borderBottom: "0.5px solid var(--rule)",
              paddingBottom: 8,
            }}
          >
            2. The Scoring System
          </h2>
          <p style={{ marginBottom: 16, lineHeight: 1.6 }}>
            Every task placed inside Axiom undergoes deterministic Cognitive
            Load (CL) computation rather than just measuring duration.
          </p>

          <div
            className="trace-log"
            style={{ padding: "20px", marginBottom: 24 }}
          >
            <div
              className="log-line rule"
              style={{ fontSize: 13, textTransform: "none", marginBottom: 8 }}
            >
              Engine Formula:
            </div>
            <div
              style={{
                fontFamily: "var(--mono)",
                fontSize: 14,
                fontWeight: 600,
              }}
            >
              CL = base_difficulty × duration_weight × deadline_urgency ×
              type_multiplier × priority_weight
            </div>
          </div>

          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              marginBottom: 24,
              fontFamily: "var(--mono)",
              fontSize: 12,
            }}
          >
            <thead>
              <tr
                style={{
                  borderBottom: "0.5px solid var(--ink)",
                  textAlign: "left",
                }}
              >
                <th style={{ padding: "8px 0" }}>Task Type</th>
                <th>Multiplier</th>
                <th>Character</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: "0.5px dotted var(--rule)" }}>
                <td style={{ padding: "8px 0" }}>Learning New Concept</td>
                <td>
                  <strong>1.4×</strong>
                </td>
                <td>Highest cognitive tax</td>
              </tr>
              <tr style={{ borderBottom: "0.5px dotted var(--rule)" }}>
                <td style={{ padding: "8px 0" }}>Problem Solving</td>
                <td>
                  <strong>1.3×</strong>
                </td>
                <td>Deep analytical work</td>
              </tr>
              <tr style={{ borderBottom: "0.5px dotted var(--rule)" }}>
                <td style={{ padding: "8px 0" }}>Writing / Drafting</td>
                <td>
                  <strong>1.1×</strong>
                </td>
                <td>Creative synthesis</td>
              </tr>
              <tr style={{ borderBottom: "0.5px dotted var(--rule)" }}>
                <td style={{ padding: "8px 0" }}>Revision</td>
                <td>
                  <strong>0.9×</strong>
                </td>
                <td>Familiar material recall</td>
              </tr>
              <tr style={{ borderBottom: "0.5px dotted var(--rule)" }}>
                <td style={{ padding: "8px 0" }}>Reading</td>
                <td>
                  <strong>0.8×</strong>
                </td>
                <td>Passive intake</td>
              </tr>
              <tr style={{ borderBottom: "0.5px dotted var(--rule)" }}>
                <td style={{ padding: "8px 0" }}>Administrative</td>
                <td>
                  <strong>0.6×</strong>
                </td>
                <td>Routine execution</td>
              </tr>
              <tr>
                <td style={{ padding: "8px 0", color: "var(--safe)" }}>
                  Recreational
                </td>
                <td style={{ color: "var(--safe)" }}>
                  <strong>−1.0×</strong>
                </td>
                <td style={{ color: "var(--safe)" }}>
                  Negative CL — restores bandwidth
                </td>
              </tr>
            </tbody>
          </table>

          <ul style={{ paddingLeft: 24, lineHeight: 1.6, color: "var(--ink)" }}>
            <li style={{ marginBottom: 8 }}>
              <strong>Priority weights</strong> scale the final CL: High = 1.5×,
              Normal = 1.0×, Low = 0.7×.
            </li>
            <li style={{ marginBottom: 8 }}>
              <strong>Deadline urgency</strong> amplifies CL when deadlines
              approach: ≤1 day = 1.8×, ≤3 days = 1.4×, ≤7 days = 1.1×.
            </li>
            <li>
              <strong>Recreational tasks</strong> (walks, breaks) produce
              negative CL. Completing them mathematically restores your
              bandwidth budget.
            </li>
          </ul>
        </section>

        {/* ── 3. The 96 × 7 Matrix ── */}
        <section style={{ marginBottom: 48 }}>
          <h2
            style={{
              fontSize: 24,
              marginBottom: 16,
              borderBottom: "0.5px solid var(--rule)",
              paddingBottom: 8,
            }}
          >
            3. The 96 × 7 Master Matrix
          </h2>
          <p style={{ marginBottom: 16, lineHeight: 1.6 }}>
            Your week is chunked into 672 cells — 96 fifteen-minute slots per
            day across 7 days. The <strong>Tasks / Matrix</strong> view renders
            this grid. Tasks can be dragged onto specific cells for manual
            placement, or auto-scheduled using the bin-packing algorithm.
          </p>

          <ul style={{ paddingLeft: 24, lineHeight: 1.6, color: "var(--ink)" }}>
            <li style={{ marginBottom: 8 }}>
              <strong>Bandwidth Curve:</strong> Upon onboarding, the system
              generates 96 bandwidth values following an ultradian rhythm model
              — peaks at ~10:00 and ~15:00, natural troughs at ~13:00 and
              ~20:00.
            </li>
            <li style={{ marginBottom: 8 }}>
              <strong>Slot Fitness Scoring:</strong> Each candidate slot is
              scored using{" "}
              <code
                style={{
                  background: "var(--card-bg)",
                  padding: "2px 6px",
                  border: "0.5px solid var(--rule)",
                }}
              >
                bandwidth − taskCL + productiveHourBonus + contextSwitchPenalty
                + deadlineProximityBonus
              </code>
              . The highest-scoring slot wins.
            </li>
            <li style={{ marginBottom: 8 }}>
              <strong>Productive Hour Learning:</strong> As you complete tasks,
              matrix slots progress through states: Unknown → Observed → Tracked
              → Confirmed. Confirmed hours receive a +1.2 fitness bonus for
              future scheduling.
            </li>
            <li>
              <strong>Context Switch Penalty:</strong> ≥3 subject switches in a
              window: −2.0 penalty. ≥2 switches: −0.8 penalty. The engine
              naturally clusters similar subjects together.
            </li>
          </ul>
        </section>

        {/* ── 4. Energy Management ── */}
        <section style={{ marginBottom: 48 }}>
          <h2
            style={{
              fontSize: 24,
              marginBottom: 16,
              borderBottom: "0.5px solid var(--rule)",
              paddingBottom: 8,
            }}
          >
            4. Energy Management & Burnout
          </h2>
          <p style={{ marginBottom: 16, lineHeight: 1.6 }}>
            Calibrate your energy in real-time using the 5-level toggle in the
            navigation bar. This directly scales the bandwidth curve, gating
            what the scheduler can assign.
          </p>

          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              marginBottom: 24,
              fontFamily: "var(--mono)",
              fontSize: 12,
            }}
          >
            <thead>
              <tr
                style={{
                  borderBottom: "0.5px solid var(--ink)",
                  textAlign: "left",
                }}
              >
                <th style={{ padding: "8px 0" }}>Level</th>
                <th>Label</th>
                <th>Bandwidth Multiplier</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: "0.5px dotted var(--rule)" }}>
                <td style={{ padding: "8px 0" }}>+2</td>
                <td>Peak</td>
                <td>1.4× (expanded capacity)</td>
              </tr>
              <tr style={{ borderBottom: "0.5px dotted var(--rule)" }}>
                <td style={{ padding: "8px 0" }}>+1</td>
                <td>Energised</td>
                <td>1.2×</td>
              </tr>
              <tr style={{ borderBottom: "0.5px dotted var(--rule)" }}>
                <td style={{ padding: "8px 0" }}>0</td>
                <td>Baseline</td>
                <td>1.0× (default)</td>
              </tr>
              <tr style={{ borderBottom: "0.5px dotted var(--rule)" }}>
                <td style={{ padding: "8px 0" }}>−1</td>
                <td>Depleted</td>
                <td>0.7×</td>
              </tr>
              <tr>
                <td style={{ padding: "8px 0", color: "var(--vermillion)" }}>
                  −2
                </td>
                <td style={{ color: "var(--vermillion)" }}>Exhausted</td>
                <td style={{ color: "var(--vermillion)" }}>
                  0.5× (hard limiter)
                </td>
              </tr>
            </tbody>
          </table>

          <h3 style={{ fontSize: 16, marginBottom: 12, marginTop: 24 }}>
            Burnout Risk States
          </h3>
          <p style={{ marginBottom: 16, lineHeight: 1.6 }}>
            A banner appears on the Dashboard when your aggregate load triggers
            a burnout state transition:
          </p>
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              marginBottom: 32,
              fontFamily: "var(--mono)",
              fontSize: 12,
            }}
          >
            <thead>
              <tr
                style={{
                  borderBottom: "0.5px solid var(--ink)",
                  textAlign: "left",
                }}
              >
                <th style={{ padding: "8px 0" }}>State</th>
                <th>Trigger</th>
                <th>System Response</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: "0.5px dotted var(--rule)" }}>
                <td style={{ padding: "8px 0" }}>
                  <strong>Safe</strong>
                </td>
                <td>Net CL ≤ 35</td>
                <td>Standard scheduling — unrestricted.</td>
              </tr>
              <tr style={{ borderBottom: "0.5px dotted var(--rule)" }}>
                <td style={{ padding: "8px 0" }}>
                  <strong>Watch</strong>
                </td>
                <td>Elevated load</td>
                <td>Monitoring active. Aggregate duration tracked.</td>
              </tr>
              <tr style={{ borderBottom: "0.5px dotted var(--rule)" }}>
                <td style={{ padding: "8px 0" }}>
                  <strong>Warning</strong>
                </td>
                <td>Sustained overload</td>
                <td>Forced light day recommended within 72 hours.</td>
              </tr>
              <tr>
                <td style={{ padding: "8px 0", color: "var(--vermillion)" }}>
                  <strong>Critical</strong>
                </td>
                <td style={{ color: "var(--vermillion)" }}>Energy −2</td>
                <td style={{ color: "var(--vermillion)" }}>
                  High CL tasks blocked from scheduling. Hard limiter active.
                </td>
              </tr>
            </tbody>
          </table>
        </section>

        {/* ── 5. Your Profile ── */}
        <section style={{ marginBottom: 48 }}>
          <h2
            style={{
              fontSize: 24,
              marginBottom: 16,
              borderBottom: "0.5px solid var(--rule)",
              paddingBottom: 8,
            }}
          >
            5. Your Profile
          </h2>
          <p style={{ marginBottom: 16, lineHeight: 1.6 }}>
            Navigate to <strong>Profile</strong> from the header to view and
            edit all onboarding calibration data. The following sections are
            editable in-place:
          </p>

          <ul style={{ paddingLeft: 24, lineHeight: 1.6, color: "var(--ink)" }}>
            <li style={{ marginBottom: 8 }}>
              <strong>Sleep / Wake Cycle:</strong> Adjust your wake and sleep
              times. The bandwidth curve recalibrates based on these boundaries.
            </li>
            <li style={{ marginBottom: 8 }}>
              <strong>Peak Focus Windows:</strong> Up to 2 time windows where
              you perform your best cognitive work. The scheduler applies a +1.2
              fitness bonus to these slots.
            </li>
            <li style={{ marginBottom: 8 }}>
              <strong>Low Energy Windows:</strong> Up to 2 time windows you
              consistently experience dips. The scheduler avoids placing high-CL
              tasks here.
            </li>
            <li style={{ marginBottom: 8 }}>
              <strong>Availability Blocks:</strong> Recurring commitments
              (classes, meetings, commutes) that are hard-excluded from
              scheduling. Configurable per day-of-week.
            </li>
            <li>
              <strong>Cognitive Preferences:</strong> Session style, task
              switching buffer, and deadline management style (read-only —
              configured during onboarding).
            </li>
          </ul>
        </section>

        {/* ── 6. Conflicts ── */}
        <section style={{ marginBottom: 48 }}>
          <h2
            style={{
              fontSize: 24,
              marginBottom: 16,
              borderBottom: "0.5px solid var(--rule)",
              paddingBottom: 8,
            }}
          >
            6. Resolving Conflicts & Reasoning Chains
          </h2>
          <p style={{ marginBottom: 16, lineHeight: 1.6 }}>
            When the bin-packing algorithm encounters spatial paradoxes (e.g. a
            deadline expires before sufficient free capacity exists), it halts
            and invokes the Conflict Resolver panel.
          </p>
          <div className="trace-log" style={{ padding: "20px" }}>
            <div
              className="log-line conflict"
              style={{ fontSize: 13, textTransform: "none", margin: 0 }}
            >
              Resolution Vector Options:
            </div>
            <ul style={{ marginTop: 12, paddingLeft: 16, marginBottom: 0 }}>
              <li style={{ paddingBottom: 6 }}>
                <strong>DEFER:</strong> Shifts the task to the next available
                day, given flexible deadlines.
              </li>
              <li style={{ paddingBottom: 6 }}>
                <strong>SACRIFICE DEPTH:</strong> Mutates task CL × 0.7. Allows
                fitting at the cost of accepting a lower-quality output (30%
                reduction).
              </li>
              <li>
                <strong>EXTEND DEADLINE:</strong> Mutates the deadline
                constraint. Only valid for Low or Normal priority. High-priority
                tasks escalate to manual user override.
              </li>
            </ul>
          </div>
          <p style={{ marginTop: 16, lineHeight: 1.6 }}>
            The <strong>Reasoning Chain</strong> panel (toggle from Dashboard)
            shows every evaluation step the scheduler took — slot candidates,
            fitness scores, applied rules, and rejected alternatives — providing
            full algorithmic transparency.
          </p>
        </section>

        {/* ── 7. Navigation ── */}
        <section style={{ marginBottom: 48 }}>
          <h2
            style={{
              fontSize: 24,
              marginBottom: 16,
              borderBottom: "0.5px solid var(--rule)",
              paddingBottom: 8,
            }}
          >
            7. Navigation Reference
          </h2>

          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              fontFamily: "var(--mono)",
              fontSize: 12,
            }}
          >
            <thead>
              <tr
                style={{
                  borderBottom: "0.5px solid var(--ink)",
                  textAlign: "left",
                }}
              >
                <th style={{ padding: "8px 0" }}>Page</th>
                <th>Purpose</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: "0.5px dotted var(--rule)" }}>
                <td style={{ padding: "8px 0" }}>
                  <strong>Dashboard</strong>
                </td>
                <td>
                  Today&apos;s schedule, bandwidth curve, CL stats, current
                  focus block, reasoning chain toggle.
                </td>
              </tr>
              <tr style={{ borderBottom: "0.5px dotted var(--rule)" }}>
                <td style={{ padding: "8px 0" }}>
                  <strong>Tasks / Matrix</strong>
                </td>
                <td>
                  96×7 drag-and-drop grid, syllabus-powered task generation,
                  auto-scheduler, task detail sidebar.
                </td>
              </tr>
              <tr style={{ borderBottom: "0.5px dotted var(--rule)" }}>
                <td style={{ padding: "8px 0" }}>
                  <strong>Report</strong>
                </td>
                <td>
                  Daily and weekly analytics — schedule adherence, CL balance,
                  context switching, burnout trend.
                </td>
              </tr>
              <tr style={{ borderBottom: "0.5px dotted var(--rule)" }}>
                <td style={{ padding: "8px 0" }}>
                  <strong>Profile</strong>
                </td>
                <td>
                  View and edit onboarding data — sleep cycle, energy windows,
                  availability blocks.
                </td>
              </tr>
              <tr>
                <td style={{ padding: "8px 0" }}>
                  <strong>Guide</strong>
                </td>
                <td>This page. System reference manual.</td>
              </tr>
            </tbody>
          </table>
        </section>

        {/* ── 8. Task Lifecycle ── */}
        <section style={{ marginBottom: 48 }}>
          <h2
            style={{
              fontSize: 24,
              marginBottom: 16,
              borderBottom: "0.5px solid var(--rule)",
              paddingBottom: 8,
            }}
          >
            8. Task Lifecycle States
          </h2>
          <p style={{ marginBottom: 16, lineHeight: 1.6 }}>
            Every task progresses through a deterministic state machine:
          </p>

          <div className="trace-log" style={{ padding: "20px" }}>
            <div
              style={{
                fontFamily: "var(--mono)",
                fontSize: 12,
                lineHeight: 2.4,
              }}
            >
              <div className="log-line">
                UNSCHEDULED → task created, awaiting placement
              </div>
              <div className="log-line rule">
                SCHEDULED → placed into matrix slot by engine or manual drag
              </div>
              <div className="log-line rule">
                IN_PROGRESS → current time block matches scheduled slot
              </div>
              <div className="log-line" style={{ color: "var(--safe)" }}>
                COMPLETED → user marked as done (✓)
              </div>
              <div className="log-line">
                RESCHEDULED → moved to different slot post-scheduling
              </div>
              <div className="log-line" style={{ color: "var(--watch)" }}>
                SKIPPED → time block passed without completion
              </div>
              <div className="log-line conflict">
                SACRIFICED → CL reduced by 30% via conflict resolution
              </div>
              <div className="log-line">
                DEADLINE_EXTENDED → deadline constraint relaxed via conflict
                resolution
              </div>
            </div>
          </div>
        </section>

        <div
          style={{
            marginTop: 80,
            borderTop: "0.5px solid var(--rule)",
            paddingTop: 32,
            display: "flex",
            justifyContent: "space-between",
          }}
        >
          <div className="meta-text">Axiom v0.2</div>
          <a href="/dashboard" className="btn btn-sm">
            Return to Dashboard
          </a>
        </div>
      </main>
    </div>
  );
}

export default function TutorialPage() {
  return (
    <AppProvider>
      <TutorialContent />
    </AppProvider>
  );
}
