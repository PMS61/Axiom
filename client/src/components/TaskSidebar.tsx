/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Task Summary Sidebar
   Left column showing task pool overview, today's stats,
   and quick-reference information.
   ═══════════════════════════════════════════════════════════ */

"use client";

import { useApp } from "@/lib/store";
import { TASK_TYPE_LABELS } from "@/lib/types";

export default function TaskSidebar() {
  const { state, dispatch } = useApp();
  const tasks = state.tasks;

  // Compute stats
  const totalCL = tasks
    .filter((t) => t.scheduledSlot && t.etask > 0)
    .reduce((sum, t) => sum + t.etask, 0);
  const recoveryCL = tasks
    .filter((t) => t.etask < 0)
    .reduce((sum, t) => sum + t.etask, 0);
  const netCL = +(totalCL + recoveryCL).toFixed(1);
  const scheduledCount = tasks.filter((t) => t.state === "scheduled").length;
  const completedCount = tasks.filter((t) => t.state === "completed").length;
  const skippedCount = tasks.filter((t) => t.state === "skipped").length;
  const unscheduledCount = tasks.filter((t) => t.state === "unscheduled").length;

  // Group by subject
  const subjectMap = new Map<string, number>();
  for (const t of tasks) {
    if (t.subject) {
      subjectMap.set(t.subject, (subjectMap.get(t.subject) ?? 0) + Math.abs(t.etask));
    }
  }

  return (
    <div className="rule-right" style={{ padding: "var(--sp-4)" }}>
      {/* Today's CL Budget */}
      <div style={{ marginBottom: "var(--sp-6)" }}>
        <h2
          className="font-display"
          style={{
            fontSize: "1rem",
            letterSpacing: "0.08em",
            marginBottom: "var(--sp-3)",
            paddingBottom: "var(--sp-2)",
            borderBottom: "0.5px solid var(--rule)",
          }}
        >
          CL BUDGET
        </h2>
        <div className="responsive-2col" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--sp-4)" }}>
          <div>
            <span className="score-label">TOTAL CL</span>
            <div className="score-large">{totalCL.toFixed(1)}</div>
          </div>
          <div>
            <span className="score-label">RECOVERY</span>
            <div className="score-large" style={{ color: "var(--energy-peak)" }}>
              {recoveryCL.toFixed(1)}
            </div>
          </div>
          <div>
            <span className="score-label">NET CL</span>
            <div
              className="score-large"
              style={{ color: netCL > 40 ? "var(--vermillion)" : undefined }}
            >
              {netCL}
            </div>
          </div>
          <div>
            <span className="score-label">TASKS</span>
            <div className="score-large">{tasks.length}</div>
          </div>
        </div>
      </div>

      {/* Task States */}
      <div style={{ marginBottom: "var(--sp-6)" }}>
        <h2
          className="font-display"
          style={{
            fontSize: "1rem",
            letterSpacing: "0.08em",
            marginBottom: "var(--sp-3)",
            paddingBottom: "var(--sp-2)",
            borderBottom: "0.5px solid var(--rule)",
          }}
        >
          TASK STATES
        </h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--sp-2)" }}>
          <div className="rule-bottom" style={{ display: "flex", justifyContent: "space-between", paddingBottom: "var(--sp-2)" }}>
            <span className="task-meta">SCHEDULED</span>
            <span className="font-mono" style={{ fontWeight: 600 }}>{scheduledCount}</span>
          </div>
          <div className="rule-bottom" style={{ display: "flex", justifyContent: "space-between", paddingBottom: "var(--sp-2)" }}>
            <span className="task-meta">COMPLETED</span>
            <span className="font-mono" style={{ fontWeight: 600, color: "var(--state-completed)" }}>{completedCount}</span>
          </div>
          <div className="rule-bottom" style={{ display: "flex", justifyContent: "space-between", paddingBottom: "var(--sp-2)" }}>
            <span className="task-meta">SKIPPED</span>
            <span className="font-mono" style={{ fontWeight: 600, color: "var(--state-skipped)" }}>{skippedCount}</span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span className="task-meta">UNSCHEDULED</span>
            <span className="font-mono" style={{ fontWeight: 600 }}>{unscheduledCount}</span>
          </div>
        </div>
      </div>

      {/* CL by Subject */}
      <div style={{ marginBottom: "var(--sp-6)" }}>
        <h2
          className="font-display"
          style={{
            fontSize: "1rem",
            letterSpacing: "0.08em",
            marginBottom: "var(--sp-3)",
            paddingBottom: "var(--sp-2)",
            borderBottom: "0.5px solid var(--rule)",
          }}
        >
          CL BY SUBJECT
        </h2>
        {Array.from(subjectMap.entries())
          .sort((a, b) => b[1] - a[1])
          .map(([subject, cl]) => (
            <div
              key={subject}
              className="rule-bottom"
              style={{
                display: "flex",
                justifyContent: "space-between",
                padding: "var(--sp-2) 0",
              }}
            >
              <span className="task-meta">{subject.toUpperCase()}</span>
              <span className="font-mono" style={{ fontWeight: 500, fontSize: "0.75rem" }}>
                {cl.toFixed(1)}
              </span>
            </div>
          ))}
      </div>

      {/* Type Multiplier Reference */}
      <div style={{ marginBottom: "var(--sp-6)" }}>
        <h2
          className="font-display"
          style={{
            fontSize: "1rem",
            letterSpacing: "0.08em",
            marginBottom: "var(--sp-3)",
            paddingBottom: "var(--sp-2)",
            borderBottom: "0.5px solid var(--rule)",
          }}
        >
          TYPE MULTIPLIERS
        </h2>
        {(Object.entries(TASK_TYPE_LABELS) as [string, string][]).map(
          ([key, label]) => {
            const mul = {
              learning: 1.4,
              problem_solving: 1.3,
              writing: 1.1,
              revision: 0.9,
              reading: 0.8,
              administrative: 0.6,
              recreational: "-0.5 to -2.0",
            }[key];
            return (
              <div
                key={key}
                className="rule-bottom"
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "var(--sp-1) 0",
                }}
              >
                <span className="task-meta">{label.toUpperCase()}</span>
                <span className="font-mono" style={{ fontSize: "0.6875rem" }}>
                  {typeof mul === "number" ? `x${mul}` : mul}
                </span>
              </div>
            );
          },
        )}
      </div>

      {/* Scheduler Controls */}
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--sp-2)" }}>
        <button
          className="btn btn-primary"
          style={{ width: "100%", justifyContent: "center" }}
          onClick={() => dispatch({ type: "RUN_SCHEDULER" })}
        >
          RUN SCHEDULER
        </button>
        <button
          className="btn"
          style={{ width: "100%", justifyContent: "center" }}
          onClick={() => dispatch({ type: "GENERATE_REPORT" })}
        >
          GENERATE REPORT
        </button>
      </div>
    </div>
  );
}
