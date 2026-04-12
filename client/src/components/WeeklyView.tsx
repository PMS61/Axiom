/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Weekly View
   7-column grid showing daily CL distribution and
   task density across the week.
   ═══════════════════════════════════════════════════════════ */

"use client";

import { useApp } from "@/lib/store";
import { formatShortDate } from "@/lib/engine";
import TaskBlock from "./TaskBlock";

const DAY_NAMES = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"];

export default function WeeklyView() {
  const { state } = useApp();

  // Generate 7 days starting from current date
  const today = state.currentDate;
  const weekDays = Array.from({ length: 7 }, (_, i) => {
    const d = new Date(today);
    d.setDate(today.getDate() - today.getDay() + 1 + i); // Mon-Sun
    return d;
  });

  // All tasks (for now they're all day 0 — in a real app they'd be distributed)
  const tasksForDay = (dayIndex: number) => {
    if (dayIndex === new Date().getDay() - 1) {
      return state.tasks
        .filter((t) => t.scheduledSlot)
        .sort((a, b) => (a.scheduledSlot?.startSlot ?? 0) - (b.scheduledSlot?.startSlot ?? 0));
    }
    return [];
  };

  // CL per day
  const clPerDay = weekDays.map((_, i) => {
    const dayTasks = tasksForDay(i);
    return dayTasks.reduce((sum, t) => sum + (t.etask > 0 ? t.etask : 0), 0);
  });

  const maxDayCL = Math.max(...clPerDay, 1);

  return (
    <div style={{ padding: "var(--sp-4)" }}>
      <h2
        className="font-display"
        style={{
          fontSize: "1rem",
          letterSpacing: "0.08em",
          marginBottom: "var(--sp-4)",
          paddingBottom: "var(--sp-2)",
          borderBottom: "0.5px solid var(--rule)",
        }}
      >
        WEEKLY OVERVIEW
      </h2>

      {/* CL histogram */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(7, 1fr)",
          gap: 1,
          marginBottom: "var(--sp-6)",
          height: 80,
          alignItems: "end",
        }}
      >
        {clPerDay.map((cl, i) => {
          const isToday = weekDays[i].toDateString() === new Date().toDateString();
          const height = maxDayCL > 0 ? (cl / maxDayCL) * 60 : 0;
          return (
            <div key={i} style={{ textAlign: "center" }}>
              <div
                className="font-mono"
                style={{
                  fontSize: "0.625rem",
                  marginBottom: "var(--sp-1)",
                  color: cl > 35 ? "var(--vermillion)" : "#9CA3AF",
                  fontWeight: cl > 0 ? 600 : 400,
                }}
              >
                {cl > 0 ? cl.toFixed(1) : "—"}
              </div>
              <div
                style={{
                  height: Math.max(height, 2),
                  background: isToday ? "var(--ink)" : "var(--rule)",
                  width: "100%",
                }}
              />
            </div>
          );
        })}
      </div>

      {/* Day labels */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(7, 1fr)",
          gap: 1,
          marginBottom: "var(--sp-6)",
        }}
      >
        {weekDays.map((d, i) => {
          const isToday = d.toDateString() === new Date().toDateString();
          return (
            <div
              key={i}
              className="rule-top"
              style={{
                textAlign: "center",
                paddingTop: "var(--sp-2)",
              }}
            >
              <div
                className="font-mono"
                style={{
                  fontSize: "0.625rem",
                  fontWeight: isToday ? 700 : 400,
                  letterSpacing: "0.08em",
                  color: isToday ? "var(--ink)" : "#9CA3AF",
                }}
              >
                {DAY_NAMES[i]}
              </div>
              <div
                className="font-mono"
                style={{
                  fontSize: "0.625rem",
                  color: "#9CA3AF",
                }}
              >
                {formatShortDate(d)}
              </div>
            </div>
          );
        })}
      </div>

      {/* Today's tasks list */}
      <div>
        <h3
          className="font-display"
          style={{
            fontSize: "0.875rem",
            letterSpacing: "0.08em",
            marginBottom: "var(--sp-3)",
            paddingBottom: "var(--sp-2)",
            borderBottom: "0.5px solid var(--rule)",
          }}
        >
          TODAY&apos;S TASKS
        </h3>
        {state.tasks
          .filter((t) => t.scheduledSlot)
          .sort((a, b) => (a.scheduledSlot?.startSlot ?? 0) - (b.scheduledSlot?.startSlot ?? 0))
          .map((task) => (
            <TaskBlock key={task.id} task={task} isCompact />
          ))}
      </div>
    </div>
  );
}
