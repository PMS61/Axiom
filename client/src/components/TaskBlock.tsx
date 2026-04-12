/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Task Block
   Matches landing page trace-log aesthetic: left-border
   coded lines, clean hover states, generous padding.
   ═══════════════════════════════════════════════════════════ */

"use client";

import { useState, useEffect } from "react";
import { useApp } from "@/lib/store";
import { slotToTime, formatDuration } from "@/lib/engine";
import type { Task } from "@/lib/types";

interface TaskBlockProps {
  task: Task;
  isCompact?: boolean;
}

export default function TaskBlock({ task, isCompact = false }: TaskBlockProps) {
  const { state, dispatch } = useApp();
  const isHighlighted = state.highlightedTaskId === task.id;
  const isRecreational = task.type === "recreational";
  const borderColor = isRecreational
    ? "var(--safe)"
    : Math.abs(task.etask) > 7
      ? "var(--vermillion)"
      : "var(--ink)";

  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    let interval: any;
    if (task.state === "in_progress") {
      interval = setInterval(() => setElapsedSeconds(s => s + 1), 1000);
    }
    return () => clearInterval(interval);
  }, [task.state]);

  const formatTime = (totalSeconds: number) => {
    const h = Math.floor(totalSeconds / 3600);
    const m = Math.floor((totalSeconds % 3600) / 60).toString().padStart(2, '0');
    const s = (totalSeconds % 60).toString().padStart(2, '0');
    return h > 0 ? `${h}:${m}:${s}` : `${m}:${s}`;
  };

  const timeRange = task.scheduledSlot
    ? `${slotToTime(task.scheduledSlot.startSlot)}–${slotToTime(task.scheduledSlot.endSlot)}`
    : "—";

  const isCompleted = task.state === "completed";

  return (
    <div
      className="task-block"
      style={{ 
        borderLeft: `3px solid ${borderColor}`,
        opacity: isCompleted ? 0.6 : 1,
        background: isCompleted ? "var(--bg)" : undefined
      }}
      onMouseEnter={() => dispatch({ type: "HIGHLIGHT_TASK", payload: task.id })}
      onMouseLeave={() => dispatch({ type: "HIGHLIGHT_TASK", payload: null })}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        {/* Left: CL + Name */}
        <div style={{ display: "flex", gap: 16, alignItems: "baseline" }}>
          <span
            style={{
              fontFamily: "var(--mono)",
              fontSize: 24,
              fontWeight: 700,
              lineHeight: 1,
              minWidth: 48,
              color: isRecreational
                ? "var(--safe)"
                : Math.abs(task.etask) > 7
                  ? "var(--vermillion)"
                  : "var(--ink)",
              textDecoration: isCompleted ? "line-through" : "none"
            }}
          >
            {task.etask.toFixed(1)}
          </span>
          <div>
            <div style={{ 
              fontWeight: 500, 
              fontSize: 13, 
              marginBottom: 4,
              textDecoration: isCompleted ? "line-through" : "none",
              color: isCompleted ? "var(--muted)" : "inherit"
            }}>
              {task.name}
            </div>
            {!isCompact && (
              <div className="meta-text">
                {timeRange} · {formatDuration(task.completionTime)}
                {task.subject && ` · ${task.subject}`}
              </div>
            )}
          </div>
        </div>

        {/* Right: State tag & Timer */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8 }}>
          <span className={`status-tag status-${task.state === "in_progress" ? "inprogress" : task.state}`}>
            {task.state === "in_progress"
              ? "Active"
              : task.state.replace(/_/g, " ")}
          </span>
          {(task.state === "in_progress" || elapsedSeconds > 0) && (
            <div style={{ fontFamily: "var(--mono)", fontSize: 13, fontWeight: 700, color: task.state === "in_progress" ? "var(--vermillion)" : "var(--ink)" }}>
              {formatTime(elapsedSeconds)}
            </div>
          )}
        </div>
      </div>

      {/* Expanded detail on hover */}
      {isHighlighted && !isCompact && (
        <div style={{ marginTop: 12, paddingTop: 12, borderTop: "0.5px solid var(--rule)" }}>
          {/* CL breakdown as a trace-log line */}
          {task.etaskBreakdown && (
            <div className="log-line" style={{ fontSize: 11, padding: "0 0 0 16px", marginLeft: 8, lineHeight: 2 }}>
              Etask = {task.etaskBreakdown.baseDifficulty} × {task.etaskBreakdown.durationWeight} × {task.etaskBreakdown.deadlineUrgency} × {task.etaskBreakdown.typeMultiplier} × {task.etaskBreakdown.priorityWeight} = {task.etask}
            </div>
          )}

          {/* Actions */}
          {(task.state === "scheduled" || task.state === "in_progress" || task.state === "unscheduled") && (
            <div style={{ display: "flex", gap: 8, marginTop: 12, marginLeft: 24 }}>
              {(task.state === "scheduled" || task.state === "unscheduled") && (
                <button
                  className="btn btn-sm"
                  onClick={async () => {
                    dispatch({ type: "UPDATE_TASK_STATE", payload: { taskId: task.id, state: "in_progress" } });
                    const { updateTaskStateAndSlot } = await import("@/app/actions/tasks");
                    await updateTaskStateAndSlot(task.id, "in_progress", task.scheduledSlot ?? undefined);
                  }}
                >
                  START
                </button>
              )}
              <button
                className="btn btn-sm"
                onClick={async () => {
                  dispatch({ type: "UPDATE_TASK_STATE", payload: { taskId: task.id, state: "completed" } });
                  const { updateTaskStateAndSlot } = await import("@/app/actions/tasks");
                  await updateTaskStateAndSlot(task.id, "completed", task.scheduledSlot ?? undefined);
                }}
              >
                COMPLETE
              </button>
              {task.state === "scheduled" && (
                <button
                  className="btn btn-sm"
                  onClick={async () => {
                    dispatch({ type: "UPDATE_TASK_STATE", payload: { taskId: task.id, state: "skipped" } });
                    const { updateTaskStateAndSlot } = await import("@/app/actions/tasks");
                    await updateTaskStateAndSlot(task.id, "skipped", task.scheduledSlot ?? undefined);
                  }}
                >
                  SKIP
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
