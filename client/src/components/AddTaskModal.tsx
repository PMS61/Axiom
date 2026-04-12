/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Add Task Modal
   Landing-page aesthetic: generous padding, meta-text
   labels, clean inputs, cta-btn style actions.
   ═══════════════════════════════════════════════════════════ */

"use client";

import { useState } from "react";
import { useApp } from "@/lib/store";
import { TASK_TYPE_LABELS } from "@/lib/types";
import type { TaskType, TaskPriority } from "@/lib/types";

import { addTaskToPool, getTasks } from "@/app/actions/tasks";

export default function AddTaskModal() {
  const { state, dispatch } = useApp();

  const [name, setName] = useState("");
  const [type, setType] = useState<TaskType>("problem_solving");
  const [difficulty, setDifficulty] = useState(5);
  const [duration, setDuration] = useState(60);
  const [priority, setPriority] = useState<TaskPriority>("normal");
  const [subject, setSubject] = useState("");
  const [deadline, setDeadline] = useState("");
  const [energyRecovery, setEnergyRecovery] = useState(-1.5);

  if (!state.isAddTaskOpen) return null;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;

    // Close the modal immediately so the user gets feedback
    dispatch({ type: "TOGGLE_ADD_TASK" });

    // ── Direct Scoring Pipeline ────────────────────────────
    // Bypasses the roadmap/AI generation entirely.
    // The task is scored and added to unscheduled pool only.
    const result = await addTaskToPool({
      name: name.trim(),
      type,
      difficulty,
      completionTime: duration,
      priority,
      subject: subject.trim() || undefined,
      deadline: deadline || undefined,
    });

    if (result.error) {
      console.error("[AddTaskModal] addTaskToPool failed:", result.error);
      alert(`Failed to add task: ${result.error}`);
    } else {
      console.log(`[AddTaskModal] Task "${name.trim()}" scored and added to unscheduled pool.`);
    }

    // Refresh the store with whatever is now in the DB
    const { tasks, error: fetchErr } = await getTasks();
    if (!fetchErr && tasks) {
      dispatch({ type: "SET_TASKS", payload: tasks });
    }

    // Reset form
    setName("");
    setDifficulty(5);
    setDuration(60);
    setSubject("");
    setDeadline("");
  }

  const typeMultiplier = {
    learning: 1.4, problem_solving: 1.3, writing: 1.1,
    revision: 0.9, reading: 0.8, administrative: 0.6, recreational: 1.0,
  }[type];
  const prioWeight = { high: 1.5, normal: 1.0, low: 0.7 }[priority];

  return (
    <div className="modal-overlay" onClick={() => dispatch({ type: "TOGGLE_ADD_TASK" })}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="meta-text" style={{ marginBottom: 16 }}>New Task</div>
        <h2 style={{ marginBottom: 32 }}>Add to Schedule</h2>

        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: 24 }}>
            <label htmlFor="task-name">Task Name</label>
            <input
              id="task-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Graph Theory — Dijkstra's Implementation"
              autoFocus
            />
          </div>

          <div className="responsive-2col" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 24 }}>
            <div>
              <label htmlFor="task-type">Type</label>
              <select id="task-type" value={type} onChange={(e) => setType(e.target.value as TaskType)}>
                {Object.entries(TASK_TYPE_LABELS).map(([key, label]) => (
                  <option key={key} value={key}>{label}</option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="task-priority">Priority</label>
              <select id="task-priority" value={priority} onChange={(e) => setPriority(e.target.value as TaskPriority)}>
                <option value="high">High (×1.5)</option>
                <option value="normal">Normal (×1.0)</option>
                <option value="low">Low (×0.7)</option>
              </select>
            </div>
          </div>

          <div className="responsive-2col" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 24 }}>
            <div>
              <label htmlFor="task-difficulty">Difficulty: {difficulty}/10</label>
              <input
                id="task-difficulty"
                type="range"
                min={1} max={10}
                value={difficulty}
                onChange={(e) => setDifficulty(Number(e.target.value))}
                style={{ borderBottom: "none", cursor: "pointer", accentColor: "var(--ink)" }}
              />
            </div>
            <div>
              <label htmlFor="task-duration">Duration (minutes)</label>
              <input
                id="task-duration"
                type="number"
                min={15} max={180} step={15}
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
              />
            </div>
          </div>

          <div className="responsive-2col" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 24 }}>
            <div>
              <label htmlFor="task-subject">Subject</label>
              <input
                id="task-subject"
                type="text"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                placeholder="e.g., Algorithms"
              />
            </div>
            <div>
              <label htmlFor="task-deadline">Deadline</label>
              <input
                id="task-deadline"
                type="datetime-local"
                value={deadline}
                onChange={(e) => setDeadline(e.target.value)}
                style={{ colorScheme: "light" }}
              />
            </div>
          </div>

          {type === "recreational" && (
            <div style={{ marginBottom: 24 }}>
              <label htmlFor="task-recovery">Energy Recovery: {energyRecovery}</label>
              <input
                id="task-recovery"
                type="range"
                min={-3} max={-0.5} step={0.5}
                value={energyRecovery}
                onChange={(e) => setEnergyRecovery(Number(e.target.value))}
                style={{ borderBottom: "none", cursor: "pointer", accentColor: "var(--safe)" }}
              />
              <div className="meta-text" style={{ marginTop: 4 }}>
                Restores {Math.abs(energyRecovery)} CL of bandwidth
              </div>
            </div>
          )}

          {/* CL preview — trace-log style */}
          <div className="trace-log" style={{ padding: 16, marginBottom: 24 }}>
            <div className="log-line rule" style={{ fontSize: 11 }}>
              CL = {difficulty} × {Math.min(duration / 60, 2).toFixed(2)} × {deadline ? "urgency" : "1.00"} × {typeMultiplier} × {prioWeight}
            </div>
          </div>

          <div style={{ display: "flex", gap: 12, justifyContent: "flex-end" }}>
            <button type="button" className="btn" onClick={() => dispatch({ type: "TOGGLE_ADD_TASK" })}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary">
              Add Task
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
