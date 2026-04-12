/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Interactive Scheduler View
   Landing-page aesthetic: clean tabular drag/drop UI, 
   RAG input field, and the 96x7 Matrix.
   ═══════════════════════════════════════════════════════════ */

"use client";

import { useState, useEffect, useRef } from "react";
import { useApp } from "@/lib/store";
import MatrixView from "./MatrixView";
import AddTaskModal from "@/components/AddTaskModal";
import Header from "@/components/Header";
import { TASK_TYPE_LABELS, Task } from "@/lib/types";
import { 
  getTasks, 
  deleteTask, 
  updateTaskStateAndSlot, 
  syncTasks, 
  addTask, 
  saveTask,
  getSource, 
  saveSource, 
  clearSources, 
  clearUnscheduledTasks,
  triggerReschedule,
  getLatestSchedule,
  generateTasksFromRoadmapGoal,
} from "@/app/actions/tasks";
import { getUserProfile } from "@/app/actions/auth";
import { isSlotBlocked } from "@/lib/engine";
import { enrichTask } from "@/lib/scoring";
import { DndContext, DragEndEvent, useDraggable, useSensors, useSensor, PointerSensor } from "@dnd-kit/core";

function DraggablePoolTask({ task, onClick }: { task: Task; onClick: () => void }) {
  const { attributes, listeners, setNodeRef, transform } = useDraggable({
    id: task.id,
    data: { task },
  });

  const style = transform ? {
    transform: `translate3d(${transform.x}px, ${transform.y}px, 0)`,
    zIndex: 1000001,
    opacity: 0.8,
  } : undefined;

  return (
    <div 
      ref={setNodeRef}
      {...listeners}
      {...attributes}
      style={{ 
        padding: 16, border: "0.5px solid var(--rule)", cursor: "grab", background: "var(--bg)",
        transition: "border-color 0.2s",
        ...style
      }} 
      onMouseEnter={(e) => !transform && (e.currentTarget.style.borderColor = "var(--ink)")}
      onMouseLeave={(e) => !transform && (e.currentTarget.style.borderColor = "var(--rule)")}
      onClick={(e) => {
        if (!transform) onClick();
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
        <div className="meta-text" style={{ color: "var(--muted)" }}>
          {TASK_TYPE_LABELS[task.type] || task.type.toUpperCase()}
        </div>
        {/* sequence replaces the old .order field */}
        <span className="meta-text" style={{ fontSize: 10 }}>#{task.sequence ?? "?"}</span>
      </div>
      <div style={{ fontWeight: 600, fontSize: 16, marginBottom: 8 }}>{task.name}</div>
      <div style={{ fontSize: 13, color: "var(--muted)", display: "flex", justifyContent: "space-between" }}>
        {/* etask replaces the old .cl field; completionTime replaces .duration */}
        <span>Etask: {task.etask.toFixed(1)}</span>
        <span>{task.completionTime}m</span>
      </div>
    </div>
  );
}

function SchedulerContent() {
  const { state, dispatch } = useApp();
  const [selectedTask, setSelectedTask] = useState<any>(null);
  const [isEditingTask, setIsEditingTask] = useState(false);
  const [editTaskData, setEditTaskData] = useState<any>({});
  const [isUnscheduledModalOpen, setIsUnscheduledModalOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8,
      },
    })
  );
  
  // ── RAG / Goal Input State ──
  const [ragInput, setRagInput] = useState("");
  const [ragTopic, setRagTopic] = useState("");
  const [ragDeadline, setRagDeadline] = useState("in 2 weeks");
  const [ragStatus, setRagStatus] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [ragProgress, setRagProgress] = useState("");
  const [ragError, setRagError] = useState("");


  useEffect(() => {
    async function load() {
      // ── 1. Load tasks from DB ──
      const { tasks, error } = await getTasks();
      if (!error && tasks) {
        dispatch({ type: "SET_TASKS", payload: tasks });
      }

      // ── 2. Load user profile ──
      const { user, error: profileErr } = await getUserProfile();
      if (!profileErr && user) {
        dispatch({ type: "SET_USER_PROFILE", payload: user });
      }

      // ── 3. Populate Matrix from stored schedule (no re-computation) ──
      // In the new system the schedule is persisted in the `schedules` table by
      // the server action after every task creation / reschedule.  We just read
      // what is already there instead of re-running the scheduler client-side.
      const { days, unscheduled, error: schedErr } = await getLatestSchedule();
      if (!schedErr && days && days.length > 0) {
        dispatch({
          type: "SET_SECTIONS",
          payload: {
            days,
            weights: { morning: 0.4, afternoon: 0.35, evening: 0.25 },
            log: [],
          },
        });
      }
    }
    load();
  }, [dispatch]);

  // ── Auto-schedule: delegates to the server-side deterministic scheduler ──
  // triggerReschedule reads ALL tasks from the DB, re-runs the scheduler, and
  // upserts the schedules table.  We then fetch the result and push to the store.
  const handleAutoSchedule = async () => {
    const today = new Date().toISOString().split("T")[0]!;
    const rescoredPool = state.tasks
      .filter((t) => t.state === "unscheduled")
      .map((t) => {
        const scored = enrichTask(t, today);
        return {
          ...t,
          etask: scored.eTask,
          urgency: scored.urgency,
          score: scored.score,
          daysRemaining: scored.daysRemaining,
        } as Task;
      });

    if (rescoredPool.length > 0) {
      const mergedTasks = state.tasks.map((t) => {
        const rescored = rescoredPool.find((p) => p.id === t.id);
        return rescored ?? t;
      });
      dispatch({ type: "SET_TASKS", payload: mergedTasks });
      await syncTasks(rescoredPool);
    }

    const { error } = await triggerReschedule();
    if (error) {
      console.error("[TasksView] triggerReschedule failed:", error);
      return;
    }
    const [{ days, error: schedErr }, { tasks, error: tasksErr }] = await Promise.all([
      getLatestSchedule(),
      getTasks(),
    ]);

    if (!schedErr) {
      dispatch({
        type: "SET_SECTIONS",
        payload: {
          days: days ?? [],
          weights: { morning: 0.4, afternoon: 0.35, evening: 0.25 },
          log: [],
        },
      });
    }

    if (!tasksErr && tasks) {
      dispatch({ type: "SET_TASKS", payload: tasks });
    }
  };

  const handleMatrixTaskClick = async (taskFromMatrix: any) => {
    const taskId = taskFromMatrix?.originalTaskId ?? taskFromMatrix?.id;
    if (!taskId) return;
    const task = state.tasks.find((t) => t.id === taskId);
    if (!task) return;

    setSelectedTask(task);
  };

  // ── "Generate Plan" button in the RAG/Syllabus panel ──
  // Previously this ran buildTasksFromContext (old RAG engine).  Now it feeds the
  // topic (and optional syllabus pasted into the textarea) through the full AI
  // pipeline: Gemini → DAG nodes → scoring → INSERT tasks → scheduler.
  const handleExtractTasks = async () => {
    if (!ragTopic.trim()) return setRagError("Please enter a topic or learning goal.");

    setRagStatus("loading");
    setRagError("");
    setRagProgress("Generating roadmap with AI...");

    try {
      // The syllabus textarea content is passed as the optional `syllabus` hint
      // so Gemini can use it to tailor the generated DAG nodes.
      const result = await generateTasksFromRoadmapGoal(
        ragTopic.trim(),
        ragDeadline || undefined,
        ragInput.trim() || undefined,   // syllabus text from the textarea
      );

      if (result.error) {
        setRagError(result.error);
        setRagStatus("error");
        return;
      }

      setRagProgress(`Generated ${result.count} tasks! Refreshing...`);

      // Refresh store from DB so the task list and matrix are up to date
      const { tasks, error: fetchErr } = await getTasks();
      if (!fetchErr && tasks) {
        dispatch({ type: "SET_TASKS", payload: tasks });
      }

      const { days, error: schedErr } = await getLatestSchedule();
      if (!schedErr && days && days.length > 0) {
        dispatch({
          type: "SET_SECTIONS",
          payload: {
            days,
            weights: { morning: 0.4, afternoon: 0.35, evening: 0.25 },
            log: [],
          },
        });
      }

      setRagStatus("done");
      setRagInput("");
    } catch (err: any) {
      console.error("Extraction error:", err);
      setRagError(err.message || "Failed to generate tasks.");
      setRagStatus("error");
    }
  };

  async function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event;
    if (over && over.id) {
      const dropId = String(over.id);
      if (dropId.startsWith("slot-")) {
        const parts = dropId.split("-");
        const day = parseInt(parts[1], 10);
        const slot = parseInt(parts[2], 10);
        
        const draggedTask = active.data.current?.task as
          | { originalTaskId?: string }
          | undefined;
        const taskId = draggedTask?.originalTaskId ?? String(active.id);
        const task = state.tasks.find((t) => t.id === taskId);
        
        // Correctly map the day index (0-6) into the actual day of the week (Sun-Sat)
        const today = new Date();
        const dropDate = new Date(today);
        dropDate.setDate(today.getDate() + day);
        const actualDayOfWeek = dropDate.getDay();
        
        // ── Validation: Check if slot is blocked ──
        if (isSlotBlocked(slot, actualDayOfWeek, state.userProfile)) {
          console.warn("[Matrix] Drop rejected: Slot is blocked by availability/sleep constraints.");
          return;
        }
        
        dispatch({
          type: "SCHEDULE_TASK_MANUALLY",
          payload: { taskId, startSlot: slot, day }
        });

        if (task) {
           // completionTime replaces the old .duration field
           const slotsNeeded = Math.ceil(task.completionTime / 15);
           const newSlot = {
             startSlot: slot,
             endSlot: slot + slotsNeeded,
             day,
             fitnessScore: 5.0,
             reasoningSteps: []
           };
           await updateTaskStateAndSlot(taskId, "scheduled", newSlot);
        }
      }
    }
  }

  return (
    <DndContext sensors={sensors} onDragEnd={handleDragEnd}>
      <div style={{ minHeight: "100vh" }}>
        <Header />

        {/* RAG Input Section */}
        <section className="container section-rule" style={{ paddingTop: 40, paddingBottom: 40 }}>
          <div className="meta-text" style={{ marginBottom: 16 }}>Syllabus Plan Generator</div>
          <h2 style={{ marginBottom: 16 }}>Generate Tasks from Syllabus</h2>
          
          {ragStatus === "loading" ? (
            <div style={{ background: "var(--card-bg)", border: "0.5px solid var(--rule)", padding: "48px 24px", textAlign: "center", display: "flex", flexDirection: "column", gap: 24 }}>
              <div style={{ fontFamily: "var(--mono)", fontSize: 13, color: "var(--vermillion)" }}>
                {ragProgress}...
              </div>
              <div style={{ width: "100%", height: 1, background: "var(--rule)", position: "relative", overflow: "hidden" }}>
                <div style={{ 
                  position: "absolute", 
                  height: "100%", 
                  background: "var(--vermillion)", 
                  width: "60%", 
                  left: "-60%",
                  animation: "pgbar 2s ease-in-out infinite"
                }} />
              </div>
              <style jsx>{`
                @keyframes pgbar {
                  0% { left: -60%; width: 30%; }
                  50% { left: 40%; width: 60%; }
                  100% { left: 100%; width: 30%; }
                }
              `}</style>
            </div>
          ) : (
            <div style={{ background: "var(--card-bg)", border: "0.5px solid var(--rule)", padding: 24, display: "flex", flexDirection: "column", gap: 16 }}>
              {ragError && (
                <div style={{ padding: "8px 12px", background: "rgba(255,100,100,0.1)", color: "var(--vermillion)", fontSize: 13, border: "0.5px solid var(--vermillion)", marginBottom: 8 }}>
                  {ragError}
                </div>
              )}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <div>
                  <label className="meta-text" style={{ marginBottom: 8, display: "block" }}>Topic/Goal</label>
                  <input 
                    type="text"
                    placeholder="Graph Algorithms, Midterm Review, etc."
                    value={ragTopic}
                    onChange={(e) => setRagTopic(e.target.value)}
                    style={{ width: "100%", padding: "8px 12px", background: "var(--bg)", border: "0.5px solid var(--rule)", color: "var(--fg)", fontFamily: "var(--mono)", fontSize: 13, outline: "none" }}
                  />
                </div>
                <div>
                  <label className="meta-text" style={{ marginBottom: 8, display: "block" }}>Final Deadline</label>
                  <input 
                    type="text"
                    placeholder="in 2 weeks, in 3 days, etc."
                    value={ragDeadline}
                    onChange={(e) => setRagDeadline(e.target.value)}
                    style={{ width: "100%", padding: "8px 12px", background: "var(--bg)", border: "0.5px solid var(--rule)", color: "var(--fg)", fontFamily: "var(--mono)", fontSize: 13, outline: "none" }}
                  />
                </div>
              </div>
              <textarea 
                placeholder="Paste syllabus segments or course modules here. Axiom will use this as context when generating your learning roadmap..."
                value={ragInput}
                onChange={(e) => setRagInput(e.target.value)}
                style={{ 
                  width: "100%", 
                  minHeight: 120, 
                  border: "none", 
                  background: "transparent",
                  resize: "vertical",
                  fontFamily: "var(--mono)",
                  fontSize: 13,
                  outline: "none",
                  padding: "12px 0"
                }}
              />
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderTop: "0.5px solid var(--rule)", paddingTop: 16 }}>
                <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
                  <span className="meta-text" style={{ color: "var(--muted)" }}>AI-Powered Roadmap Generation (Gemini)</span>
                </div>
                <div style={{ display: "flex", gap: 12 }}>
                  <button 
                    className="btn btn-primary"
                    onClick={handleExtractTasks}
                  >
                    Generate Plan &gt;
                  </button>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Interactive Matrix Drag/Drop Panel */}
        <main className="container" style={{ paddingTop: 40, paddingBottom: 80 }}>
          <div className="meta-text" style={{ marginBottom: 16, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span>Global 96×7 Schedule Matrix</span>
            <div style={{ display: "flex", gap: 12 }}>
              <button 
                className="btn btn-sm" 
                style={{ background: "var(--card-bg)" }} 
                onClick={() => setIsUnscheduledModalOpen(true)}
              >
                Unscheduled Pool ({state.tasks.filter((t) => t.state === "unscheduled").length})
              </button>
              <button className="btn btn-sm btn-primary" onClick={handleAutoSchedule}>Auto Schedule</button>
            </div>
          </div>
          <MatrixView onTaskClick={handleMatrixTaskClick} />
        </main>

        {/* Unscheduled Pool Modal */}
        {isUnscheduledModalOpen && (
          <div style={{
            position: "fixed",
            top: 0, left: 0, right: 0, bottom: 0,
            background: "rgba(10,10,12,0.8)",
            backdropFilter: "blur(4px)",
            display: "flex",
            justifyContent: "flex-end",
            zIndex: 99999
          }}>
            <div style={{
              background: "var(--card-bg)",
              width: "100%",
              maxWidth: 400,
              borderLeft: "0.5px solid var(--rule)",
              display: "flex",
              flexDirection: "column",
              height: "100%",
              padding: "32px 24px",
              overflowY: "auto"
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 32 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <h2 style={{ margin: 0, fontSize: 24, fontWeight: 700 }}>Unscheduled Pool</h2>
                  <button 
                    style={{ background: "transparent", border: "0.5px solid var(--rule)", color: "var(--muted)", padding: "4px 8px", fontSize: 11, cursor: "pointer", borderRadius: 4 }}
                    onClick={async () => {
                      if (confirm("Clear ALL unscheduled tasks? This cannot be undone.")) {
                        const { error } = await clearUnscheduledTasks();
                        if (error) alert(error);
                        else {
                          const { tasks } = await getTasks();
                          if (tasks) dispatch({ type: "SET_TASKS", payload: tasks });
                        }
                      }
                    }}
                  >
                    Clear All
                  </button>
                </div>
                <button 
                  style={{ background: "transparent", border: "none", color: "var(--fg)", fontSize: 24, cursor: "pointer" }}
                  onClick={() => setIsUnscheduledModalOpen(false)}
                >
                  ✕
                </button>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {state.tasks
                  .filter(t => t.state === "unscheduled")
                  // sequence replaces the old .order sort key
                  .sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0))
                  .length === 0 ? (
                  <p style={{ color: "var(--muted)" }}>No tasks in the pool. Paste a syllabus to generate a study plan!</p>
                ) : (
                  state.tasks
                    .filter(t => t.state === "unscheduled")
                    .sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0))
                    .map(task => (
                      <DraggablePoolTask 
                        key={task.id} 
                        task={task} 
                        onClick={() => {
                          setSelectedTask(task);
                          setIsUnscheduledModalOpen(false);
                        }} 
                      />
                    ))
                )}
              </div>
            </div>
          </div>
        )}

        <AddTaskModal />

        {selectedTask && (
          <div style={{
            position: "fixed",
            top: 0, right: 0, bottom: 0,
            width: "100%",
            maxWidth: 400,
            background: "var(--card-bg)",
            borderLeft: "0.5px solid var(--rule)",
            zIndex: 100000,
            padding: "32px 24px",
            boxShadow: "-10px 0 40px rgba(0,0,0,0.1)",
            overflowY: "auto"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 32 }}>
              <h2 style={{ fontSize: 24, margin: 0, fontWeight: 700 }}>Task Details</h2>
              <button 
                style={{ cursor: "pointer", background: "transparent", border: "none", fontSize: 20 }}
                onClick={() => setSelectedTask(null)}
              >
                ✕
              </button>
            </div>

            {isEditingTask ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 24, marginBottom: 32 }}>
                <div>
                  <div className="meta-text" style={{ marginBottom: 12 }}>Task Name</div>
                  <input 
                    className="input-base" 
                    value={editTaskData.name || ""} 
                    onChange={e => setEditTaskData({...editTaskData, name: e.target.value})} 
                    style={{ width: "100%", padding: 12, border: "0.5px solid var(--rule)", background: "var(--bg)", color: "var(--ink)", fontWeight: 600 }}
                  />
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                  <div>
                    <div className="meta-text" style={{ marginBottom: 12 }}>Type</div>
                    <select 
                      value={editTaskData.type} 
                      onChange={e => setEditTaskData({...editTaskData, type: e.target.value as any})}
                      style={{ width: "100%", padding: 12, border: "0.5px solid var(--rule)", background: "var(--bg)", color: "var(--ink)" }}
                    >
                      <option value="learning">Learning</option>
                      <option value="problem_solving">Problem Solving</option>
                      <option value="project">Project</option>
                      <option value="revision">Revision</option>
                    </select>
                  </div>
                  <div>
                    <div className="meta-text" style={{ marginBottom: 12 }}>Duration (min)</div>
                    <input 
                      type="number"
                      value={editTaskData.completionTime} 
                      onChange={e => setEditTaskData({...editTaskData, completionTime: parseInt(e.target.value)})}
                      style={{ width: "100%", padding: 12, border: "0.5px solid var(--rule)", background: "var(--bg)", color: "var(--ink)" }}
                    />
                  </div>
                </div>
                <div style={{ display: "flex", gap: 12, marginTop: 12 }}>
                  <button 
                    className="btn btn-primary" 
                    style={{ flex: 1 }}
                    onClick={async () => {
                      const updatedTask = { ...selectedTask, ...editTaskData } as Task;
                      dispatch({ type: "UPDATE_TASK", payload: updatedTask });
                      await saveTask(updatedTask);
                      setSelectedTask(updatedTask);
                      setIsEditingTask(false);
                    }}
                  >
                    Save Changes
                  </button>
                  <button className="btn btn-secondary" style={{ flex: 1 }} onClick={() => setIsEditingTask(false)}>Cancel</button>
                </div>
              </div>
            ) : (
              <>
                <div className="meta-text" style={{ marginBottom: 12, display: "flex", justifyContent: "space-between" }}>
                  <span>Task Name</span>
                  <span 
                    style={{ color: "var(--muted)", cursor: "pointer", textDecoration: "underline", fontSize: 10 }}
                    onClick={() => { setEditTaskData(selectedTask); setIsEditingTask(true); }}
                  >
                    Edit Task
                  </span>
                </div>
                <h3 style={{ fontSize: 16, lineHeight: 1.4, marginBottom: 32, fontWeight: 600, color: "var(--ink)" }}>{selectedTask.name}</h3>
                <div className="trace-log" style={{ padding: 24, marginBottom: 32 }}>
                  <div className="log-line rule">State: {selectedTask.state.toUpperCase()}</div>
                  <div className="log-line rule">Type: {selectedTask.type}</div>
                  {/* completionTime replaces .duration */}
                  <div className="log-line rule">Duration: {selectedTask.completionTime}m</div>
                  <div className="log-line rule">Priority: {selectedTask.priority}</div>
                  <div className="log-line rule">Difficulty: {selectedTask.difficulty}/10</div>
                  {selectedTask.urgency != null && (
                    <div className="log-line rule">Urgency: {selectedTask.urgency.toFixed(3)}</div>
                  )}
                  {selectedTask.score != null && (
                    <div className="log-line rule">Score: {selectedTask.score.toFixed(3)}</div>
                  )}
                </div>
                <div className="meta-text" style={{ marginBottom: 16 }}>Axiom Cost (Etask)</div>
                <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 20 }}>
                  {/* etask replaces .cl */}
                  <span style={{ fontSize: 32, fontWeight: 700 }}>{selectedTask.etask.toFixed(2)}</span>
                </div>
              </>
            )}

            <div style={{ marginTop: 32, display: "flex", justifyContent: "space-between", gap: 12 }}>
              {selectedTask.state === "scheduled" && (
                <button 
                  className="btn btn-primary" style={{ flex: 1 }}
                  onClick={async () => {
                    dispatch({ type: "UPDATE_TASK_STATE", payload: { taskId: selectedTask.id, state: "completed" } });
                    await updateTaskStateAndSlot(selectedTask.id, "completed", selectedTask.scheduledSlot);
                    setSelectedTask(null);
                  }}
                >
                  Mark as Complete ✓
                </button>
              )}
              {selectedTask.state === "scheduled" && (
                <button 
                  className="btn" 
                  style={{ flex: 1 }}
                  onClick={async () => {
                    dispatch({ type: "UPDATE_TASK_STATE", payload: { taskId: selectedTask.id, state: "unscheduled" } });
                    await updateTaskStateAndSlot(selectedTask.id, "unscheduled", undefined);
                    setSelectedTask(null);
                  }}
                >
                  Unschedule [U]
                </button>
              )}
              <button 
                className="btn text-vermillion" 
                style={{ flex: selectedTask.state === "scheduled" ? 0 : 1, borderColor: "var(--vermillion)" }}
                onClick={async () => {
                  dispatch({ type: "DELETE_TASK", payload: selectedTask.id });
                  setSelectedTask(null);
                  await deleteTask(selectedTask.id);
                }}
              >
                Delete Task
              </button>
            </div>
          </div>
        )}
      </div>
    </DndContext>
  );
}

export default function TasksView() {
  return <SchedulerContent />;
}
