/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Dashboard
   Landing-page aesthetic: generous whitespace, two-column
   layout, trace-log reasoning panel, chart-style bandwidth.
   ═══════════════════════════════════════════════════════════ */

"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getUserProfile } from "@/app/actions/auth";
import {
  deleteTask,
  getTasks,
  saveTask,
  updateTaskStateAndSlot,
  clearUnscheduledTasks,
  getLatestSchedule,
  triggerReschedule,
} from "@/app/actions/tasks";
import AddTaskModal from "@/components/AddTaskModal";
import BandwidthCurve from "@/components/BandwidthCurve";
import ConflictPanel from "@/components/ConflictPanel";
import Header from "@/components/Header";
import ReasoningChain from "@/components/ReasoningChain";
import TrendsSummary from "@/components/trends/TrendsSummary";
import { formatDateHeading, formatDuration, slotToTime, computeCalibratedMultipliers } from "@/lib/engine";
import { markSectionComplete } from "@/app/actions/tasks";
import { useApp } from "@/lib/store";
import { TASK_TYPE_LABELS } from "@/lib/types";
import type { Task, SectionSchedule, DaySchedule, RunSchedulerResult } from "@/lib/types";
import type { TrendResult } from "@/lib/trend-engine/types";

// ── Section View Component ─────────────────────────────────

const SECTION_COLORS: Record<string, string> = {
  morning: "var(--safe)",
  afternoon: "var(--ink)",
  evening: "var(--muted)",
};

function SectionView({
  section,
  tasks,
  onComplete,
  onMarkSectionDone,
  onSkip,
}: {
  section: SectionSchedule;
  tasks: Task[];
  onComplete: (taskId: string) => Promise<void>;
  onMarkSectionDone: () => Promise<void>;
  onSkip?: (taskId: string) => Promise<void>;
}) {
  const color = SECTION_COLORS[section.section] ?? "var(--ink)";
  const pct = section.axiomBudget > 0
    ? Math.min(100, (section.axiomUsed / section.axiomBudget) * 100)
    : 0;

  return (
    <div style={{ marginBottom: 32 }}>
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 16, marginBottom: 12 }}>
        <div className="meta-text" style={{ color }}>{section.section.toUpperCase()}</div>
        <div className="meta-text" style={{ fontSize: 10, color: "var(--muted)" }}>
          {section.axiomUsed.toFixed(1)} / {section.axiomBudget.toFixed(1)} AXIOMS
        </div>
      </div>

      {/* Axiom progress bar */}
      <div style={{ height: 2, background: "var(--rule)", marginBottom: 16 }}>
        <div style={{
          height: "100%",
          width: `${pct}%`,
          background: pct > 90 ? "var(--vermillion)" : color,
          transition: "width 0.3s",
        }} />
      </div>

      {section.tasks.length === 0 ? (
        <div style={{ padding: "16px 0", color: "var(--muted)", fontSize: 13, textAlign: "center" }}>No tasks in this section.</div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {section.tasks.map((item, idx) => {
            const task = tasks.find((t) => t.id === item.taskId);
            const isCompleted = task?.state === "completed";
            return (
              <div
                key={item.chunkId ?? item.taskId + idx}
                style={{
                  padding: "12px 16px",
                  border: "0.5px solid var(--rule)",
                  background: isCompleted ? "var(--bg)" : "var(--card-bg)",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  opacity: isCompleted ? 0.5 : 1,
                }}
              >
                <div>
                  <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 4 }}>
                    {item.taskName}
                    {item.chunkId && (
                      <span className="meta-text" style={{ marginLeft: 8, fontSize: 10 }}>
                        chunk {item.chunkId.split("_chunk_")[1]}
                      </span>
                    )}
                  </div>
                  <div className="meta-text" style={{ fontSize: 10, color: "var(--muted)" }}>
                    {item.completionTime}m
                    {item.isRecreational
                      ? ` · +${item.axiomGain.toFixed(1)} axiom restore`
                      : ` · ${item.axiomCost.toFixed(2)} axioms`}
                  </div>
                </div>
                {!item.isRecreational && !isCompleted && (
                  <div style={{ display: "flex", gap: 4 }}>
                    <button
                      type="button"
                      className="btn btn-sm"
                      style={{ fontSize: 11, padding: "4px 10px" }}
                      onClick={() => onComplete(item.taskId)}
                    >
                      ✓
                    </button>
                    {onSkip && (
                      <button
                        type="button"
                        className="btn btn-sm"
                        style={{ fontSize: 11, padding: "4px 10px", background: "var(--rule)" }}
                        onClick={() => onSkip(item.taskId)}
                        title="Skip task (re-enters pool)"
                      >
                        →
                      </button>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <button
        type="button"
        className="btn btn-sm"
        style={{ marginTop: 12, fontSize: 10, color: "var(--muted)", width: "100%" }}
        onClick={onMarkSectionDone}
      >
        Mark {section.section} done (record performance)
      </button>
    </div>
  );
}

// ── Dashboard Content ──────────────────────────────────────
import {
  readStoredUserProfile,
  toDashboardProfilePayload,
  toStoredUserProfileFromServerUser,
  writeStoredUserProfile,
} from "@/lib/userProfileStorage";

function DashboardContent({ initialTrends = [] }: { initialTrends?: TrendResult[] }) {
  const { state, dispatch } = useApp();
  const [showReasoning, setShowReasoning] = useState(false);
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [isUnscheduledModalOpen, setIsUnscheduledModalOpen] = useState(false);
  const [isLoadingTasks, setIsLoadingTasks] = useState(true);
  const [isScheduling, setIsScheduling] = useState(false);
  const [scheduleError, setScheduleError] = useState<string | null>(null);
  const [viewDay, setViewDay] = useState(0); // which day offset to display

  // Tracks the last-persisted snapshot of each task (id → serialized state+slot)
  const prevTasksSnapRef = useRef<Map<string, string>>(new Map());
  const isInitializedRef = useRef(false);

  // ── Load tasks from DB on mount ────────────────────────
  useEffect(() => {
    if (isInitializedRef.current) return;
    isInitializedRef.current = true;

    const localProfile = readStoredUserProfile();
    if (localProfile) {
      dispatch({
        type: "SET_USER_PROFILE",
        payload: toDashboardProfilePayload(localProfile),
      });
    }

    getTasks().then(async ({ tasks, error }) => {
      setIsLoadingTasks(false);
      if (error || !tasks) return;
      
      dispatch({ type: "INIT_TASKS", payload: tasks });
      
      if (tasks.length > 0) {
        prevTasksSnapRef.current = new Map(
          tasks.map((t) => [t.id, JSON.stringify({
            s: t.state,
            d: t.difficulty,
            dur: t.completionTime,
            p: t.priority,
            n: t.name,
            slot: t.scheduledSlot ?? null,
          })]),
        );

        // Read the persisted schedule — no re-computation on load
        const { days, error: schedErr } = await getLatestSchedule();
        if (!schedErr && days && days.length > 0) {
          dispatch({
            type: "SET_SECTIONS",
            payload: {
              days,
              weights: { morning: 0.40, afternoon: 0.35, evening: 0.25 },
              log: [],
            },
          });
        }
      }
    });

    getUserProfile().then(({ user, error }) => {
      if (error || !user) return;

      const normalized = toStoredUserProfileFromServerUser(user);
      if (normalized) {
        writeStoredUserProfile(normalized);
        dispatch({
          type: "SET_USER_PROFILE",
          payload: toDashboardProfilePayload(normalized),
        });
        return;
      }

      dispatch({ type: "SET_USER_PROFILE", payload: user });
    });
  }, [dispatch]);

  // ── Persist changed tasks to DB ────────────────────────
  useEffect(() => {
    if (isLoadingTasks) return;

    const snap = prevTasksSnapRef.current;
    const toSave: Task[] = [];

    for (const task of state.tasks) {
      const key = JSON.stringify({
        s: task.state,
        d: task.difficulty,
        dur: task.completionTime,
        p: task.priority,
        n: task.name,
        slot: task.scheduledSlot ?? null,
      });
      if (snap.get(task.id) !== key) {
        toSave.push(task);
        snap.set(task.id, key);
      }
    }

    for (const task of toSave) {
      saveTask(task); // fire-and-forget; UI already updated
    }
  }, [state.tasks, isLoadingTasks]);

  // ── Axiom-based section scheduler ────────────────────────
  // Delegates to the server-side deterministic scheduler via triggerReschedule,
  // then fetches the persisted result from the `schedules` table.
  const runAxiomScheduler = useCallback(async () => {
    setIsScheduling(true);
    setScheduleError(null);

    try {
      const { error } = await triggerReschedule();
      if (error) {
        setScheduleError(error);
        setIsScheduling(false);
        return;
      }

      const [{ days, error: schedErr }, { tasks, error: tasksErr }] = await Promise.all([
        getLatestSchedule(),
        getTasks(),
      ]);
      if (schedErr) {
        setScheduleError(schedErr);
        setIsScheduling(false);
        return;
      }

      dispatch({
        type: "SET_SECTIONS",
        payload: {
          days: days ?? [],
          weights: { morning: 0.40, afternoon: 0.35, evening: 0.25 },
          log: [],
        },
      });
      if (!tasksErr && tasks) {
        dispatch({ type: "INIT_TASKS", payload: tasks });
      }
      dispatch({ type: "SET_DAY_PHASE", payload: "active" });
    } catch (err) {
      console.error("Scheduler error:", err);
      setScheduleError("Scheduler failed. Please try again.");
    } finally {
      setIsScheduling(false);
    }
  }, [dispatch]);

  const todaySchedule = useMemo(() => {
    const today = state.scheduledDays.find((d) => d.dayOffset === 0);

    if (today) {
      const sectionFallbackStart = (section: string): number => {
        if (section === "morning") return 24;
        if (section === "afternoon") return 48;
        return 72;
      };

      const sectionItems = today.sections.flatMap((section) =>
        section.tasks
          .filter((item) => !item.isRecreational)
          .map((item) => {
            const task = state.tasks.find((t) => t.id === item.taskId);
            const startSlot = item.startSlot ?? sectionFallbackStart(section.section);
            const endSlot = startSlot + Math.ceil(item.completionTime / 15);

            return {
              task,
              taskId: item.taskId,
              startSlot,
              endSlot,
              axiomCost: item.axiomCost,
              isRecreational: item.isRecreational,
            };
          }),
      );

      const activeEntries = sectionItems.filter(
        (entry) => entry.task && entry.task.state !== "completed" && entry.task.state !== "skipped",
      );
      const completedEntries = sectionItems.filter(
        (entry) => entry.task && entry.task.state === "completed",
      );

      const totalCL = sectionItems.reduce((sum, entry) => {
        if (!entry.task) return sum;
        return entry.task.etask > 0 ? sum + entry.task.etask : sum;
      }, 0);

      const recoveryCL = Math.abs(
        sectionItems.reduce((sum, entry) => {
          if (!entry.task) return sum;
          return entry.task.etask < 0 ? sum + entry.task.etask : sum;
        }, 0),
      );

      return {
        entries: sectionItems.sort((a, b) => a.startSlot - b.startSlot),
        activeCount: activeEntries.length,
        completedCount: completedEntries.length,
        totalCL,
        recoveryCL,
      };
    }

    const fallbackEntries = state.tasks
      .filter((t) => t.scheduledSlot && t.scheduledSlot.day === 0 && t.state !== "unscheduled")
      .map((task) => ({
        task,
        taskId: task.id,
        startSlot: task.scheduledSlot?.startSlot ?? 0,
        endSlot: task.scheduledSlot?.endSlot ?? 0,
        axiomCost: task.etask,
        isRecreational: task.etask < 0,
      }))
      .sort((a, b) => a.startSlot - b.startSlot);

    const totalCL = state.tasks
      .filter((t) => t.etask > 0 && t.scheduledSlot && t.scheduledSlot.day === 0)
      .reduce((s, t) => s + t.etask, 0);

    const recoveryCL = Math.abs(
      state.tasks
        .filter((t) => t.etask < 0 && t.scheduledSlot && t.scheduledSlot.day === 0)
        .reduce((s, t) => s + t.etask, 0),
    );

    return {
      entries: fallbackEntries,
      activeCount: fallbackEntries.filter((e) => e.task.state !== "completed" && e.task.state !== "skipped").length,
      completedCount: fallbackEntries.filter((e) => e.task.state === "completed").length,
      totalCL,
      recoveryCL,
    };
  }, [state.scheduledDays, state.tasks]);

  const scheduled = todaySchedule.entries;
  const totalCL = todaySchedule.totalCL;
  const recoveryCL = todaySchedule.recoveryCL;

  // Section view helpers
  const hasSections = state.scheduledDays.length > 0;
  const viewDayData: DaySchedule | undefined = state.scheduledDays.find(
    (d) => d.dayOffset === viewDay,
  );

  return (
    <div style={{ minHeight: "100vh" }}>
      <Header />

      {state.burnoutRisk !== "safe" && (
        <div style={{
          background: state.burnoutRisk === "critical" ? "var(--vermillion)" : "var(--card-bg)",
          color: state.burnoutRisk === "critical" ? "var(--bg)" : "var(--vermillion)",
          padding: "12px 24px",
          textAlign: "center",
          borderBottom: "0.5px solid var(--rule)",
          fontFamily: "var(--mono)",
          fontSize: 13,
          fontWeight: 600
        }}>
          [!] BURNOUT RISK STATE: {state.burnoutRisk.toUpperCase()} — {
            state.burnoutRisk === "critical"
              ? "Critical overload detected. High CL tasks are now blocked from scheduling."
              : state.burnoutRisk === "warning"
                ? "Elevated continuous load. Forced light day recommended within 72 hours."
                : "Watch state active. Monitoring aggregate duration."
          }
        </div>
      )}

      {/* ── Hero Section: Date + Stats ── */}
      <section className="container section-rule" style={{ paddingTop: 60, paddingBottom: 40 }}>
        <div className="responsive-grid-hero" style={{ display: "grid", gridTemplateColumns: "1fr 1.5fr", gap: 60, alignItems: "start" }}>
          <div>
            <div className="meta-text" style={{ marginBottom: 16 }}>
              {state.dayPhase.toUpperCase()} PHASE
            </div>
            <h1 style={{ fontSize: 42, marginBottom: 24 }}>
              {formatDateHeading(state.currentDate)}
            </h1>
            <p style={{ color: "var(--muted)", maxWidth: 380, marginBottom: 24 }}>
              {todaySchedule.activeCount} tasks scheduled · {todaySchedule.completedCount} completed
            </p>

            <div
              className="meta-text responsive-grid-stats"
              style={{
                borderTop: "0.5px solid var(--rule)",
                paddingTop: 16,
                display: "grid",
                gridTemplateColumns: "1fr 1fr 1fr",
                gap: 16,
              }}
            >
              <div>
                <div style={{ fontSize: 10, color: "var(--muted)" }}>Total CL</div>
                <div style={{ fontWeight: 700, fontSize: 20, marginTop: 4, color: "var(--ink)" }}>
                  {totalCL.toFixed(1)}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 10, color: "var(--muted)" }}>Recovery</div>
                <div style={{ fontWeight: 700, fontSize: 20, marginTop: 4, color: "var(--safe)" }}>
                  -{recoveryCL.toFixed(1)}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 10, color: "var(--muted)" }}>Net</div>
                <div style={{
                  fontWeight: 700,
                  fontSize: 20,
                  marginTop: 4,
                  color: (totalCL - recoveryCL) > 35 ? "var(--vermillion)" : "var(--ink)",
                }}>
                  {(totalCL - recoveryCL).toFixed(1)}
                </div>
              </div>
            </div>
          </div>

          {/* Bandwidth chart */}
          <div>
            <div className="meta-text" style={{ marginBottom: 12, display: "flex", justifyContent: "space-between" }}>
              <span>Mental Energy / Today</span>
              <span style={{ color: "var(--vermillion)" }}>CL Demand</span>
            </div>
            <BandwidthCurve />
          </div>
        </div>
      </section>

      {/* ── Scheduler Controls ── */}
      <section className="container section-rule" style={{ padding: "16px 24px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          <button
            className="btn btn-sm btn-primary"
            onClick={runAxiomScheduler}
            disabled={isScheduling}
            style={{ opacity: isScheduling ? 0.6 : 1, cursor: isScheduling ? "not-allowed" : "pointer" }}
          >
            {isScheduling ? "Scheduling…" : "Run Axiom Scheduler"}
          </button>
          {hasSections && (
            <button
              className="btn btn-sm"
              onClick={() => dispatch({ type: "RECALIBRATE" })}
              title="Insert recovery blocks for High CL tasks"
            >
              Recalibrate
            </button>
          )}
          {hasSections && (
            <button
              className="btn btn-sm"
              onClick={() => dispatch({ type: "CLEAR_SCHEDULED" })}
              style={{ color: "var(--muted)" }}
            >
              Clear Schedule
            </button>
          )}
          <button
            className="btn btn-sm"
            onClick={() => setShowReasoning(!showReasoning)}
          >
            {showReasoning ? "Hide Reasoning" : `Reasoning Chain (${state.reasoningChain.length})`}
          </button>
          {isLoadingTasks && (
            <span className="meta-text" style={{ color: "var(--muted)" }}>Loading tasks…</span>
          )}
        </div>
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          {scheduleError && (
            <span className="meta-text" style={{ color: "var(--vermillion)", fontSize: 11 }}>
              [!] {scheduleError}
            </span>
          )}
          <button
            className="btn btn-sm"
            style={{ background: "var(--card-bg)" }}
            onClick={() => setIsUnscheduledModalOpen(true)}
          >
            Unscheduled Pool ({state.tasks.filter((t) => t.state === "unscheduled").length})
          </button>
        </div>
      </section>

      {/* ── Main content ── */}
      <section
        className={`container ${showReasoning ? 'responsive-grid-split' : ''}`}
        style={{
          paddingTop: 40,
          paddingBottom: 80,
          display: "grid",
          gridTemplateColumns: showReasoning ? "1fr 1fr" : "1fr",
          gap: 60,
        }}
      >
        {/* Schedule List */}
        <div>
          {/* Current Task Box */}
          {(() => {
            const now = new Date();
            const currentSlot = Math.floor((now.getHours() * 60 + now.getMinutes()) / 15);

            const taskInCurrentSlot = scheduled.find((entry) =>
              currentSlot >= entry.startSlot && currentSlot < entry.endSlot,
            );

            const currentTask =
              taskInCurrentSlot?.task && taskInCurrentSlot.task.state !== "completed"
                ? taskInCurrentSlot.task
                : null;

            if (!currentTask) {
              return (
                <div style={{ marginBottom: 48 }}>
                  <div className="meta-text" style={{ marginBottom: 16 }}>Current Focus</div>
                  <div style={{
                    background: "var(--bg)",
                    border: "0.5px dashed var(--muted)",
                    padding: "32px 24px",
                    textAlign: "center"
                  }}>
                    <div className="meta-text" style={{ marginBottom: 24, color: "var(--muted)" }}>
                      Slot {currentSlot} Active · No tasks scheduled for current time blocks.
                    </div>
                    <div style={{ display: "flex", gap: 16, justifyContent: "center" }}>
                      <button className="btn btn-primary" onClick={() => dispatch({ type: "TOGGLE_ADD_TASK" })}>
                        + Add Task
                      </button>
                    </div>
                  </div>
                </div>
              );
            }

            return (
              <div style={{ marginBottom: 48 }}>
                <div className="meta-text" style={{ marginBottom: 16 }}>Current Focus</div>
                <div style={{
                  background: "var(--card-bg)",
                  border: "0.5px solid var(--rule)",
                  borderLeft: `4px solid ${Math.abs(currentTask.etask) > 7 ? 'var(--vermillion)' : 'var(--ink)'}`,
                  padding: "24px",
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                    <h3 style={{ margin: 0, fontSize: 18, fontWeight: 700 }}>{currentTask.name}</h3>
                    <span className="meta-text">
                      {slotToTime(taskInCurrentSlot!.startSlot)} – {slotToTime(taskInCurrentSlot!.endSlot)}
                    </span>
                  </div>
                  <div className="meta-text" style={{ marginBottom: 24, display: "flex", gap: 12 }}>
                    <span style={{ fontWeight: 700, color: "var(--ink)" }}>Etask {(currentTask.etask ?? 0).toFixed(1)}</span>
                    <span>·</span>
                    <span>{formatDuration(currentTask.completionTime)}</span>
                  </div>
                  <button
                    className="btn btn-primary"
                    style={{ width: "100%" }}
                    onClick={async () => {
                      dispatch({ type: "UPDATE_TASK_STATE", payload: { taskId: currentTask.id, state: "completed" } });
                      await updateTaskStateAndSlot(currentTask.id, "completed", {
                        day: 0,
                        startSlot: taskInCurrentSlot!.startSlot,
                        endSlot: taskInCurrentSlot!.endSlot,
                        fitnessScore: 5,
                      });
                    }}
                  >
                    Mark as Complete ✓
                  </button>
                </div>
              </div>
            );
          })()}

          {/* ── Section-based schedule view ── */}
          {hasSections ? (
            <>
              {/* Day selector */}
              <div style={{ display: "flex", gap: 8, marginBottom: 24, overflowX: "auto" }}>
                {state.scheduledDays.filter(d => d.dayOffset <= 6).map((d) => (
                  <button
                    key={d.dayOffset}
                    className="btn btn-sm"
                    onClick={() => setViewDay(d.dayOffset)}
                    style={{
                      background: viewDay === d.dayOffset ? "var(--ink)" : "var(--card-bg)",
                      color: viewDay === d.dayOffset ? "var(--bg)" : "var(--ink)",
                      whiteSpace: "nowrap",
                      flexShrink: 0,
                    }}
                  >
                    {d.dayOffset === 0 ? "Today" : `+${d.dayOffset}d`} · {d.date.slice(5)}
                  </button>
                ))}
              </div>

              {viewDayData ? (
                <>
                  {/* Day axiom summary */}
                  <div style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(3, 1fr)",
                    gap: 12,
                    marginBottom: 32,
                    padding: "16px",
                    border: "0.5px solid var(--rule)",
                    background: "var(--card-bg)",
                  }}>
                    <div>
                      <div className="meta-text" style={{ fontSize: 9, color: "var(--muted)" }}>AXIOMS USED</div>
                      <div style={{ fontWeight: 700, fontSize: 18, marginTop: 4 }}>{viewDayData.totalAxiomsUsed.toFixed(1)}</div>
                    </div>
                    <div>
                      <div className="meta-text" style={{ fontSize: 9, color: "var(--muted)" }}>REMAINING</div>
                      <div style={{ fontWeight: 700, fontSize: 18, marginTop: 4, color: "var(--safe)" }}>{viewDayData.totalAxiomsRemaining.toFixed(1)}</div>
                    </div>
                    <div>
                      <div className="meta-text" style={{ fontSize: 9, color: "var(--muted)" }}>DIVERSITY H</div>
                      <div style={{
                        fontWeight: 700, fontSize: 18, marginTop: 4,
                        color: viewDayData.diversityEntropy > 0.5 ? "var(--safe)" : "var(--vermillion)",
                      }}>
                        {viewDayData.diversityEntropy.toFixed(3)}
                      </div>
                    </div>
                  </div>

                  {/* Section columns */}
                  <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
                    {viewDayData.sections.map((section) => (
                      <SectionView
                        key={section.section}
                        section={section}
                        tasks={state.tasks}
                        onComplete={async (taskId) => {
                          dispatch({ type: "UPDATE_TASK_STATE", payload: { taskId, state: "completed" } });
                          await updateTaskStateAndSlot(taskId, "completed", undefined);
                        }}
                        onMarkSectionDone={async () => {
                          await markSectionComplete(section.section as any, section.axiomBudget, section.axiomUsed);
                          // Compute and store calibrated multipliers based on completion rates
                          const calibrated = computeCalibratedMultipliers(state.tasks);
                          if (calibrated) {
                            dispatch({ type: "CALIBRATE_MULTIPLIERS", payload: calibrated });
                          }
                        }}
                        onSkip={async (taskId) => {
                          dispatch({ type: "UPDATE_TASK_STATE", payload: { taskId, state: "unscheduled" } });
                          await updateTaskStateAndSlot(taskId, "unscheduled", undefined);
                        }}
                      />
                    ))}
                  </div>
                </>
              ) : (
                <div style={{ padding: "40px 0", textAlign: "center", color: "var(--muted)" }}>
                  No data for this day.
                </div>
              )}
            </>
          ) : (
            <div style={{ padding: "40px 0", textAlign: "center" }}>
              <p style={{ color: "var(--muted)", marginBottom: 16 }}>
                No tasks scheduled yet. Add tasks and run the Axiom Scheduler.
              </p>
              <button className="btn" onClick={() => dispatch({ type: "TOGGLE_ADD_TASK" })}>
                + Add First Task
              </button>
            </div>
          )}
        </div>

        {/* Reasoning chain */}
        {showReasoning && <ReasoningChain />}
      </section>

      {/* ── Trend Intelligence Summary ── */}
      <TrendsSummary trends={initialTrends} />

      <AddTaskModal />
      <ConflictPanel />

      {/* ── Unscheduled Pool Modal ── */}
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
                        if (tasks) dispatch({ type: "INIT_TASKS", payload: tasks });
                      }
                    }
                  }}
                >
                  Clear All
                </button>
              </div>
              <button
                style={{ background: "transparent", border: "none", color: "var(--ink)", fontSize: 24, cursor: "pointer" }}
                onClick={() => setIsUnscheduledModalOpen(false)}
              >
                ✕
              </button>
            </div>
            {state.tasks
              .filter(t => t.state === "unscheduled")
              .sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0))
              .length === 0 ? (
              <p style={{ color: "var(--muted)" }}>No tasks in the pool. Generate a syllabus plan first!</p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {state.tasks
                  .filter(t => t.state === "unscheduled")
                  .sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0))
                  .map(task => (
                  <div
                    key={task.id}
                    style={{ padding: 16, border: "0.5px solid var(--rule)", cursor: "pointer", background: "var(--bg)" }}
                    onClick={() => {
                      setSelectedTask(task);
                      setIsUnscheduledModalOpen(false);
                    }}
                  >
                    <div className="meta-text" style={{ marginBottom: 8, color: "var(--muted)" }}>
                      {TASK_TYPE_LABELS[task.type] ?? task.type.toUpperCase()}
                    </div>
                    <div style={{ fontWeight: 600, fontSize: 16, marginBottom: 8 }}>{task.name}</div>
                    <div style={{ fontSize: 13, color: "var(--muted)", display: "flex", justifyContent: "space-between" }}>
                      <span>Etask: {(task.etask ?? 0).toFixed(1)}</span>
                      <span>{task.completionTime}m</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Task Detail Panel ── */}
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

          <div className="meta-text" style={{ marginBottom: 12 }}>Task Name</div>
          <h3 style={{ fontSize: 16, lineHeight: 1.4, marginBottom: 32, fontWeight: 600, color: "var(--ink)" }}>{selectedTask.name}</h3>

          <div className="meta-text" style={{ marginBottom: 16 }}>Properties</div>
          <div className="trace-log" style={{ padding: 24, marginBottom: 32 }}>
            <div className="log-line rule">State: {selectedTask.state.toUpperCase()}</div>
            <div className="log-line rule">Type: {selectedTask.type}</div>
            <div className="log-line rule">Duration: {selectedTask.completionTime}m</div>
            <div className="log-line rule">Priority: {selectedTask.priority}</div>
            <div className="log-line rule">Difficulty: {selectedTask.difficulty}/10</div>
          </div>

          <div className="meta-text" style={{ marginBottom: 16 }}>Axiom Cost (Etask)</div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 24 }}>
            <span style={{ fontSize: 32, fontWeight: 700, color: Math.abs(selectedTask.etask ?? 0) > 7 ? 'var(--vermillion)' : 'var(--ink)' }}>
              {(selectedTask.etask ?? 0).toFixed(2)}
            </span>
            <span className="meta-text">Etask</span>
          </div>

          {selectedTask.etaskBreakdown && (
            <>
              <div className="meta-text" style={{ marginBottom: 12 }}>Etask Breakdown</div>
              <div style={{ border: "0.5px solid var(--rule)", padding: "16px", background: "var(--bg)" }}>
                <pre style={{ margin: 0, fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)", whiteSpace: "pre-wrap" }}>
                  {JSON.stringify(selectedTask.etaskBreakdown, null, 2)}
                </pre>
              </div>
            </>
          )}

          <div style={{ marginTop: 32, display: "flex", gap: 12, justifyContent: "flex-end" }}>
            {selectedTask.state === "scheduled" && (
              <button
                className="btn"
                onClick={async () => {
                  dispatch({ type: "UPDATE_TASK_STATE", payload: { taskId: selectedTask.id, state: "unscheduled" } });
                  await updateTaskStateAndSlot(selectedTask.id, "unscheduled", undefined);
                  setSelectedTask(null);
                }}
              >
                Unschedule [U]
              </button>
            )}
            {selectedTask.state !== "completed" && (
              <button
                className="btn btn-primary"
                onClick={async () => {
                  dispatch({ type: "UPDATE_TASK_STATE", payload: { taskId: selectedTask.id, state: "completed" } });
                  setSelectedTask(null);
                  await updateTaskStateAndSlot(selectedTask.id, "completed", selectedTask.scheduledSlot ?? undefined);
                }}
              >
                Mark as Complete ✓
              </button>
            )}
            <button
              className="btn"
              style={{ color: "var(--vermillion)", borderColor: "var(--vermillion)" }}
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
  );
}

export default function Dashboard({ initialTrends }: { initialTrends?: TrendResult[] }) {
  return <DashboardContent initialTrends={initialTrends} />;
}
