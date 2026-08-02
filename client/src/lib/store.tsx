/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Application Store (React Context)
   Client-side state management for all scheduling data.
   ═══════════════════════════════════════════════════════════ */

"use client";

import { createContext, useContext, useReducer, type ReactNode } from "react";
import type {
  Task,
  TaskType,
  TaskPriority,
  TaskState,
  EnergyLevel,
  DayPhase,
  BurnoutRisk,
  BandwidthCurve,
  ReasoningStep,
  Workspace,
  DailyReport,
  ScheduleConflict,
  DaySchedule,
  AdaptiveRoadmap,
  RoadmapDAGNode,
  RoadmapGoal,
} from "./types";
import {
  computeCL,
  generateDefaultBandwidthCurve,
  applyEnergyMultiplier,
  slotToTime,
  generatePlacementReasoning,
  durationToSlots,
  isSlotBlocked,
  runSchedulingAlgorithm,
  calculateBurnoutRisk,
} from "./engine";
import { generateDailyInsight } from "./templates";
import { runScheduler, taskToRow } from "./schedule";

function determineContentType(timePhase: string): Task["contentType"] {
  switch (timePhase) {
    case "morning": return "shortbits";
    case "afternoon": return "ppt";
    case "evening": return "storytelling";
    default: return "one-shot";
  }
}

// ── State Shape ───────────────────────────────────────────

export interface AppState {
  // Core data
  tasks: Task[];
  workspaces: Workspace[];

  // Energy & bandwidth
  energyLevel: EnergyLevel;
  bandwidthCurve: BandwidthCurve;
  adjustedBandwidthCurve: BandwidthCurve;

  // Day state
  dayPhase: DayPhase;
  currentDate: Date;
  burnoutRisk: BurnoutRisk;

  confirmedSlots: string[]; // keys like "dayIdx_slot"

  // Reasoning
  reasoningChain: ReasoningStep[];
  highlightedTaskId: string | null;

  // Conflicts
  activeConflict: ScheduleConflict | null;

  // Report
  dailyReport: DailyReport | null;

  // UI state
  isAddTaskOpen: boolean;
  selectedView: "day" | "week" | "report";

  // User Profile / Calibration
  userProfile: {
    peakFocusWindows: any[];
    lowEnergyWindows: any[];
    fixedCommitments: any[];
    hardExclusions: any[];
    wakeTime: number;
    sleepTime: number;
  } | null;

  // Section schedule (output of runSchedulerAction)
  scheduledDays: DaySchedule[];

  // Section weights (from DB feedback)
  sectionWeights: { morning: number; afternoon: number; evening: number };

  // Scheduler reasoning log
  schedulerLog: string[];

  // Theme
  theme: "light" | "dark";

  // Calibrated Etask multipliers (learned from completion data)
  calibratedMultipliers: {
    learning: number;
    problem_solving: number;
    writing: number;
    revision: number;
    reading: number;
    administrative: number;
  } | null;

  // ── Adaptive Roadmap Agent ──────────────────────────────
  activeRoadmap: AdaptiveRoadmap | null;
  roadmapGenerating: boolean;

  // ── Voice Layer ─────────────────────────────────────────
  isVoiceActive: boolean;
  voiceTranscript: string;
}

// ── Actions ───────────────────────────────────────────────

type Action =
  | { type: "ADD_TASK"; payload: Task }
  | { type: "UPDATE_TASK"; payload: Task }
  | { type: "UPDATE_TASK_STATE"; payload: { taskId: string; state: TaskState } }
  | { type: "DELETE_TASK"; payload: string }
  | { type: "SET_ENERGY"; payload: EnergyLevel }
  | { type: "SET_DAY_PHASE"; payload: DayPhase }
  | { type: "HIGHLIGHT_TASK"; payload: string | null }
  | { type: "TOGGLE_ADD_TASK" }
  | { type: "SET_VIEW"; payload: "day" | "week" | "report" }
  | { type: "RUN_SCHEDULER" }
  | { type: "SET_CONFLICT"; payload: ScheduleConflict | null }
  | {
      type: "RESOLVE_CONFLICT";
      payload: { taskId: string; resolution: string };
    }
  | { type: "GENERATE_REPORT" }
  | {
      type: "SCHEDULE_TASK_MANUALLY";
      payload: { taskId: string; startSlot: number; day: number };
    }
  | { type: "INIT_TASKS"; payload: Task[] }
  | { type: "BULK_UPDATE_TASKS"; payload: Task[] }
  | { type: "SET_TASKS"; payload: Task[] }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  | { type: "SET_USER_PROFILE"; payload: any }
  | { type: "CLEAR_SCHEDULED" }
  | {
      type: "SET_SECTIONS";
      payload: {
        days: DaySchedule[];
        weights: { morning: number; afternoon: number; evening: number };
        log: string[];
      };
    }
  | { type: "RECALIBRATE" }
  | { type: "TOGGLE_CONFIRM_SLOT"; payload: string }
  | { type: "SET_THEME"; payload: "light" | "dark" }
  | {
      type: "ADAPTIVE_RESCHEDULE";
      payload: {
        reason:
          | "task_completed"
          | "task_skipped"
          | "new_task"
          | "energy_changed";
      };
    }
  | {
      type: "CALIBRATE_MULTIPLIERS";
      payload: {
        learning: number;
        problem_solving: number;
        writing: number;
        revision: number;
        reading: number;
        administrative: number;
      };
    }
  | { type: "RUN_WEEKLY_OPTIMIZATION" }
  // ── Adaptive Roadmap Agent ─────────────────────────────
  | { type: "SET_ROADMAP"; payload: AdaptiveRoadmap }
  | { type: "CLEAR_ROADMAP" }
  | { type: "SET_ROADMAP_GENERATING"; payload: boolean }
  | { type: "UPDATE_NODE_MASTERY"; payload: { nodeId: string; score: number } }
  | { type: "UPDATE_ROADMAP_NODES"; payload: RoadmapDAGNode[] }
  | { type: "LINK_NODE_TO_TASK"; payload: { nodeId: string; taskId: string } }
  // ── Voice Layer ─────────────────────────────────────────
  | { type: "SET_VOICE_ACTIVE"; payload: boolean }
  | { type: "SET_VOICE_TRANSCRIPT"; payload: string };

// ── ID Generator ──────────────────────────────────────────

let _idCounter = 100;
function nextId(): string {
  _idCounter += 1;
  return `task-${_idCounter}`;
}

// ── Reducer ───────────────────────────────────────────────

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "ADD_TASK": {
      const newTask = action.payload;
      const newTasks = [...state.tasks, newTask];

      // Generate reasoning for the add
      const addReasoning: ReasoningStep = {
        number: state.reasoningChain.length + 1,
        text: `NEW TASK: "${newTask.name}" classified. Etask=${newTask.etask}. Type=${newTask.type}. Pool=${newTask.deadline ? "deadline" : newTask.type === "recreational" ? "energy" : "floating"}.`,
      };

      return {
        ...state,
        tasks: newTasks,
        burnoutRisk: calculateBurnoutRisk(newTasks),
        reasoningChain: [...state.reasoningChain, addReasoning],
        isAddTaskOpen: false,
      };
    }

    case "UPDATE_TASK": {
      const updatedTask = action.payload;
      const newTasks = state.tasks.map((t) =>
        t.id === updatedTask.id ? updatedTask : t,
      );
      return {
        ...state,
        tasks: newTasks,
        burnoutRisk: calculateBurnoutRisk(newTasks),
      };
    }

    case "SET_TASKS": {
      const newTasks = action.payload;
      return {
        ...state,
        tasks: newTasks,
        burnoutRisk: calculateBurnoutRisk(newTasks),
      };
    }

    case "SET_USER_PROFILE":
      return {
        ...state,
        userProfile: {
          peakFocusWindows: action.payload.peak_focus_windows || [],
          lowEnergyWindows: action.payload.low_energy_windows || [],
          fixedCommitments: action.payload.fixed_commitments || [],
          hardExclusions: action.payload.hard_exclusions || [],
          wakeTime: action.payload.wake_time || 420,
          sleepTime: action.payload.sleep_time || 1380,
        },
      };

    case "UPDATE_TASK_STATE": {
      const { taskId, state: newState } = action.payload;
      let newScheduledDays = state.scheduledDays;

      // Only remove if moved to unscheduled. Completed tasks should stay in place to render grayed out in Dashboard.
      if (newState === "unscheduled") {
        newScheduledDays = state.scheduledDays.map((day) => ({
          ...day,
          sections: day.sections.map((sec) => ({
            ...sec,
            tasks: sec.tasks.filter((t) => t.taskId !== taskId),
          })),
        }));
      }

      return {
        ...state,
        tasks: state.tasks.map((t) =>
          t.id === taskId
            ? {
                ...t,
                state: newState,
                scheduledSlot:
                  newState === "unscheduled" ? undefined : t.scheduledSlot,
              }
            : t,
        ),
        scheduledDays: newScheduledDays,
      };
    }

    case "DELETE_TASK": {
      const taskId = action.payload;
      const newScheduledDays = state.scheduledDays.map((day) => ({
        ...day,
        sections: day.sections.map((sec) => ({
          ...sec,
          tasks: sec.tasks.filter((t) => t.taskId !== taskId),
        })),
      }));

      return {
        ...state,
        tasks: state.tasks.filter((t) => t.id !== taskId),
        scheduledDays: newScheduledDays,
      };
    }

    case "SET_ENERGY": {
      const newCurve = applyEnergyMultiplier(
        state.bandwidthCurve,
        action.payload,
      );
      const energyReasoning: ReasoningStep = {
        number: state.reasoningChain.length + 1,
        text: `ENERGY UPDATE: Level set to ${action.payload}. Bandwidth curve recalculated with ×${action.payload >= 0 ? "+" : ""}${action.payload} multiplier.`,
      };
      return {
        ...state,
        energyLevel: action.payload,
        adjustedBandwidthCurve: newCurve,
        reasoningChain: [...state.reasoningChain, energyReasoning],
      };
    }

    case "SET_DAY_PHASE": {
      return { ...state, dayPhase: action.payload };
    }

    case "HIGHLIGHT_TASK": {
      return { ...state, highlightedTaskId: action.payload };
    }

    case "TOGGLE_ADD_TASK": {
      return { ...state, isAddTaskOpen: !state.isAddTaskOpen };
    }

    case "SET_VIEW": {
      return { ...state, selectedView: action.payload };
    }

    case "RUN_SCHEDULER": {
      const output = runScheduler({
        tasks: state.tasks.map(taskToRow),
        start_date: state.currentDate.toISOString().split("T")[0],
        section_weights: state.sectionWeights,
        wake_time: state.userProfile?.wakeTime ?? 420,
        sleep_time: state.userProfile?.sleepTime ?? 1320,
        fixed_commitments: state.userProfile?.fixedCommitments ?? [],
        hard_exclusions: state.userProfile?.hardExclusions ?? [],
      });

      // Update task states
      const scheduledIds = new Set<string>();
      for (const day of output.days) {
        for (const section of day.sections) {
          for (const item of section.tasks) {
            scheduledIds.add(item.taskId);
          }
        }
      }

      const updatedTasks = state.tasks.map((t) => {
        if (scheduledIds.has(t.id)) {
          return {
            ...t,
            state: "scheduled" as TaskState,
            scheduledSlot: undefined,
          };
        } else if (output.unscheduled.includes(t.id)) {
          return {
            ...t,
            state: "unscheduled" as TaskState,
            scheduledSlot: undefined,
          };
        }
        return t;
      });

      const newSteps = output.reasoningLog.map((text, i) => ({
        number: state.reasoningChain.length + i + 1,
        text,
        isRule:
          text.startsWith("RULE") ||
          text.includes("Budget") ||
          text.includes("Anti-starvation"),
        isAction: text.includes("placed") || text.includes("Schedule"),
      }));

      return {
        ...state,
        tasks: updatedTasks,
        scheduledDays: output.days,
        schedulerLog: output.reasoningLog,
        reasoningChain: [...state.reasoningChain, ...newSteps],
        burnoutRisk: calculateBurnoutRisk(updatedTasks, output.days),
      };
    }

    case "SET_CONFLICT": {
      return { ...state, activeConflict: action.payload };
    }

    case "RESOLVE_CONFLICT": {
      const { taskId, resolution } = action.payload;
      let newTasks = [...state.tasks];
      const tIdx = newTasks.findIndex((t) => t.id === taskId);
      let stepMessage = `CONFLICT RESOLVED: ${resolution.toUpperCase()}`;

      if (tIdx >= 0) {
        const t = newTasks[tIdx];
        if (resolution === "sacrifice") {
          newTasks[tIdx] = {
            ...t,
            etask: t.etask * 0.7,
            state: "sacrificed",
            etaskBreakdown: t.etaskBreakdown
              ? { ...t.etaskBreakdown, total: t.etask * 0.7 }
              : undefined,
          };
          stepMessage = `Depth Sacrificed for "${t.name}". Etask reduced by 30% from ${t.etask.toFixed(1)} to ${(t.etask * 0.7).toFixed(1)}.`;
        } else if (resolution === "extend_deadline") {
          newTasks[tIdx] = { ...t, state: "deadline_extended" };
          stepMessage = `Deadline Extended for "${t.name}" (Allowed due to low/normal priority).`;
        } else if (resolution === "defer") {
          newTasks[tIdx] = { ...t, state: "rescheduled" };
          stepMessage = `Task "${t.name}" deferred to next scheduling loop.`;
        }
      }

      return {
        ...state,
        tasks: newTasks,
        activeConflict: null,
        reasoningChain: [
          ...state.reasoningChain,
          {
            number: state.reasoningChain.length + 1,
            text: stepMessage,
            isRule: true,
          },
        ],
      };
    }

    case "GENERATE_REPORT": {
      const tasks = state.tasks;
      const scheduled = tasks.filter(
        (t) =>
          t.state === "scheduled" ||
          t.state === "completed" ||
          t.state === "skipped",
      );
      const completed = tasks.filter((t) => t.state === "completed");
      const adherence =
        scheduled.length > 0 ? (completed.length / scheduled.length) * 100 : 0;

      const report: DailyReport = {
        date: state.currentDate.toISOString(),
        scheduleAdherence: +adherence.toFixed(0),
        clBalance: 72,
        productiveHoursAccuracy: 68,
        contextSwitchingScore: 81,
        energyManagement: 75,
        deadlineHitRate: 90,
        burnoutRiskTrend: state.burnoutRisk,
        topInsight:
          scheduled.length === 0
            ? "No tasks scheduled. Add tasks and run the scheduler."
            : generateDailyInsight({
                adherencePercentage: +adherence.toFixed(0),
                totalEtask: tasks.reduce((sum, t) => sum + Math.abs(t.etask), 0),
                highCLTasksPlacedInPeakCount: scheduled.filter(
                  (t) => Math.abs(t.etask) > 5,
                ).length,
                unresolvedConflictsCount: state.activeConflict ? 1 : 0,
                contextSwitchPenalty: 3.5,
                burnoutRisk: state.burnoutRisk,
                energyDeficit: 4,
              }),
      };

      return { ...state, dailyReport: report, dayPhase: "complete" };
    }

    case "SCHEDULE_TASK_MANUALLY": {
      const task = state.tasks.find((t) => t.id === action.payload.taskId);
      if (!task) return state;
      const slotsNeeded = durationToSlots(task.completionTime);

      const newReasoning: ReasoningStep = {
        number: state.reasoningChain.length + 1,
        text: `MANUAL SCHEDULING: User dragged "${task.name}" into Matrix view at day ${action.payload.day}, slot ${action.payload.startSlot}.`,
        isAction: true,
        relatedTaskId: task.id,
      };

      const sectionName =
        action.payload.startSlot < 48
          ? "morning"
          : action.payload.startSlot < 72
            ? "afternoon"
            : "evening";

      const updatedTasks = state.tasks.map((t) =>
        t.id === action.payload.taskId
          ? {
              ...t,
              state: "scheduled" as TaskState,
              contentType: determineContentType(sectionName),
              scheduledSlot: {
                startSlot: action.payload.startSlot,
                endSlot: action.payload.startSlot + slotsNeeded,
                day: action.payload.day,
                fitnessScore: 5.0, // Arbitrary for manual override
                reasoningSteps: [newReasoning],
              },
            }
          : t,
      );

      // Also update scheduledDays so it shows up in dashboards
      // 1. First, remove this task from ANY existing days/sections to prevent duplication
      const newScheduledDays = state.scheduledDays.map((day) => ({
        ...day,
        sections: day.sections.map((sec) => ({
          ...sec,
          tasks: sec.tasks.filter((st) => st.taskId !== action.payload.taskId),
        })),
      }));

      // The sectionName is computed above
      let dayData = newScheduledDays.find(
        (d) => d.dayOffset === action.payload.day,
      );
      if (!dayData) {
        // Create a skeleton day if it doesn't exist
        dayData = {
          dayOffset: action.payload.day,
          date: new Date(
            state.currentDate.getTime() + action.payload.day * 86400000,
          )
            .toISOString()
            .split("T")[0],
          sections: [
            {
              section: "morning",
              tasks: [],
              axiomUsed: 0,
              axiomBudget: 20,
              axiomRemaining: 20,
            },
            {
              section: "afternoon",
              tasks: [],
              axiomUsed: 0,
              axiomBudget: 17.5,
              axiomRemaining: 17.5,
            },
            {
              section: "evening",
              tasks: [],
              axiomUsed: 0,
              axiomBudget: 12.5,
              axiomRemaining: 12.5,
            },
          ],
          totalAxiomsUsed: 0,
          totalAxiomsRemaining: 50,
          diversityEntropy: 0,
          loadAcceptable: true,
        };
        newScheduledDays.push(dayData);
      }

      const section = dayData.sections.find((s) => s.section === sectionName);
      if (section) {
        const axiomResult = computeCL(
          task.difficulty,
          task.completionTime,
          0,
          task.type,
          task.priority,
        );
        const axiomCost = axiomResult.total;

        section.tasks.push({
          taskId: task.id,
          taskName: task.name,
          completionTime: task.completionTime,
          axiomCost: axiomCost,
          axiomGain: 0,
          isRecreational: task.type === "recreational",
          startSlot: action.payload.startSlot,
        });

        // Recalculate axiom usage for the section/day
        section.axiomUsed = section.tasks.reduce(
          (sum, t) => sum + t.axiomCost,
          0,
        );
        section.axiomRemaining = +(
          section.axiomBudget - section.axiomUsed
        ).toFixed(2);
        dayData.totalAxiomsUsed = dayData.sections.reduce(
          (sum, s) => sum + s.axiomUsed,
          0,
        );
        dayData.totalAxiomsRemaining = +(50 - dayData.totalAxiomsUsed).toFixed(
          2,
        );
      }

      return {
        ...state,
        tasks: updatedTasks,
        scheduledDays: newScheduledDays,
        reasoningChain: [...state.reasoningChain, newReasoning],
        burnoutRisk: calculateBurnoutRisk(updatedTasks, newScheduledDays),
      };
    }

    case "INIT_TASKS": {
      // Replace sample tasks with DB tasks on first load.
      // If the store already has user-created tasks (not sample data), merge.
      return {
        ...state,
        tasks: action.payload,
        reasoningChain: [
          {
            number: 1,
            text: `SCHEDULER INITIALIZED. Loaded ${action.payload.length} tasks from database.`,
            isRule: true,
          },
        ],
      };
    }

    case "BULK_UPDATE_TASKS": {
      // Merge incoming tasks (from API schedule result) into state.
      // Tasks in payload overwrite existing ones by id.
      const updateMap = new Map(action.payload.map((t) => [t.id, t]));
      const merged = state.tasks.map((t) => updateMap.get(t.id) ?? t);
      // Add any new tasks not already in state
      const existingIds = new Set(state.tasks.map((t) => t.id));
      for (const t of action.payload) {
        if (!existingIds.has(t.id)) merged.push(t);
      }
      return { ...state, tasks: merged };
    }

    case "CLEAR_SCHEDULED": {
      return {
        ...state,
        tasks: state.tasks.map((t) =>
          t.state === "scheduled"
            ? { ...t, state: "unscheduled" as TaskState }
            : t,
        ),
        scheduledDays: [],
        schedulerLog: [],
      };
    }

    case "SET_SECTIONS": {
      const { days, weights, log } = action.payload;
      // Mark tasks that appear in the schedule as scheduled
      const scheduledTaskProps = new Map<string, { state: TaskState, contentType?: Task["contentType"] }>();
      for (const day of days) {
        for (const section of day.sections) {
          const cType = determineContentType(section.section);
          for (const item of section.tasks) {
            scheduledTaskProps.set(item.taskId, { state: "scheduled", contentType: cType });
          }
        }
      }
      return {
        ...state,
        scheduledDays: days,
        sectionWeights: weights,
        schedulerLog: log,
        tasks: state.tasks.map((t) => {
          const props = scheduledTaskProps.get(t.id);
          if (props && t.state === "unscheduled") {
            return { ...t, state: props.state, contentType: props.contentType };
          }
          return t;
        }),
        reasoningChain: [
          ...state.reasoningChain,
          {
            number: state.reasoningChain.length + 1,
            text: `SCHEDULER RUN: ${days.length} days planned. ${log[log.length - 1] ?? ""}`,
            isRule: true,
          },
        ],
      };
    }

    case "RECALIBRATE": {
      // Find tasks with Etask > 8 and ensure they are followed by recovery
      const highCLTasks = state.tasks.filter(
        (t) => Math.abs(t.etask) > 8 && t.state === "scheduled",
      );
      const newRecoveryTasks: Task[] = [];

      highCLTasks.forEach((task) => {
        // Create a recovery block for this task
        const recoveryTask: Task = {
          id: `recovery-${task.id}-${Date.now()}`,
          name: `Focus Reset (Ref: ${task.name})`,
          type: "recreational",
          completionTime: 30,
          priority: "normal",
          difficulty: 1,
          subject: task.subject,
          etask: -5, // Restorative
          etaskBreakdown: {
            baseDifficulty: 1,
            durationWeight: 0.5,
            deadlineUrgency: 1,
            typeMultiplier: -1,
            priorityWeight: 1,
            total: -5,
          },
          state: "unscheduled",
          sequence: (task.sequence ?? 0) + 0.5,
          createdAt: new Date().toISOString(),
        };
        newRecoveryTasks.push(recoveryTask);
      });

      if (newRecoveryTasks.length === 0) return state;

      const updatedTasks = [...state.tasks, ...newRecoveryTasks];
      const output = runScheduler({
        tasks: updatedTasks.map(taskToRow),
        start_date: state.currentDate.toISOString().split("T")[0],
        section_weights: state.sectionWeights,
        wake_time: state.userProfile?.wakeTime ?? 420,
        sleep_time: state.userProfile?.sleepTime ?? 1320,
        fixed_commitments: state.userProfile?.fixedCommitments ?? [],
        hard_exclusions: state.userProfile?.hardExclusions ?? [],
      });

      // Update task states
      const scheduledIds = new Set<string>();
      for (const day of output.days) {
        for (const section of day.sections) {
          for (const item of section.tasks) {
            scheduledIds.add(item.taskId);
          }
        }
      }

      const tasksWithNewStatus = updatedTasks.map((t) => {
        if (scheduledIds.has(t.id)) {
          return {
            ...t,
            state: "scheduled" as TaskState,
            scheduledSlot: undefined,
          };
        } else if (output.unscheduled.includes(t.id)) {
          return {
            ...t,
            state: "unscheduled" as TaskState,
            scheduledSlot: undefined,
          };
        }
        return t;
      });

      const newSteps = output.reasoningLog.map((text, i) => ({
        number: state.reasoningChain.length + i + 1,
        text,
        isRule:
          text.startsWith("RULE") ||
          text.includes("Budget") ||
          text.includes("Anti-starvation"),
        isAction: text.includes("placed") || text.includes("Schedule"),
      }));

      return {
        ...state,
        tasks: tasksWithNewStatus,
        scheduledDays: output.days,
        schedulerLog: output.reasoningLog,
        reasoningChain: [...state.reasoningChain, ...newSteps],
        burnoutRisk: calculateBurnoutRisk(tasksWithNewStatus, output.days),
      };
    }

    case "TOGGLE_CONFIRM_SLOT": {
      const key = action.payload;
      const isConfirmed = state.confirmedSlots.includes(key);
      const newSlots = isConfirmed
        ? state.confirmedSlots.filter((s) => s !== key)
        : [...state.confirmedSlots, key];

      return {
        ...state,
        confirmedSlots: newSlots,
      };
    }

    case "SET_THEME":
      return { ...state, theme: action.payload };

    case "CALIBRATE_MULTIPLIERS": {
      const mult = action.payload;
      const calibrationReasoning: ReasoningStep = {
        number: state.reasoningChain.length + 1,
        text: `Etask CALIBRATION: Updated multipliers based on completion rates — Learning: ${mult.learning.toFixed(2)}, Problem Solving: ${mult.problem_solving.toFixed(2)}, Writing: ${mult.writing.toFixed(2)}, Revision: ${mult.revision.toFixed(2)}, Reading: ${mult.reading.toFixed(2)}, Admin: ${mult.administrative.toFixed(2)}`,
        isRule: true,
      };
      return {
        ...state,
        calibratedMultipliers: mult,
        reasoningChain: [...state.reasoningChain, calibrationReasoning],
      };
    }

    case "ADAPTIVE_RESCHEDULE": {
      const reason = action.payload.reason;

      // Build reason log
      const reasonLog: string[] = [];
      switch (reason) {
        case "task_completed":
          reasonLog.push(
            "ADAPTIVE: Task completed early - checking for free slots",
          );
          break;
        case "task_skipped":
          reasonLog.push(
            "ADAPTIVE: Task skipped - re-entering pool with urgency bump",
          );
          break;
        case "new_task":
          reasonLog.push(
            "ADAPTIVE: New urgent task added - recalculating schedule",
          );
          break;
        case "energy_changed":
          reasonLog.push(
            `ADAPTIVE: Energy level changed to ${state.energyLevel} - recalculating bandwidth`,
          );
          break;
      }

      // Check for forced light day due to burnout (Etask cap at 4.0 when critical)
      let tasksToSchedule = state.tasks;
      if (state.burnoutRisk === "critical") {
        tasksToSchedule = state.tasks.filter((t) => Math.abs(t.etask) <= 4.0);
        reasonLog.push(
          "FORCED LIGHT DAY: Blocking high Etask tasks (>4.0) due to critical burnout risk",
        );
      }

      // Run scheduler on filtered tasks
      const output = runScheduler({
        tasks: tasksToSchedule.map(taskToRow),
        start_date: state.currentDate.toISOString().split("T")[0],
        section_weights: state.sectionWeights,
        wake_time: state.userProfile?.wakeTime ?? 420,
        sleep_time: state.userProfile?.sleepTime ?? 1320,
        fixed_commitments: state.userProfile?.fixedCommitments ?? [],
        hard_exclusions: state.userProfile?.hardExclusions ?? [],
      });

      // Update task states
      const scheduledIds = new Set<string>();
      for (const day of output.days) {
        for (const section of day.sections) {
          for (const item of section.tasks) {
            scheduledIds.add(item.taskId);
          }
        }
      }

      const updatedTasks = state.tasks.map((t) => {
        if (scheduledIds.has(t.id) && t.state === "unscheduled") {
          return {
            ...t,
            state: "scheduled" as TaskState,
            scheduledSlot: undefined,
          };
        } else if (
          output.unscheduled.includes(t.id) &&
          t.state === "scheduled"
        ) {
          return {
            ...t,
            state: "unscheduled" as TaskState,
            scheduledSlot: undefined,
          };
        }
        return t;
      });

      const newSteps = output.reasoningLog.map((text, i) => ({
        number: state.reasoningChain.length + i + 1,
        text: `[ADAPTIVE] ${text}`,
        isRule:
          text.startsWith("RULE") ||
          text.includes("Budget") ||
          text.includes("Anti-starvation"),
        isAction: text.includes("placed") || text.includes("Schedule"),
      }));

      return {
        ...state,
        tasks: updatedTasks,
        scheduledDays: output.days,
        schedulerLog: [
          ...state.schedulerLog,
          ...reasonLog,
          ...output.reasoningLog,
        ],
        reasoningChain: [...state.reasoningChain, ...newSteps],
        burnoutRisk: calculateBurnoutRisk(updatedTasks, output.days),
      };
    }

    case "RUN_WEEKLY_OPTIMIZATION": {
      const tasks = state.tasks;
      const now = new Date();

      // Check for forced light day due to burnout
      if (state.burnoutRisk === "critical") {
        const lightDayReasoning: ReasoningStep = {
          number: state.reasoningChain.length + 1,
          text: "FORCED LIGHT DAY: Burnout risk is critical. High Etask tasks (>4.0) blocked from scheduling until recovery.",
          isRule: true,
        };

        // Filter out high Etask tasks
        const filteredTasks = tasks.map((t) => {
          if (Math.abs(t.etask) > 4.0 && t.state === "unscheduled") {
            return { ...t, state: "unscheduled" as TaskState };
          }
          return t;
        });

        return {
          ...state,
          tasks: filteredTasks,
          reasoningChain: [...state.reasoningChain, lightDayReasoning],
        };
      }

      // Get all tasks with deadlines in next 7 days
      const weekTasks = tasks.filter((t) => {
        if (!t.deadline) return false;
        const deadline = new Date(t.deadline);
        const daysUntil =
          (deadline.getTime() - now.getTime()) / (1000 * 3600 * 24);
        return daysUntil >= 0 && daysUntil <= 7;
      });

      if (weekTasks.length === 0) {
        return state;
      }

      // Rank by urgency × Etask × priority
      const scored = weekTasks
        .map((t) => {
          const deadline = new Date(t.deadline!);
          const daysUntil = Math.max(
            0,
            (deadline.getTime() - now.getTime()) / (1000 * 3600 * 24),
          );
          const priorityNum = { high: 3, normal: 2, low: 1 }[t.priority];
          const urgency = priorityNum / (daysUntil + 1);
          const composite = urgency * Math.abs(t.etask) * priorityNum;
          return { task: t, daysUntil, composite };
        })
        .sort((a, b) => b.composite - a.composite);

      // Distribute across days to balance load
      const dayLoad: Record<number, number> = {};
      for (let i = 0; i < 7; i++) dayLoad[i] = 0;

      const distributed: { taskId: string; day: number }[] = [];
      for (const item of scored) {
        // Find day with lowest load
        let bestDay = 0;
        let minLoad = Infinity;
        for (let d = 0; d <= Math.min(7, Math.ceil(item.daysUntil)); d++) {
          if (dayLoad[d] < minLoad) {
            minLoad = dayLoad[d];
            bestDay = d;
          }
        }
        dayLoad[bestDay] += Math.abs(item.task.etask);
        distributed.push({ taskId: item.task.id, day: bestDay });
      }

      const optReasoning: ReasoningStep = {
        number: state.reasoningChain.length + 1,
        text: `WEEKLY OPTIMIZATION: Distributed ${scored.length} deadline tasks across 7 days. Load balance: ${Object.entries(
          dayLoad,
        )
          .map(([d, l]) => `Day${d}=${l.toFixed(1)}`)
          .join(", ")}`,
        isRule: true,
      };

      return {
        ...state,
        reasoningChain: [...state.reasoningChain, optReasoning],
      };
    }

    // ── Adaptive Roadmap Agent cases ──────────────────────

    case "SET_ROADMAP": {
      return {
        ...state,
        activeRoadmap: action.payload,
        roadmapGenerating: false,
      };
    }

    case "CLEAR_ROADMAP": {
      return { ...state, activeRoadmap: null };
    }

    case "SET_ROADMAP_GENERATING": {
      return { ...state, roadmapGenerating: action.payload };
    }

    case "UPDATE_NODE_MASTERY": {
      if (!state.activeRoadmap) return state;
      const { nodeId, score } = action.payload;

      // Update the node's masteryScore
      const updatedNodes = state.activeRoadmap.nodes.map((n) =>
        n.id === nodeId ? { ...n, masteryScore: score } : n,
      );

      // Recompute nodeStates reactively
      const recomputed = updatedNodes.map((node) => {
        const mastery =
          updatedNodes.find((n) => n.id === node.id)?.masteryScore ?? 0;
        if (mastery >= 0.75) return { ...node, nodeState: "mastered" as const };
        const allPrereqsMet = node.prerequisites.every((prereqId) => {
          const prereq = updatedNodes.find((n) => n.id === prereqId);
          return prereq && prereq.masteryScore >= 0.75;
        });
        if (allPrereqsMet || node.prerequisites.length === 0) {
          return {
            ...node,
            nodeState:
              mastery > 0 ? ("in_progress" as const) : ("available" as const),
          };
        }
        return { ...node, nodeState: "locked" as const };
      });

      return {
        ...state,
        activeRoadmap: {
          ...state.activeRoadmap,
          nodes: recomputed,
          lastUpdatedAt: new Date().toISOString(),
        },
      };
    }

    case "UPDATE_ROADMAP_NODES": {
      if (!state.activeRoadmap) return state;
      return {
        ...state,
        activeRoadmap: {
          ...state.activeRoadmap,
          nodes: action.payload,
          lastUpdatedAt: new Date().toISOString(),
        },
      };
    }

    case "LINK_NODE_TO_TASK": {
      if (!state.activeRoadmap) return state;
      const { nodeId, taskId } = action.payload;
      return {
        ...state,
        activeRoadmap: {
          ...state.activeRoadmap,
          nodes: state.activeRoadmap.nodes.map((n) =>
            n.id === nodeId ? { ...n, taskId } : n,
          ),
        },
      };
    }

    // ── Voice Layer ───────────────────────────────────────

    case "SET_VOICE_ACTIVE":
      return { ...state, isVoiceActive: action.payload };

    case "SET_VOICE_TRANSCRIPT":
      return { ...state, voiceTranscript: action.payload };

    default:
      return state;
  }
}

// ── Initial State ─────────────────────────────────────────

const baseCurve = generateDefaultBandwidthCurve();

const initialState: AppState = {
  tasks: [],
  workspaces: [
    {
      id: "ws-1",
      name: "Computer Science",
      type: "course",
      subjects: ["Algorithms", "Operating Systems", "Databases"],
    },
    {
      id: "ws-2",
      name: "Research Project",
      type: "project",
      subjects: ["Research", "Mathematics"],
    },
  ],
  energyLevel: 0,
  bandwidthCurve: baseCurve,
  adjustedBandwidthCurve: baseCurve,
  dayPhase: "active",
  currentDate: new Date(),
  burnoutRisk: "safe",
  confirmedSlots: [],
  reasoningChain: [
    {
      number: 1,
      text: "SCHEDULER INITIALIZED. 96x7 matrix loaded.",
      isRule: true,
    },
    { number: 2, text: "Awaiting task data synchronization." },
  ],
  highlightedTaskId: null,
  activeConflict: null,
  dailyReport: null,
  isAddTaskOpen: false,
  selectedView: "day",
  userProfile: null,
  scheduledDays: [],
  sectionWeights: { morning: 0.4, afternoon: 0.35, evening: 0.25 },
  schedulerLog: [],
  theme: "light",
  calibratedMultipliers: null,
  activeRoadmap: null,
  roadmapGenerating: false,
  isVoiceActive: false,
  voiceTranscript: "",
};

// ── Context ───────────────────────────────────────────────

const AppContext = createContext<{
  state: AppState;
  dispatch: React.Dispatch<Action>;
} | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within AppProvider");
  return ctx;
}
