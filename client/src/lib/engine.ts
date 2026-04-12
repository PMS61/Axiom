/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Scheduling Engine Utilities
   All computations are deterministic. No inference.
   ═══════════════════════════════════════════════════════════ */

import type {
  Task,
  TaskType,
  TaskPriority,
  BandwidthCurve,
  EnergyLevel,
  ReasoningStep,
  SlotCandidate,
  ScheduleConflict,
  ConflictResolution,
  DaySchedule,
} from "./types";

import type { CLBreakdown } from "./types";

// ── Slot / Time Helpers ───────────────────────────────────

/** Convert slot index (0-95) to time string "HH:MM" */
export function slotToTime(slot: number): string {
  const hours = Math.floor(slot / 4);
  const minutes = (slot % 4) * 15;
  return `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}`;
}

/** Convert time string "HH:MM" to slot index */
export function timeToSlot(time: string): number {
  const [h, m] = time.split(":").map(Number);
  return h * 4 + Math.floor(m / 15);
}

/** Duration in minutes to number of slots */
export function durationToSlots(minutes: number): number {
  return Math.ceil(minutes / 15);
}

/** Check if a slot is blocked by sleep or fixed commitments */
export function isSlotBlocked(slot: number, dayIdx: number, profile: any): boolean {
  if (!profile) return false;
  const slotMinutes = slot * 15;
  
  // 1. Sleep
  const { wakeTime, sleepTime } = profile;
  if (wakeTime !== null && sleepTime !== null) {
    if (sleepTime > wakeTime) {
      if (slotMinutes >= sleepTime || slotMinutes < wakeTime) return true;
    } else {
      if (slotMinutes >= sleepTime && slotMinutes < wakeTime) return true;
    }
  }
  
  // 2. Fixed Commitments
  for (const block of (profile.fixedCommitments || [])) {
    if (block.days.includes(dayIdx)) {
      if (slotMinutes >= block.start_min && slotMinutes < block.end_min) return true;
    }
  }
  
  // 3. Hard Exclusions
  for (const block of (profile.hardExclusions || [])) {
    if (block.days.includes(dayIdx)) {
      if (slotMinutes >= block.start_min && slotMinutes < block.end_min) return true;
    }
  }
  
  return false;
}

/** Format completionTime for display */
export function formatDuration(minutes: number): string {
  if (minutes < 60) return `${minutes}min`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m > 0 ? `${h}h ${m}min` : `${h}h`;
}

// ── Cognitive Load Calculator ─────────────────────────────

const TYPE_MULTIPLIERS: Record<TaskType, number> = {
  learning: 1.4,
  problem_solving: 1.3,
  writing: 1.1,
  revision: 0.9,
  reading: 0.8,
  administrative: 0.6,
  recreational: -1.0,
};

const PRIO_WEIGHTS: Record<TaskPriority, number> = {
  high: 1.5,
  normal: 1.0,
  low: 0.7,
};

/** Map Cognitive Load (Etask) total to a specific border class weighting for the UI. */
export function etaskToBorderClass(etask: number | undefined | null): string {
  const absoluteCl = Math.abs(etask || 0);
  const weight = Math.max(1, Math.min(10, Math.round(absoluteCl)));
  return `task-block-border-${weight}`;
}

export function computeCL(
  difficulty: number,
  durationMinutes: number,
  deadlineDays: number | null,
  taskType: TaskType,
  priority: TaskPriority,
): CLBreakdown {
  const baseDifficulty = difficulty;
  const durationWeight = Math.min(durationMinutes / 60, 2.0);
  const deadlineUrgency =
    deadlineDays !== null
      ? deadlineDays <= 1
        ? 1.8
        : deadlineDays <= 3
          ? 1.4
          : deadlineDays <= 7
            ? 1.1
            : 1.0
      : 1.0;
  const typeMultiplier = TYPE_MULTIPLIERS[taskType];
  const priorityWeight = PRIO_WEIGHTS[priority];

  const total = +(
    baseDifficulty *
    durationWeight *
    deadlineUrgency *
    Math.abs(typeMultiplier) *
    priorityWeight *
    (typeMultiplier < 0 ? -1 : 1)
  ).toFixed(1);

  return {
    baseDifficulty,
    durationWeight: +durationWeight.toFixed(2),
    deadlineUrgency: +deadlineUrgency.toFixed(2),
    typeMultiplier,
    priorityWeight,
    total,
  };
}

// ── Default Bandwidth Curve ───────────────────────────────
// Ultradian rhythm: peaks at ~10:00, ~15:00; troughs at ~13:00, ~20:00

export function generateDefaultBandwidthCurve(): BandwidthCurve {
  const curve: number[] = [];
  for (let slot = 0; slot < 96; slot++) {
    const hour = slot / 4;
    let bw: number;
    if (hour < 6) bw = 1.0;
    else if (hour < 8) bw = 3.0 + (hour - 6) * 2.0;
    else if (hour < 10) bw = 7.0 + (hour - 8) * 1.5;
    else if (hour < 11) bw = 10.0;
    else if (hour < 13) bw = 10.0 - (hour - 11) * 2.5;
    else if (hour < 14) bw = 5.0;
    else if (hour < 16) bw = 5.0 + (hour - 14) * 2.0;
    else if (hour < 17) bw = 9.0;
    else if (hour < 20) bw = 9.0 - (hour - 17) * 2.0;
    else if (hour < 22) bw = 3.0 - (hour - 20) * 1.0;
    else bw = 1.0;
    curve.push(+Math.max(bw, 0.5).toFixed(1));
  }
  return curve;
}

export function applyEnergyMultiplier(
  curve: BandwidthCurve,
  energy: EnergyLevel,
): BandwidthCurve {
  const multipliers: Record<EnergyLevel, number> = {
    [-2]: 0.5,
    [-1]: 0.7,
    [0]: 1.0,
    [1]: 1.2,
    [2]: 1.4,
  };
  const mul = multipliers[energy];
  return curve.map((v) => +(v * mul).toFixed(1));
}

// ── Slot Fitness Scoring ──────────────────────────────────

export function computeSlotFitness(
  slotIndex: number,
  taskCL: number,
  bandwidth: number,
  isProductiveHour: boolean,
  contextSwitchCount: number,
  dayIdx: number,
  hoursToDeadline: number | null,
): SlotCandidate {
  const productiveHourBonus = isProductiveHour ? 1.2 : 0;
  const contextSwitchPenalty = contextSwitchCount >= 3 ? -2.0 : contextSwitchCount >= 2 ? -0.8 : 0;
  const deadlineProximityBonus =
    hoursToDeadline !== null
      ? hoursToDeadline <= 6
        ? 2.0
        : hoursToDeadline <= 24
          ? 1.0
          : 0
      : 0;

  const fitnessScore = +(
    bandwidth -
    taskCL +
    productiveHourBonus +
    contextSwitchPenalty +
    deadlineProximityBonus
  ).toFixed(1);

  return {
    slotIndex,
    startSlot: slotIndex,
    dayIdx,
    fitnessScore,
    bandwidthRemaining: +(bandwidth - taskCL).toFixed(1),
    productiveHourBonus,
    contextSwitchPenalty,
    deadlineProximityBonus,
    isWinner: false,
  };
}

// ── Sacrifice Impact Calculator ───────────────────────────

export function computeSacrificeImpact(
  originalCL: number,
  sacrificeFactor: number = 0.7,
): { reducedCL: number; impactPercent: number } {
  const reducedCL = +(originalCL * sacrificeFactor).toFixed(1);
  const impactPercent = +((1 - sacrificeFactor) * 100).toFixed(0);
  return { reducedCL, impactPercent };
}

// ── Etask Border Weight ──────────────────────────────────────

/** Determine burnout risk based on last 72h + next 48h of planned Etask */
export function calculateBurnoutRisk(tasks: Task[], scheduledDays: DaySchedule[] = []): "safe" | "watch" | "warning" | "critical" {
  // 1. Axiom-based calculation (First source of truth)
  if (scheduledDays.length > 0) {
    const next3Days = scheduledDays.slice(0, 3);
    const maxAxioms = Math.max(...next3Days.map(d => d.totalAxiomsUsed), 0);
    const avgAxioms = next3Days.reduce((sum, d) => sum + d.totalAxiomsUsed, 0) / next3Days.length;

    if (maxAxioms > 45 || avgAxioms > 40) return "critical";
    if (maxAxioms > 35 || avgAxioms > 30) return "warning";
    if (maxAxioms > 25) return "watch";
    return "safe";
  }

  // 2. Legacy/Fallback based on Task structures (e.g. before first scheduler run)
  const scheduled = tasks.filter(t => t.scheduledSlot && (t.state === "scheduled" || t.state === "completed"));
  if (scheduled.length === 0) return "safe";

  const dailyEtask: Record<number, number> = {};
  for (const t of scheduled) {
    const d = t.scheduledSlot!.day;
    if (d >= 0 && d <= 2) {
      dailyEtask[d] = (dailyEtask[d] || 0) + Math.abs(t.etask);
    }
  }

  const values = Object.values(dailyEtask);
  const maxCL = Math.max(...values, 0);
  const avgCL = values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0;

  if (maxCL > 45 || avgCL > 40) return "critical";
  if (maxCL > 35 || avgCL > 30) return "warning";
  if (maxCL > 25) return "watch";
  return "safe";
}

// ── Core Scheduling Algorithm ──────────────────────────────

export function runSchedulingAlgorithm(
  tasks: Task[],
  bandwidthCurve: BandwidthCurve,
  userProfile: any,
  existingReasoningChain: ReasoningStep[]
): { tasks: Task[]; conflict: ScheduleConflict | null; reasoningChain: ReasoningStep[] } {
  const unscheduled = tasks.filter((t) => t.state === "unscheduled");
  const scheduled = tasks.filter((t) => t.state !== "unscheduled");
  const bw = bandwidthCurve;

  const occupied = new Map<string, string>(); // slot key -> subject
  for (const t of scheduled) {
    if (t.scheduledSlot) {
      for (let s = t.scheduledSlot.startSlot; s < t.scheduledSlot.endSlot; s++) {
        occupied.set(`${t.scheduledSlot.day}_${s}`, t.subject || "none");
      }
    }
  }

  const sorted = [...unscheduled].sort((a, b) => {
    const urgA = a.deadline ? (new Date(a.deadline).getTime() - Date.now()) / 3600000 : 999;
    const urgB = b.deadline ? (new Date(b.deadline).getTime() - Date.now()) / 3600000 : 999;
    if (urgA !== urgB) return urgA - urgB;
    const priOrder = { high: 0, normal: 1, low: 2 };
    if (priOrder[a.priority] !== priOrder[b.priority])
      return priOrder[a.priority] - priOrder[b.priority];
    
    // Syllabus-First: use original document sequence as final tie-breaker
    if (a.sequence !== undefined && b.sequence !== undefined && a.sequence !== b.sequence) {
      return a.sequence - b.sequence;
    }
    
    return Math.abs(b.etask) - Math.abs(a.etask);
  });

  const reasoningChain: ReasoningStep[] = [
    ...existingReasoningChain,
    {
      number: existingReasoningChain.length + 1,
      text: `SCHEDULER RUN: ${sorted.length} tasks in pool. Scanning 7-day window.`,
      isRule: true,
    },
  ];

  const today = new Date();

  const placedTasks: Task[] = [...scheduled];
  let conflict: ScheduleConflict | null = null;

  for (const task of sorted) {
    const slotsNeeded = durationToSlots(task.completionTime);
    const candidates: any[] = [];

    // Scan days 0-6 relative to today
    for (let dIdx = 0; dIdx < 7; dIdx++) {
      const d = new Date(today);
      d.setDate(today.getDate() + dIdx);
      const actualDayOfWeek = d.getDay();

      // Scan available windows (6:00 - 22:00 = slots 24-88)
      for (let start = 24; start <= 88 - slotsNeeded; start++) {
        let slotFree = true;
        let minBW = Infinity;
        let energyBonus = 0;

        let contextSwitchCount = 0;
        for (let s = start; s < start + slotsNeeded; s++) {
          const key = `${dIdx}_${s}`;
          if (occupied.has(key) || isSlotBlocked(s, actualDayOfWeek, userProfile)) {
            slotFree = false;
            break;
          }
          minBW = Math.min(minBW, bw[s]);
          if (userProfile) {
            const slotMins = s * 15;
            const isPeak = userProfile.peakFocusWindows?.some((w: any) => slotMins >= w.start_min && slotMins < w.end_min);
            if (isPeak) energyBonus += 0.5;
          }
        }

        // Check for subject transitions in proximity (within 2 slots of start)
        const prevSlotSubject = occupied.get(`${dIdx}_${start - 1}`);
        if (prevSlotSubject && task.subject && prevSlotSubject !== task.subject) {
          contextSwitchCount = 1;
        }

        if (!slotFree) continue;

        const hoursToDeadline = task.deadline ? (new Date(task.deadline).getTime() - Date.now()) / 3600000 : null;
        const candidate = computeSlotFitness(start, Math.abs(task.etask), minBW, energyBonus > 0, contextSwitchCount, dIdx, hoursToDeadline);
        candidate.fitnessScore += energyBonus;
        candidate.fitnessScore -= dIdx * 2.0; // Preference for earlier days
        candidate.startSlot = start;
        candidate.dayIdx = dIdx;
        candidates.push(candidate);
      }
    }

    if (candidates.length === 0) {
      conflict = {
        taskId: task.id,
        reason: `No available slot for "${task.name}" (Etask=${task.etask}, completionTime=${task.completionTime}min)`,
        availableResolutions: task.etask > 5 ? ["defer", "sacrifice"] : ["defer", "extend_deadline"],
        reasoningSteps: [
          {
            number: 1,
            text: `CONFLICT: No valid slot for "${task.name}" before deadline.`,
            isConflict: true,
          },
        ],
      };
      reasoningChain.push({
        number: reasoningChain.length + 1,
        text: `[!] CONFLICT: Cannot place "${task.name}". No available slot found.`,
        isConflict: true,
      });
      placedTasks.push(task);
      continue;
    }

    candidates.sort((a, b) => b.fitnessScore - a.fitnessScore);
    const winner = candidates[0];

    const placedTask: Task = {
      ...task,
      state: "scheduled",
      scheduledSlot: {
        startSlot: winner.startSlot,
        endSlot: winner.startSlot + slotsNeeded,
        day: winner.dayIdx,
        fitnessScore: winner.fitnessScore,
        reasoningSteps: generatePlacementReasoning(task, candidates, winner.dayIdx),
      },
    };

    placedTasks.push(placedTask);
    for (let s = winner.startSlot; s < winner.startSlot + slotsNeeded; s++) {
      occupied.set(`${winner.dayIdx}_${s}`, task.subject || "none");
    }

    reasoningChain.push({
      number: reasoningChain.length + 1,
      text: `SUCCESS: Placed "${task.name}" at day ${winner.dayIdx}, ${slotToTime(winner.startSlot)} (Fitness: ${winner.fitnessScore.toFixed(1)})`,
    });
  }

  return { tasks: placedTasks, conflict, reasoningChain };
}

// ── Date Formatting ───────────────────────────────────────

export function formatDateHeading(date: Date): string {
  return date.toLocaleDateString("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
  });
}

export function formatShortDate(date: Date): string {
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

// ── Generate Reasoning Steps ──────────────────────────────

export function generatePlacementReasoning(
  task: Task,
  candidates: SlotCandidate[],
  winnerIndex: number,
): ReasoningStep[] {
  const winner = candidates[winnerIndex];
  const steps: ReasoningStep[] = [
    {
      number: 1,
      text: `Etask-demand("${task.name}") = ${task.etask}`,
      relatedTaskId: task.id,
    },
  ];

  // Show top candidates
  const sorted = [...candidates].sort((a, b) => b.fitnessScore - a.fitnessScore);
  const topCandidates = sorted.slice(0, 4);

  topCandidates.forEach((c) => {
    const isWinner = c.slotIndex === winner.slotIndex;
    
    // Add evaluation details as separate rules/steps to simulate branching
    if (c.productiveHourBonus > 0 && isWinner) {
      steps.push({
        number: steps.length + 1,
        text: `RULE: Productive hour bonus +${c.productiveHourBonus} applied to ${slotToTime(c.slotIndex)}`,
        isRule: true,
        relatedTaskId: task.id,
      });
    }

    if (c.contextSwitchPenalty < 0 && isWinner) {
      steps.push({
        number: steps.length + 1,
        text: `RULE: Context switch penalty ${c.contextSwitchPenalty} — excessive subject transitions in window`,
        isRule: true,
        relatedTaskId: task.id,
      });
    }

    if (!isWinner) {
      steps.push({
        number: steps.length + 1,
        text: `EVALUATE: slot ${slotToTime(c.slotIndex)} -> REJECTED (fitness=${c.fitnessScore}, lower than winner)`,
        isConflict: true,
        relatedTaskId: task.id,
      });
    } else {
       steps.push({
        number: steps.length + 1,
        text: `EVALUATE: slot ${slotToTime(c.slotIndex)} -> ACCEPTED (fitness=${c.fitnessScore}, highest available)`,
        isAction: false,
        relatedTaskId: task.id,
      });   
    }
  });

  steps.push({
    number: steps.length + 1,
    text: `ACTION: Schedule "${task.name}" at ${slotToTime(winner.slotIndex)}`,
    isAction: true,
    relatedTaskId: task.id,
  });

  return steps;
}

// ── Analytics ─────────────────────────────────────────────

export interface SchedulerAnalytics {
  avgSessionDurationPct: number;
  recoveryAdherencePct: number;
  contextSwitchesPerDay: number;
  deadlineBufferDays: number;
}

export function calculateAnalytics(tasks: Task[]): SchedulerAnalytics {
  const completed = tasks.filter(t => t.state === "completed");
  const scheduled = tasks.filter(t => t.state === "scheduled" || t.state === "completed");
  const recreational = scheduled.filter(t => t.type === "recreational");
  const recreationalCompleted = completed.filter(t => t.type === "recreational");

  const avgSessionDurationPct = scheduled.length > 0 ? (completed.length / scheduled.length) * 100 : 100;
  const recoveryAdherencePct = recreational.length > 0 ? (recreationalCompleted.length / recreational.length) * 100 : 100;

  // Track context switches in the scheduled set
  let totalSwitches = 0;
  const daySegments = new Map<number, Task[]>();
  for (const t of scheduled) {
    if (!t.scheduledSlot) continue;
    const d = t.scheduledSlot.day;
    if (!daySegments.has(d)) daySegments.set(d, []);
    daySegments.get(d)!.push(t);
  }

  for (const [day, dayTasks] of daySegments.entries()) {
    dayTasks.sort((a,b) => a.scheduledSlot!.startSlot - b.scheduledSlot!.startSlot);
    for (let i = 1; i < dayTasks.length; i++) {
      if (dayTasks[i].subject !== dayTasks[i-1].subject) {
        totalSwitches++;
      }
    }
  }

  const daysRepresented = daySegments.size || 1;
  const switchesPerDay = totalSwitches / daysRepresented;

  // Deadline Buffer: how close to deadlines are we completing?
  // Average days early(+) or late(-)
  let totalDelta = 0;
  let count = 0;
  for (const t of completed) {
    if (t.deadline) {
      const deadline = new Date(t.deadline).getTime();
      const completionTime = Date.now(); // assume completed now for simplicity
      const deltaDays = (deadline - completionTime) / (1000 * 3600 * 24);
      totalDelta += deltaDays;
      count++;
    }
  }

  return {
    avgSessionDurationPct: Math.round(avgSessionDurationPct),
    recoveryAdherencePct: Math.round(recoveryAdherencePct),
    contextSwitchesPerDay: +switchesPerDay.toFixed(1),
    deadlineBufferDays: count > 0 ? +(totalDelta / count).toFixed(1) : 0
  };
}

interface TypeCompletionStats {
  type: string;
  scheduled: number;
  completed: number;
  completionRate: number;
}

export function computeCalibratedMultipliers(tasks: Task[]): {
  learning: number;
  problem_solving: number;
  writing: number;
  revision: number;
  reading: number;
  administrative: number;
} | null {
  const completed = tasks.filter(t => t.state === "completed");
  const scheduledOrCompleted = tasks.filter(t => t.state === "scheduled" || t.state === "completed");

  if (scheduledOrCompleted.length === 0) return null;

  const typeStats: TypeCompletionStats[] = [];
  const types = ["learning", "problem_solving", "writing", "revision", "reading", "administrative"];

  for (const taskType of types) {
    const scheduled = scheduledOrCompleted.filter(t => t.type === taskType);
    const completedCount = completed.filter(t => t.type === taskType).length;
    
    typeStats.push({
      type: taskType,
      scheduled: scheduled.length,
      completed: completedCount,
      completionRate: scheduled.length > 0 ? completedCount / scheduled.length : 1.0,
    });
  }

  const defaults = {
    learning: 1.4,
    problem_solving: 1.3,
    writing: 1.1,
    revision: 0.9,
    reading: 0.8,
    administrative: 0.6,
  };

  const calibrated: Record<string, number> = {};
  for (const stat of typeStats) {
    const defaultMult = defaults[stat.type as keyof typeof defaults] ?? 1.0;
    let adjusted = defaultMult;

    if (stat.scheduled >= 3) {
      if (stat.completionRate >= 0.8) {
        adjusted = defaultMult * 0.9;
      } else if (stat.completionRate < 0.5) {
        adjusted = defaultMult * 1.2;
      }
    }

    calibrated[stat.type] = +(Math.max(0.5, Math.min(1.5, adjusted))).toFixed(2);
  }

  return calibrated as {
    learning: number;
    problem_solving: number;
    writing: number;
    revision: number;
    reading: number;
    administrative: number;
  };
}
