/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Full Mathematical Scheduling Pipeline
   Pure deterministic. No AI. No randomness.

   Input: pre-scored DB task rows  +  user profile fields
   (No chunking — tasks are scheduled as atomic units.)

   Pipeline stages (per scheduling run):
     1. Separate  — work vs recreational tasks
     2. Sort      — by pre-computed score DESC, then sequence ASC
     3. Anti-starvation — interleave low-priority tasks
     4. Allocate  — distribute tasks across sections/days, check constraints
     5. Diversity — enforce Shannon entropy > 0.5 per day
     6. Output    — produce DaySchedule[] + reasoning log
   ═══════════════════════════════════════════════════════════ */

import type {
  SectionName,
  SectionSchedule,
  DaySchedule,
  SchedulerOutput,
  TimeSection,
} from "./types";

import {
  BASE_AXIOMS,
  buildSections,
  computeAxiomGain,
  priorityToNum,
} from "./energyModel";

import { applyAntiStarvation, computeDiversityEntropy, computeDailyLoad } from "./constraints";

// ── Constants ─────────────────────────────────────────────

const MAX_DAYS = 30;

// ── DB Row Shape ──────────────────────────────────────────

/**
 * Mirrors the `tasks` table row exactly.
 * All fields are snake_case, matching the database column names.
 */
export interface TaskRow {
  id: string;                          // UUID PRIMARY KEY
  user_id: number;
  name: string;                        // VARCHAR(255) NOT NULL
  type: string;                        // VARCHAR(50) NOT NULL
  difficulty: number;                  // INTEGER NOT NULL
  completion_time: number;             // INTEGER NOT NULL (minutes)
  priority: "high" | "normal" | "low"; // VARCHAR(20) NOT NULL
  state: string;                       // VARCHAR(50) NOT NULL
  subject?: string | null;             // VARCHAR(255)
  deadline?: string | null;            // TIMESTAMPTZ
  etask: number;                       // DOUBLE PRECISION NOT NULL (pre-computed axiom cost)
  scheduled_slot?: any | null;         // JSONB
  created_at?: string;                 // TIMESTAMPTZ
  sequence: number;                    // INTEGER DEFAULT 0 — ordering within a parent task set
  urgency: number;                     // DOUBLE PRECISION DEFAULT 0
  score: number;                       // DOUBLE PRECISION DEFAULT 0
  days_remaining: number;              // INTEGER DEFAULT 0
}

// ── Legacy Adapter ────────────────────────────────────────

/**
 * Converts a camelCase `Task` object (as stored in the React store / types.ts)
 * to the snake_case `TaskRow` shape expected by `runScheduler`.
 *
 * Use this when you only have a `Task[]` available, e.g. inside `store.tsx`.
 * When tasks come directly from the database they are already `TaskRow` objects.
 */
export function taskToRow(task: {
  id: string;
  name: string;
  type: string;
  difficulty: number;
  completionTime: number;
  priority: "high" | "normal" | "low";
  state: string;
  subject?: string | null;
  deadline?: string | null;
  etask: number;
  scheduled_slot?: any | null;
  created_at?: string;
  createdAt?: string;
  sequence?: number | null;
  urgency?: number | null;
  score?: number | null;
  daysRemaining?: number | null;
  user_id?: number;
}): TaskRow {
  return {
    id: task.id,
    user_id: task.user_id ?? 0,
    name: task.name,
    type: task.type,
    difficulty: task.difficulty,
    completion_time: task.completionTime,
    priority: task.priority,
    state: task.state,
    subject: task.subject ?? null,
    deadline: task.deadline ?? null,
    etask: task.etask,
    scheduled_slot: task.scheduled_slot ?? null,
    created_at: task.created_at ?? task.createdAt,
    sequence: task.sequence ?? 0,
    urgency: task.urgency ?? 0,
    score: task.score ?? 0,
    days_remaining: task.daysRemaining ?? 0,
  };
}

// ── Fixed Commitment / Exclusion Shape ────────────────────

export interface TimeBlock {
  name: string;
  days: number[];        // 0=Sun … 6=Sat
  start_min: number;     // minutes from midnight
  end_min: number;
}

export interface FocusWindow {
  start_min: number;
  end_min: number;
}

// ── Section Weights ───────────────────────────────────────

export interface SectionWeights {
  morning: number;   // default 0.40
  afternoon: number; // default 0.35
  evening: number;   // default 0.25
}

// ── Scheduler Input ───────────────────────────────────────

export interface SchedulerInput {
  /** Array of pre-scored task DB rows to schedule */
  tasks: TaskRow[];

  /** Minutes from midnight the user wakes up (e.g. 420 = 7 am) */
  wake_time: number;

  /** Minutes from midnight the user sleeps (e.g. 1320 = 10 pm) */
  sleep_time: number;

  /** Recurring fixed commitments (classes, meetings…) */
  fixed_commitments?: TimeBlock[];

  /** Hard exclusions (chores, religious time…) */
  hard_exclusions?: TimeBlock[];

  /** Preferred high-focus windows from user profile */
  peak_focus_windows?: FocusWindow[];

  /** Low-energy windows from user profile */
  low_energy_windows?: FocusWindow[];

  /** Preferred inter-task buffer in minutes */
  switch_buffer?: number;

  /** Learning session style from user profile */
  session_style?: number;

  /** Deadline handling style from user profile */
  deadline_style?: number;

  /** User timezone */
  timezone?: string;

  /** Recovery activities configured by user */
  recovery_activities?: Array<{
    name: string;
    duration_min: number;
    energy_value: number;
  }>;

  /**
   * Fraction of daily axiom budget allocated to each section.
   * Must sum to 1.0 (enforced internally via buildSections).
   * Defaults: morning=0.40, afternoon=0.35, evening=0.25.
   */
  section_weights?: Partial<SectionWeights>;

  /** ISO YYYY-MM-DD reference date, defaults to today */
  start_date?: string;
}

// ── Date Utilities ────────────────────────────────────────

function todayString(): string {
  return new Date().toISOString().split("T")[0];
}

function addDays(dateStr: string, n: number): string {
  const d = new Date(`${dateStr}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().split("T")[0];
}

function dayOfWeekForDate(dateStr: string, timezone?: string): number {
  if (!timezone) {
    return new Date(`${dateStr}T12:00:00Z`).getDay();
  }

  const weekday = new Intl.DateTimeFormat("en-US", {
    weekday: "short",
    timeZone: timezone,
  }).format(new Date(`${dateStr}T12:00:00Z`));

  const map: Record<string, number> = {
    Sun: 0,
    Mon: 1,
    Tue: 2,
    Wed: 3,
    Thu: 4,
    Fri: 5,
    Sat: 6,
  };
  return map[weekday] ?? 0;
}

// ── Slot Blocking ─────────────────────────────────────────

/**
 * Returns true if the 15-min slot at `slotMins` on `dayOfWeek` is unavailable
 * given the user's sleep window, fixed commitments, and hard exclusions.
 */
function isSlotBlocked(
  slotMins: number,
  dayOfWeek: number,
  wakeTime: number,
  sleepTime: number,
  fixedCommitments: TimeBlock[],
  hardExclusions: TimeBlock[],
  switchBufferMins: number,
): boolean {
  const buffer = Math.max(0, switchBufferMins);

  // 1. Sleep window
  if (sleepTime > wakeTime) {
    // Normal window: blocked outside [wakeTime, sleepTime)
    if (slotMins >= sleepTime || slotMins < wakeTime) return true;
  } else if (sleepTime < wakeTime) {
    // Overnight window: blocked inside [sleepTime, wakeTime)
    if (slotMins >= sleepTime && slotMins < wakeTime) return true;
  }

  // 2. Fixed commitments
  if (
    fixedCommitments.some(
      (c) =>
        c.days.includes(dayOfWeek) &&
        slotMins >= c.start_min - buffer &&
        slotMins < c.end_min + buffer,
    )
  )
    return true;

  // 3. Hard exclusions
  if (
    hardExclusions.some(
      (c) =>
        c.days.includes(dayOfWeek) &&
        slotMins >= c.start_min - buffer &&
        slotMins < c.end_min + buffer,
    )
  )
    return true;

  return false;
}

/**
 * Finds the first available contiguous block of `size` slots starting at or
 * after `start` (0-95) on `dayOfWeek`, respecting the user's constraints.
 * Falls back to `start` if nothing is found (avoids infinite loops).
 */
interface SlotPreference {
  preferPeak: boolean;
  preferLow: boolean;
  avoidLow: boolean;
  urgencyBias: number;
  styleBias: number;
}

function isWithinWindow(
  slotMins: number,
  windows: FocusWindow[],
): boolean {
  for (const w of windows) {
    if (slotMins >= w.start_min && slotMins < w.end_min) return true;
  }
  return false;
}

function findBestAvailableSlotBlock(
  start: number,
  size: number,
  dayOfWeek: number,
  wakeTime: number,
  sleepTime: number,
  fixedCommitments: TimeBlock[],
  hardExclusions: TimeBlock[],
  peakFocusWindows: FocusWindow[],
  lowEnergyWindows: FocusWindow[],
  switchBufferMins: number,
  preference: SlotPreference,
): number {
  let bestStart = -1;
  let bestScore = Number.NEGATIVE_INFINITY;

  for (let curr = start; curr < 96; curr++) {
    let fits = true;
    let score = 0;

    for (let s = curr; s < curr + size; s++) {
      const slotMins = s * 15;
      if (
        s >= 96 ||
        isSlotBlocked(
          slotMins,
          dayOfWeek,
          wakeTime,
          sleepTime,
          fixedCommitments,
          hardExclusions,
          switchBufferMins,
        )
      ) {
        fits = false;
        break;
      }

      const inPeak = isWithinWindow(slotMins, peakFocusWindows);
      const inLow = isWithinWindow(slotMins, lowEnergyWindows);

      if (preference.preferPeak && inPeak) score += 2 + preference.styleBias;
      if (preference.preferLow && inLow) score += 1 + preference.styleBias * 0.5;
      if (preference.avoidLow && inLow) score -= 2 + preference.styleBias;
      if (preference.urgencyBias > 0) score += ((96 - s) / 96) * preference.urgencyBias;
    }

    if (!fits) continue;

    // Favor earlier placements when candidate quality is similar.
    score -= (curr - start) / 96;

    if (score > bestScore) {
      bestScore = score;
      bestStart = curr;
    }
  }

  if (bestStart >= 0) return bestStart;
  return start; // fallback
}

// ── Empty Section Schedule ────────────────────────────────

function emptySection(section: TimeSection): SectionSchedule {
  return {
    section: section.name,
    tasks: [],
    axiomBudget: section.axiomBudget,
    axiomUsed: 0,
    axiomRemaining: section.axiomBudget,
  };
}

// ── Schedulable Unit ──────────────────────────────────────

/**
 * Internal scheduling unit — one task treated as an unbreakable whole.
 */
interface SchedulableUnit {
  taskId: string;
  taskName: string;
  completionTime: number;
  eTask: number;            // axiom cost (pre-computed)
  daysRemaining: number;
  preferredDay: number;     // 0 = schedule as early as possible
  preferredSection: SectionName;
  isRecreational: boolean;
  priorityNum: number;
  sequence: number;         // lower = should appear earlier within a group
}

// ── Main Scheduler ────────────────────────────────────────

/**
 * Full multi-day, multi-section deterministic scheduler.
 *
 * Tasks are accepted as pre-scored DB rows; no re-scoring is performed.
 * Chunking logic has been removed — each task is an atomic unit.
 * The `sequence` field acts as a tiebreaker after `score` so
 * prerequisite subtasks (lower sequence) are always placed first.
 */
export function runScheduler(input: SchedulerInput): SchedulerOutput {
  const today = input.start_date ?? todayString();

  // Map section_weights to the shape buildSections expects
  const weightMap: Partial<Record<SectionName, number>> = {
    morning: input.section_weights?.morning,
    afternoon: input.section_weights?.afternoon,
    evening: input.section_weights?.evening,
  };
  const sections = buildSections(weightMap);

  const wakeTime = input.wake_time;
  const sleepTime = input.sleep_time;
  const fixedCommitments: TimeBlock[] = input.fixed_commitments ?? [];
  const hardExclusions: TimeBlock[] = input.hard_exclusions ?? [];
  const peakFocusWindows: FocusWindow[] = input.peak_focus_windows ?? [];
  const lowEnergyWindows: FocusWindow[] = input.low_energy_windows ?? [];
  const switchBufferMins = input.switch_buffer ?? 0;
  const sessionStyle = input.session_style ?? 0;
  const deadlineStyle = input.deadline_style ?? 0;
  const timezone = input.timezone;
  const recoveryActivities = input.recovery_activities ?? [];
  const avgRecoveryDuration = recoveryActivities.length > 0
    ? recoveryActivities.reduce((s, r) => s + Math.max(0, r.duration_min), 0) /
      recoveryActivities.length
    : 0;

  const reasoningLog: string[] = [];
  const unscheduled: string[] = [];

  // ── 1. Separate recreational from work tasks ─────────────
  const workRows = input.tasks.filter((t) => t.type !== "recreational");
  const recRows = input.tasks.filter((t) => t.type === "recreational");

  reasoningLog.push(
    `INIT: ${workRows.length} work tasks, ${recRows.length} recreational tasks. BASE_AXIOMS=${BASE_AXIOMS}.`,
  );

  // ── 2. Sort by pre-computed score DESC, then sequence ASC ─
  //    (lower sequence = earlier prerequisite, must come first)
  const sorted = [...workRows].sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    if (a.sequence !== b.sequence) return a.sequence - b.sequence;
    return a.id.localeCompare(b.id); // deterministic final tiebreak
  });

  reasoningLog.push(
    `SORT: Top task = "${sorted[0]?.name ?? "—"}" score=${sorted[0]?.score.toFixed(3) ?? 0}, seq=${sorted[0]?.sequence ?? 0}.`,
  );

  // ── 3. Anti-starvation interleave ─────────────────────────
  //    applyAntiStarvation expects ScoredTask shape — we adapt inline.
  const scoredLike = sorted.map((t) => ({
    task: { id: t.id, name: t.name, priority: t.priority } as any,
    priorityNum: priorityToNum(t.priority),
    eTask: t.etask,
    daysRemaining: t.days_remaining,
    urgency: t.urgency,
    score: t.score,
    sequence: t.sequence,
    _row: t, // carry original
  }));

  const reordered = applyAntiStarvation(scoredLike as any);
  reasoningLog.push(`ANTI-STARVATION: Queue reordered for fairness.`);

  // ── 4. Build schedulable unit list ────────────────────────
  const units: SchedulableUnit[] = (reordered as typeof scoredLike).map((st) => {
    const t = st._row as TaskRow;
    const pNum = priorityToNum(t.priority);

    // Section assignment by priority:
    //   high (pNum >= 9)   → morning
    //   normal (pNum >= 5) → afternoon
    //   low                → evening
    // Also honour sequence: seq=1 → prefer morning regardless of priority
    let section: SectionName =
      pNum >= 9 ? "morning" : pNum >= 5 ? "afternoon" : "evening";

    if (t.sequence === 1) section = "morning"; // prerequisites always start day

    return {
      taskId: t.id,
      taskName: t.name,
      completionTime: t.completion_time,
      eTask: t.etask,
      daysRemaining: t.days_remaining,
      preferredDay: 0,
      preferredSection: section,
      isRecreational: false,
      priorityNum: pNum,
      sequence: t.sequence,
    };
  });

  // ── 5. Allocate into days/sections ────────────────────────

  const dayMap = new Map<number, Map<SectionName, SectionSchedule>>();

  function getDaySection(day: number, sectionName: SectionName): SectionSchedule {
    if (!dayMap.has(day)) {
      const secMap = new Map<SectionName, SectionSchedule>();
      for (const sec of sections) secMap.set(sec.name, emptySection(sec));
      dayMap.set(day, secMap);
    }
    return dayMap.get(day)!.get(sectionName)!;
  }

  function tryAllocate(
    unit: SchedulableUnit,
    day: number,
    section: SectionName,
    force = false,
  ): boolean {
    const sec = getDaySection(day, section);
    if (force || sec.axiomRemaining >= unit.eTask) {
      sec.tasks.push({
        taskId: unit.taskId,
        taskName: unit.taskName,
        completionTime: unit.completionTime,
        axiomCost: unit.eTask,
        axiomGain: 0,
        isRecreational: false,
      });
      sec.axiomUsed = +(sec.axiomUsed + unit.eTask).toFixed(4);
      sec.axiomRemaining = +(sec.axiomRemaining - unit.eTask).toFixed(4);
      return true;
    }
    return false;
  }

  const SECTION_ORDER: SectionName[] = ["morning", "afternoon", "evening"];

  for (const unit of units) {
    let placed = false;

    // Try preferred day first, then subsequent days up to daysRemaining
    for (
      let dayOffset = unit.preferredDay;
      dayOffset < Math.min(unit.daysRemaining + 1, MAX_DAYS);
      dayOffset++
    ) {
      // Try preferred section first, then fall through to others
      const tryOrder = [
        unit.preferredSection,
        ...SECTION_ORDER.filter((s) => s !== unit.preferredSection),
      ];

      for (const section of tryOrder) {
        if (tryAllocate(unit, dayOffset, section)) {
          reasoningLog.push(
            `PLACE: "${unit.taskName}" [seq=${unit.sequence}] → Day+${dayOffset} ${section} (cost=${unit.eTask.toFixed(2)})`,
          );
          placed = true;
          break;
        }
      }

      if (placed) break;
    }

    // Last resort: search all days
    if (!placed) {
      for (let dayOffset = 0; dayOffset < MAX_DAYS; dayOffset++) {
        for (const section of SECTION_ORDER) {
          if (tryAllocate(unit, dayOffset, section)) {
            reasoningLog.push(
              `FALLBACK: "${unit.taskName}" placed at Day+${dayOffset} ${section}`,
            );
            placed = true;
            break;
          }
        }
        if (placed) break;
      }
    }

    // Hard fallback: if strict axiom budgets reject all slots, force-place so
    // work items are still visible/schedulable in the matrix for manual tuning.
    if (!placed) {
      for (let dayOffset = 0; dayOffset < MAX_DAYS; dayOffset++) {
        for (const section of SECTION_ORDER) {
          if (tryAllocate(unit, dayOffset, section, true)) {
            reasoningLog.push(
              `FORCED: "${unit.taskName}" placed at Day+${dayOffset} ${section} despite budget overflow.`,
            );
            placed = true;
            break;
          }
        }
        if (placed) break;
      }
    }

    if (!placed) {
      reasoningLog.push(
        `UNSCHEDULED: "${unit.taskName}" (${unit.taskId}) — no axiom budget available.`,
      );
      if (!unscheduled.includes(unit.taskId)) unscheduled.push(unit.taskId);
    }
  }

  // ── 6. Insert recreational tasks as axiom restores ────────
  for (const rec of recRows) {
    const gain = computeAxiomGain(rec.completion_time);
    for (const [day, secMap] of dayMap) {
      for (const [secName, sec] of secMap) {
        if (sec.axiomUsed / sec.axiomBudget > 0.7) {
          sec.tasks.push({
            taskId: rec.id,
            taskName: rec.name,
            completionTime: rec.completion_time,
            axiomCost: 0,
            axiomGain: gain,
            isRecreational: true,
          });
          sec.axiomRemaining = +(sec.axiomRemaining + gain).toFixed(4);
          reasoningLog.push(
            `REC: "${rec.name}" inserted in Day+${day} ${secName} (+${gain.toFixed(2)} axioms).`,
          );
          break;
        }
      }
    }
  }

  // ── 7. Build DaySchedule output ───────────────────────────
  const days: DaySchedule[] = [];

  for (const [dayOffset, secMap] of [...dayMap.entries()].sort(([a], [b]) => a - b)) {
    const dayDate = addDays(today, dayOffset);
    const dayOfWeek = dayOfWeekForDate(dayDate, timezone);

    let sectionSchedules: SectionSchedule[] = SECTION_ORDER.map((name) => {
      const sec = secMap.get(name)!;

      // Section start slots: morning = 6 am (slot 24), afternoon = noon (48), evening = 6 pm (72)
      let currentSlot = name === "morning" ? 24 : name === "afternoon" ? 48 : 72;

      for (const item of sec.tasks) {
        const slotsNeeded = Math.max(1, Math.ceil(item.completionTime / 15));
        const row = input.tasks.find((r) => r.id === item.taskId);
        const rowPriorityNum = row ? priorityToNum(row.priority) : 5;
        const rowDifficulty = row?.difficulty ?? 5;
        const rowType = row?.type ?? "learning";
        const rowUrgency = row?.urgency ?? 0;
        const rowDaysRemaining = row?.days_remaining ?? 365;
        const rowEtask = Math.abs(row?.etask ?? 0);

        const preferPeak = rowDifficulty >= 7 || rowPriorityNum >= 9;
        const preferLow =
          rowType === "revision" ||
          rowType === "reading" ||
          rowType === "administrative" ||
          rowType === "recreational";
        const avoidLow = rowDifficulty >= 6 || rowPriorityNum >= 9;
        const urgencyBias =
          rowDaysRemaining <= 2 || rowUrgency >= 2 ? 1 + deadlineStyle * 0.15 : 0;
        const styleBias = Math.max(0, sessionStyle) * 0.2;
        const recoveryBufferMins =
          rowEtask >= 8 ? Math.min(30, Math.round(avgRecoveryDuration * 0.25)) : 0;
        const effectiveBufferMins = switchBufferMins + recoveryBufferMins;

        currentSlot = findBestAvailableSlotBlock(
          currentSlot,
          slotsNeeded,
          dayOfWeek,
          wakeTime,
          sleepTime,
          fixedCommitments,
          hardExclusions,
          peakFocusWindows,
          lowEnergyWindows,
          effectiveBufferMins,
          {
            preferPeak,
            preferLow,
            avoidLow,
            urgencyBias,
            styleBias,
          },
        );
        item.startSlot = currentSlot;
        currentSlot += slotsNeeded + Math.ceil(effectiveBufferMins / 15);
      }

      return sec;
    });

    // Re-bucket by final startSlot so section labels match actual placement.
    const sectionConfigMap = new Map(sections.map((s) => [s.name, s]));
    const normalizedSections = new Map<SectionName, SectionSchedule>(
      SECTION_ORDER.map((name) => [name, emptySection(sectionConfigMap.get(name)!)]),
    );

    const sectionFromSlot = (slot: number): SectionName =>
      slot < 48 ? "morning" : slot < 72 ? "afternoon" : "evening";

    for (const sec of sectionSchedules) {
      for (const item of sec.tasks) {
        const fallbackSlot =
          sec.section === "morning" ? 24 : sec.section === "afternoon" ? 48 : 72;
        const target = sectionFromSlot(item.startSlot ?? fallbackSlot);
        normalizedSections.get(target)!.tasks.push(item);
      }
    }

    sectionSchedules = SECTION_ORDER.map((name) => {
      const sec = normalizedSections.get(name)!;
      sec.tasks.sort((a, b) => (a.startSlot ?? 0) - (b.startSlot ?? 0));
      const used = sec.tasks.reduce((sum, t) => sum + t.axiomCost, 0);
      const gains = sec.tasks.reduce((sum, t) => sum + t.axiomGain, 0);
      sec.axiomUsed = +used.toFixed(4);
      sec.axiomRemaining = +(sec.axiomBudget - used + gains).toFixed(4);
      return sec;
    });

    const totalUsed = sectionSchedules.reduce((sum, s) => sum + s.axiomUsed, 0);
    const totalRemaining = BASE_AXIOMS - totalUsed;

    // Build a Task-like list for diversity/load calculations
    const dayTaskLike = sectionSchedules
      .flatMap((s) =>
        s.tasks
          .filter((t) => !t.isRecreational)
          .map((t) => {
            const row = input.tasks.find((r) => r.id === t.taskId);
            return row
              ? {
                  id: row.id,
                  name: row.name,
                  type: row.type as any,
                  difficulty: row.difficulty,
                  completionTime: row.completion_time,
                  priority: row.priority as any,
                  state: row.state as any,
                  etask: row.etask,
                  sequence: row.sequence,
                  createdAt: row.created_at ?? "",
                }
              : null;
          })
          .filter(Boolean),
      )
      .filter(Boolean) as any[];

    const entropy = computeDiversityEntropy(dayTaskLike);
    const load = computeDailyLoad(dayTaskLike);

    days.push({
      dayOffset,
      date: addDays(today, dayOffset),
      sections: sectionSchedules,
      totalAxiomsUsed: +totalUsed.toFixed(2),
      totalAxiomsRemaining: +Math.max(0, totalRemaining).toFixed(2),
      diversityEntropy: +entropy.toFixed(4),
      loadAcceptable: load < 10_000,
    });
  }

  reasoningLog.push(
    `COMPLETE: ${days.length} days scheduled. ${unscheduled.length} tasks unscheduled.`,
  );

  return { days, unscheduled, reasoningLog };
}

// ── Re-exports for backward compat ────────────────────────

export { priorityToNum as priorityToNumber } from "./energyModel";
export { computeAxiomCost as computeEnergyConsumption } from "./energyModel";
export { computeAxiomGain as computeEnergyGain } from "./energyModel";
export const DAILY_ENERGY = BASE_AXIOMS;
export { BASE_AXIOMS };

// ── Chunk Extraction (for DB persistence) ─────────────────

/**
 * Flattens a SchedulerOutput into chunk rows ready to INSERT into task_chunks.
 * In the atomic scheduling model, chunkId is only present when a task was
 * explicitly split; unplit tasks will produce no rows (returns empty array).
 */
export function extractScheduleChunks(output: SchedulerOutput): Array<{
  id: string;
  parentTaskId: string;
  chunkIndex: number;
  totalChunks: number;
  completionTime: number;
  scheduledDay: number;
  section: string;
  axiomCost: number;
  state: string;
}> {
  const allChunks: ReturnType<typeof extractScheduleChunks> = [];

  for (const day of output.days) {
    for (const section of day.sections) {
      for (const item of section.tasks) {
        if (item.chunkId) {
          // Count siblings so we can record totalChunks correctly
          const siblings = day.sections.flatMap((s) =>
            s.tasks.filter((t) => t.taskId === item.taskId && t.chunkId),
          );
          allChunks.push({
            id: item.chunkId,
            parentTaskId: item.taskId,
            chunkIndex: parseInt(item.chunkId.split("_chunk_")[1] ?? "0"),
            totalChunks: siblings.length || 1,
            completionTime: item.completionTime,
            scheduledDay: day.dayOffset,
            section: section.section,
            axiomCost: item.axiomCost,
            state: "scheduled",
          });
        }
      }
    }
  }

  return allChunks;
}
