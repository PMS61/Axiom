/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Schedule Constraints Engine
   Cognitive load balancing + diversity enforcement.

   Formulas:
     L_day = Σ(difficulty × completionTime)         [cognitive load]
     H     = −Σ p_i × log₂(p_i)              [Shannon diversity entropy]
     Anti-starvation: after every 3 high-priority tasks, insert 1 lower
   ═══════════════════════════════════════════════════════════ */

import type { Task } from "./types";

// ── Cognitive Load ────────────────────────────────────────

/**
 * L_day = Σ(difficulty × completionTime) for a set of tasks.
 * Returns the aggregate cognitive load for one day.
 */
export function computeDailyLoad(tasks: Task[]): number {
  return tasks.reduce((sum, t) => sum + t.difficulty * t.completionTime, 0);
}

// ── Mean & Standard Deviation ─────────────────────────────

/**
 * Returns { mean, std } of a numeric array.
 * std uses population standard deviation (N denominator).
 */
export function meanStdDev(values: number[]): { mean: number; std: number } {
  if (values.length === 0) return { mean: 0, std: 0 };
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance =
    values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
  return { mean: +mean.toFixed(4), std: +Math.sqrt(variance).toFixed(4) };
}

/**
 * Returns true if a candidate day load is within mean ± std.
 * Prevents any single day from being overloaded relative to the week.
 *
 * @param candidateLoad — load of the day being evaluated
 * @param allDayLoads   — loads of all planned days (including candidate is fine)
 */
export function isDayLoadAcceptable(
  candidateLoad: number,
  allDayLoads: number[],
): boolean {
  if (allDayLoads.length < 2) return true; // not enough data to constrain
  const { mean, std } = meanStdDev(allDayLoads);
  return candidateLoad <= mean + std;
}

// ── Diversity Entropy ─────────────────────────────────────

/**
 * H = −Σ p_i × log₂(p_i)
 * p_i = (count of tasks with type i) / total tasks.
 * H = 0 if all tasks are the same type; higher = more diverse.
 *
 * Constraint: H > 0.5 for a healthy schedule.
 */
export function computeDiversityEntropy(tasks: Task[]): number {
  if (tasks.length === 0) return 0;

  const typeCounts: Record<string, number> = {};
  for (const t of tasks) {
    typeCounts[t.type] = (typeCounts[t.type] ?? 0) + 1;
  }

  const n = tasks.length;
  let H = 0;
  for (const count of Object.values(typeCounts)) {
    const p = count / n;
    if (p > 0) H -= p * Math.log2(p);
  }

  return +H.toFixed(6);
}

/**
 * Returns true if the task set has sufficient type diversity (H > 0.5).
 */
export function hasSufficientDiversity(tasks: Task[]): boolean {
  return computeDiversityEntropy(tasks) > 0.5;
}

// ── Anti-Starvation ───────────────────────────────────────

const HIGH_PRIORITY_THRESHOLD = 7; // numeric priority ≥ this → "high"
const ANTI_STARVATION_STREAK = 3;  // insert 1 low after every N high

/**
 * Reorders a task queue so that after every ANTI_STARVATION_STREAK
 * high-priority tasks (numeric priority ≥ 7) one lower-priority task
 * is inserted. Preserves original relative sequence within each group.
 */
export function applyAntiStarvation<T extends { priorityNum: number }>(
  tasks: T[],
): T[] {
  const high = tasks.filter((t) => t.priorityNum >= HIGH_PRIORITY_THRESHOLD);
  const low = tasks.filter((t) => t.priorityNum < HIGH_PRIORITY_THRESHOLD);

  const result: T[] = [];
  let hi = 0;
  let lo = 0;
  let streak = 0;

  while (hi < high.length || lo < low.length) {
    if (streak >= ANTI_STARVATION_STREAK && lo < low.length) {
      result.push(low[lo++]);
      streak = 0;
    } else if (hi < high.length) {
      result.push(high[hi++]);
      streak++;
    } else {
      result.push(low[lo++]);
    }
  }

  return result;
}
