/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Task Scoring Engine
   Deterministic priority scoring with deadline urgency.

   Formulas:
     U     = priority_num / (D + 1)        [urgency; D = days remaining]
     Score = (2 × priority + difficulty + 3 × U) / E_task
   ═══════════════════════════════════════════════════════════ */

import { computeAxiomCost, priorityToNum } from "./energyModel";
import type { Task } from "./types";

// ── Urgency ───────────────────────────────────────────────

/**
 * U = priority_num / (D + 1)
 * D = days remaining until deadline (0 if overdue/no deadline → high urgency).
 */
export function computeUrgency(priorityNum: number, daysRemaining: number): number {
  const D = Math.max(0, daysRemaining);
  return +(priorityNum / (D + 1)).toFixed(6);
}

// ── Task Score ────────────────────────────────────────────

/**
 * Score = (2 × priority_num + difficulty + 3 × U) / E_task
 * Higher score = schedule earlier.
 * Returns 0 if E_task ≤ 0 (safety guard).
 */
export function computeTaskScore(
  priorityNum: number,
  difficulty: number,
  urgency: number,
  eTask: number,
): number {
  if (eTask <= 0) return 0;
  return +((2 * priorityNum + difficulty + 3 * urgency) / eTask).toFixed(6);
}

// ── Full Task Enrichment ───────────────────────────────────

export interface ScoredTask {
  task: Task;
  priorityNum: number;
  eTask: number;       // axiom cost
  daysRemaining: number;
  urgency: number;
  score: number;
}

/**
 * Enriches a Task with all scheduling scores.
 * @param task     — raw task
 * @param today    — reference date string YYYY-MM-DD
 */
export function enrichTask(task: Task, today: string): ScoredTask {
  const priorityNum = priorityToNum(task.priority);
  const eTask = computeAxiomCost(task.difficulty, priorityNum);

  let daysRemaining = 365; // default: no deadline pressure
  if (task.deadline) {
    const deadlineMs = new Date(task.deadline).getTime();
    const todayMs = new Date(today).getTime();
    daysRemaining = Math.max(0, Math.floor((deadlineMs - todayMs) / 86_400_000));
  }

  const urgency = computeUrgency(priorityNum, daysRemaining);
  const score = computeTaskScore(priorityNum, task.difficulty, urgency, eTask);

  return { task, priorityNum, eTask, daysRemaining, urgency, score };
}

// ── Sorting ───────────────────────────────────────────────

/**
 * Comparator for ScoredTask[].
 * Primary: score DESC. Tie-break: id ASC (deterministic).
 */
export function compareByScore(a: ScoredTask, b: ScoredTask): number {
  if (b.score !== a.score) return b.score - a.score;
  return a.task.id.localeCompare(b.task.id);
}

// ── Array-Based Subtask Scoring ───────────────────────────

export interface SubtaskInput {
  name: string;
  priority: "high" | "normal" | "low";
  difficulty: number;
  sequence: number;
  completion_time: number; // in minutes
  deadline?: string;        // optional ISO date string
}

export interface SubtaskScoreOutput {
  priorityNum: number;
  etask: number;
  completion_time: number;
  days_remaining: number;
  urgency: number;
  score: number;
  sequence: number; // returned so the caller can retain sequence info
}

/**
 * Enriches an array of sequentially generated subtasks with Axiom metrics.
 * 
 * Schema Suggestion Added: We are keeping `sequence` in the output, and using 
 * it to slightly boost the `score` of earlier sequence numbers (e.g. sequence 1 
 * gets a higher score bump than sequence 5). This makes sure the algorithm 
 * naturally favors chronological prerequisites while respecting the core math.
 */
export function scoreSubtaskArray(subtasks: SubtaskInput[], today: string): SubtaskScoreOutput[] {
  return subtasks.map(subtask => {
    const priorityNum = priorityToNum(subtask.priority);
    const etask = computeAxiomCost(subtask.difficulty, priorityNum);

    let days_remaining = 365; // Default: no deadline pressure
    if (subtask.deadline) {
      const deadlineMs = new Date(subtask.deadline).getTime();
      const todayMs = new Date(today).getTime();
      days_remaining = Math.max(0, Math.floor((deadlineMs - todayMs) / 86_400_000));
    }

    const urgency = computeUrgency(priorityNum, days_remaining);
    
    // Base Axiom Score
    let baseScore = computeTaskScore(priorityNum, subtask.difficulty, urgency, etask);
    
    // ── Suggestion ─────────────────────────────────────
    // To benefit the algorithm, lower sequence numbers (prerequisites) 
    // should naturally score higher than later sequence numbers 
    // to ensure they are handled first. We add a small sequence bonus.
    // e.g. sequence=1 gives +1.0, sequence=5 gives +0.6
    const sequenceBonus = Math.max(0, (11 - subtask.sequence) * 0.1); 
    const finalScore = +(baseScore + sequenceBonus).toFixed(6);

    return {
      priorityNum,
      etask,
      completion_time: subtask.completion_time,
      days_remaining,
      urgency,
      score: finalScore,
      sequence: subtask.sequence
    };
  });
}
