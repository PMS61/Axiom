/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Task Chunking / Fragmentation
   Splits large tasks into time-distributed fragments.

   Formulas:
     chunk_size = min(90, max(30, completionTime / ceil(completionTime / 60)))
     n          = ceil(completionTime / chunk_size)
     gap        = floor(D / n)          [days between chunks]
   ═══════════════════════════════════════════════════════════ */

import type { Task, TaskChunk, SectionName } from "./types";

// ── Chunk Size ────────────────────────────────────────────

/**
 * Computes the completionTime (minutes) of each chunk.
 * Ensures chunks are between 30–90 minutes.
 */
export function computeChunkSize(totalDuration: number): number {
  const rawChunks = Math.ceil(totalDuration / 60);
  const raw = totalDuration / rawChunks;
  return Math.min(90, Math.max(30, Math.round(raw)));
}

/**
 * Returns number of chunks needed for a given completionTime.
 * Always at least 1.
 */
export function computeChunkCount(
  totalDuration: number,
  chunkSize: number,
): number {
  return Math.max(1, Math.ceil(totalDuration / chunkSize));
}

// ── Day Gap ───────────────────────────────────────────────

/**
 * gap = floor(D / n)
 * D = days remaining; n = number of chunks.
 * Minimum gap = 1 day (so chunks land on different days unless D is 0).
 */
export function computeChunkGap(daysRemaining: number, chunkCount: number): number {
  if (chunkCount <= 1) return 0;
  return Math.max(1, Math.floor(daysRemaining / chunkCount));
}

// ── Section Assignment ─────────────────────────────────────

/**
 * Simple round-robin section assignment for chunks.
 * First chunk → morning (highest focus), subsequent → afternoon, evening.
 */
const SECTION_ROTATION: SectionName[] = ["morning", "afternoon", "evening"];

export function sectionForChunkIndex(chunkIndex: number): SectionName {
  return SECTION_ROTATION[chunkIndex % SECTION_ROTATION.length];
}

// ── Fragment Task ─────────────────────────────────────────

/**
 * Fragments a task into ordered TaskChunks distributed across days.
 * Returns an empty array for tasks ≤ 60 min (no fragmentation needed).
 *
 * @param task          — the task to fragment
 * @param daysRemaining — calendar days until deadline
 * @param axiomCost     — pre-computed axiom cost for the full task
 */
export function fragmentTask(
  task: Task,
  daysRemaining: number,
  axiomCostPerMinute: number,
): TaskChunk[] {
  // Tasks ≤ 60 min are not fragmented
  if (task.completionTime <= 60) return [];

  const chunkSize = computeChunkSize(task.completionTime);
  const n = computeChunkCount(task.completionTime, chunkSize);

  // Last chunk may be shorter if completionTime isn't evenly divisible
  const gap = computeChunkGap(daysRemaining, n);

  const chunks: TaskChunk[] = [];
  let remainingMinutes = task.completionTime;

  for (let i = 0; i < n; i++) {
    const completionTime = i === n - 1
      ? Math.max(15, remainingMinutes) // last chunk gets the remainder
      : chunkSize;
    remainingMinutes -= chunkSize;

    chunks.push({
      id: `${task.id}_chunk_${i}`,
      parentTaskId: task.id,
      chunkIndex: i,
      totalChunks: n,
      completionTime,
      scheduledDay: i * gap,
      section: sectionForChunkIndex(i),
      axiomCost: +(axiomCostPerMinute * completionTime).toFixed(4),
      state: "unscheduled",
    });
  }

  return chunks;
}

/**
 * Returns whether a task should be chunked.
 * Criterion: completionTime > 60 min AND has a deadline within 30 days.
 */
export function shouldChunk(task: Task, daysRemaining: number): boolean {
  return task.completionTime > 60 && daysRemaining <= 30;
}
