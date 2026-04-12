/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Task & Schedule API
   Handles task creation, retrieval, and automatic scheduling.

   Pipeline:
     processAndSaveGeneratedSubtasks
       → scoreSubtaskArray → INSERT tasks
       → runAndSaveSchedule → UPSERT schedules

   getLatestSchedule  — read-only access to stored DaySchedule[]
   triggerReschedule  — force a fresh scheduler run (manual button)
   ═══════════════════════════════════════════════════════════ */

"use server";

import { sql } from "@vercel/postgres";
import { cookies } from "next/headers";
import { verify } from "jsonwebtoken";
import type { Task, SchedulerOutput, RunSchedulerResult } from "@/lib/types";
import { scoreSubtaskArray, type SubtaskInput } from "@/lib/scoring";
import { runScheduler, taskToRow, type TaskRow } from "@/lib/schedule";
import crypto from "crypto";

// ── Terminal states — excluded from scheduling ─────────────
const TERMINAL_STATES = new Set(["completed", "sacrificed", "skipped"]);

// ── Auth Helper (Internal) ────────────────────────────────

async function getUserId(): Promise<number | null> {
  const cookieStore = await cookies();
  const token = cookieStore.get("token")?.value;
  if (!token || !process.env.JWT_SECRET) return null;
  try {
    const decoded = verify(token, process.env.JWT_SECRET) as { userId: number };
    return decoded.userId;
  } catch {
    return null;
  }
}

// ── Schema Bootstrap ──────────────────────────────────────

export async function createTasksTables(): Promise<void> {
  await sql`
    CREATE TABLE IF NOT EXISTS tasks (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      name VARCHAR(255) NOT NULL,
      type VARCHAR(50) NOT NULL,
      difficulty INT NOT NULL,
      completion_time INT NOT NULL,
      priority VARCHAR(20) NOT NULL DEFAULT 'normal',
      state VARCHAR(50) NOT NULL DEFAULT 'unscheduled',
      subject VARCHAR(255),
      deadline TIMESTAMPTZ,
      etask FLOAT NOT NULL DEFAULT 0,
      scheduled_slot JSONB,
      "sequence" INTEGER DEFAULT 0,
      urgency FLOAT DEFAULT 0,
      score FLOAT DEFAULT 0,
      days_remaining INTEGER DEFAULT 0,
      created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )
  `;

  await sql`
    CREATE TABLE IF NOT EXISTS section_weights (
      user_id INT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
      morning FLOAT NOT NULL DEFAULT 0.40,
      afternoon FLOAT NOT NULL DEFAULT 0.35,
      evening FLOAT NOT NULL DEFAULT 0.25,
      updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )
  `;

  await sql`
    CREATE TABLE IF NOT EXISTS section_performance (
      id SERIAL PRIMARY KEY,
      user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      section_name VARCHAR(50) NOT NULL,
      scheduled_axioms FLOAT NOT NULL,
      actual_axioms FLOAT NOT NULL,
      efficiency_ratio FLOAT NOT NULL,
      recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )
  `;

  await sql`
    CREATE TABLE IF NOT EXISTS schedules (
      user_id      INT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
      generated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
      start_date   DATE NOT NULL,
      days         JSONB NOT NULL,
      unscheduled  JSONB DEFAULT '[]'::jsonb
    )
  `;
}

export async function createScheduleTable(): Promise<void> {
  await createTasksTables();
}

// ── RowMapper ─────────────────────────────────────────────

function rowToTask(row: any): Task {
  return {
    id: row.id as string,
    name: row.name as string,
    type: row.type as Task["type"],
    difficulty: row.difficulty as number,
    completionTime: row.completion_time as number,
    priority: row.priority as Task["priority"],
    state: row.state as Task["state"],
    subject: (row.subject as string | null) ?? undefined,
    deadline: row.deadline
      ? new Date(row.deadline as string).toISOString()
      : undefined,
    etask: row.etask as number,
    sequence: (row.sequence as number | null) ?? 0,
    urgency: (row.urgency as number | null) ?? undefined,
    score: (row.score as number | null) ?? undefined,
    daysRemaining: (row.days_remaining as number | null) ?? undefined,
    scheduledSlot: (row.scheduled_slot as Task["scheduledSlot"]) ?? undefined,
    contentType: (row.content_type as Task["contentType"]) ?? undefined,
    createdAt: new Date(row.created_at as string).toISOString(),
  };
}

function normalizeJsonArray(value: unknown): any[] {
  if (Array.isArray(value)) return value;
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }
  return [];
}

// ── Internal: Fetch user profile by ID ───────────────────

async function fetchProfileById(userId: number): Promise<{
  timezone: string | null;
  wake_time: number;
  sleep_time: number;
  session_style: number;
  switch_buffer: number;
  deadline_style: number;
  peak_focus_windows: any[];
  low_energy_windows: any[];
  fixed_commitments: any[];
  hard_exclusions: any[];
  recovery_activities: any[];
} | null> {
  try {
    const result = await sql`
      SELECT
        timezone,
        wake_time,
        sleep_time,
        session_style,
        switch_buffer,
        deadline_style,
        peak_focus_windows,
        low_energy_windows,
        fixed_commitments,
        hard_exclusions,
        recovery_activities
      FROM users WHERE id = ${userId} LIMIT 1
    `;
    if (result.rows.length === 0) return null;
    const r = result.rows[0];
    return {
      timezone: (r.timezone as string | null) ?? null,
      wake_time: (r.wake_time as number) ?? 420,
      sleep_time: (r.sleep_time as number) ?? 1320,
      session_style: (r.session_style as number) ?? 0,
      switch_buffer: (r.switch_buffer as number) ?? 0,
      deadline_style: (r.deadline_style as number) ?? 0,
      peak_focus_windows: normalizeJsonArray(r.peak_focus_windows),
      low_energy_windows: normalizeJsonArray(r.low_energy_windows),
      fixed_commitments: normalizeJsonArray(r.fixed_commitments),
      hard_exclusions: normalizeJsonArray(r.hard_exclusions),
      recovery_activities: normalizeJsonArray(r.recovery_activities),
    };
  } catch {
    return null;
  }
}

// ── Internal: Fetch section weights by user ID ────────────

async function fetchSectionWeightsById(userId: number): Promise<{
  morning: number;
  afternoon: number;
  evening: number;
}> {
  const defaults = { morning: 0.40, afternoon: 0.35, evening: 0.25 };
  try {
    await createTasksTables();
    const result = await sql`
      SELECT morning, afternoon, evening
      FROM section_weights WHERE user_id = ${userId} LIMIT 1
    `;
    if (result.rows.length === 0) {
      await sql`
        INSERT INTO section_weights (user_id, morning, afternoon, evening)
        VALUES (${userId}, ${defaults.morning}, ${defaults.afternoon}, ${defaults.evening})
        ON CONFLICT (user_id) DO NOTHING
      `;
      return defaults;
    }
    const r = result.rows[0];
    return {
      morning:   (r.morning   as number) ?? defaults.morning,
      afternoon: (r.afternoon as number) ?? defaults.afternoon,
      evening:   (r.evening   as number) ?? defaults.evening,
    };
  } catch {
    // Table may not exist yet — fall back to defaults silently
    return defaults;
  }
}

// ── Internal: Run scheduler + upsert schedules table ─────

async function runAndSaveSchedule(userId: number): Promise<SchedulerOutput | null> {
  await createTasksTables();
  // 1. Fetch ALL tasks for this user
  let rawRows: any[];
  try {
    const result = await sql`
      SELECT id, user_id, name, type, difficulty, completion_time, priority,
             state, subject, deadline, etask, scheduled_slot, created_at,
             "sequence", urgency, score, days_remaining
      FROM tasks
      WHERE user_id = ${userId}
      ORDER BY "sequence" ASC, created_at ASC
    `;
    rawRows = result.rows;
  } catch (err: any) {
    if (err.message?.includes('relation "tasks" does not exist')) {
      await createTasksTables();
      return null; // no tasks yet → nothing to schedule
    }
    throw err;
  }

  // 2. Filter — exclude terminal states, keep everything else
  const activeRows: TaskRow[] = rawRows
    .filter((r) => !TERMINAL_STATES.has(r.state as string))
    .map((r) => ({
      id: r.id as string,
      user_id: r.user_id as number,
      name: r.name as string,
      type: r.type as string,
      difficulty: r.difficulty as number,
      completion_time: r.completion_time as number,
      priority: r.priority as "high" | "normal" | "low",
      state: r.state as string,
      subject: (r.subject as string | null) ?? null,
      deadline: r.deadline
        ? new Date(r.deadline as string).toISOString()
        : null,
      etask: r.etask as number,
      scheduled_slot: r.scheduled_slot ?? null,
      created_at: r.created_at
        ? new Date(r.created_at as string).toISOString()
        : undefined,
      sequence: (r.sequence as number) ?? 0,
      urgency: (r.urgency as number) ?? 0,
      score: (r.score as number) ?? 0,
      days_remaining: (r.days_remaining as number) ?? 0,
    }));

  if (activeRows.length === 0) return null;

  // 3. Fetch user profile and section weights in parallel
  const [profile, sectionWeights] = await Promise.all([
    fetchProfileById(userId),
    fetchSectionWeightsById(userId),
  ]);

  // 4. Run the pure scheduler
  const today = new Date().toISOString().split("T")[0];
  const output = runScheduler({
    tasks: activeRows,
    timezone: profile?.timezone ?? undefined,
    wake_time: profile?.wake_time ?? 420,
    sleep_time: profile?.sleep_time ?? 1320,
    session_style: profile?.session_style ?? 0,
    switch_buffer: profile?.switch_buffer ?? 0,
    deadline_style: profile?.deadline_style ?? 0,
    peak_focus_windows: (profile?.peak_focus_windows ?? []) as any,
    low_energy_windows: (profile?.low_energy_windows ?? []) as any,
    fixed_commitments: (profile?.fixed_commitments ?? []) as any,
    hard_exclusions: (profile?.hard_exclusions ?? []) as any,
    recovery_activities: (profile?.recovery_activities ?? []) as any,
    section_weights: sectionWeights,
    start_date: today,
  });

  // 5. Upsert into schedules (one active schedule per user)
  const daysJson = JSON.stringify(output.days);
  const unscheduledJson = JSON.stringify(output.unscheduled);

  const doUpsert = async () => sql`
    INSERT INTO schedules (user_id, start_date, days, unscheduled, generated_at)
    VALUES (${userId}, ${today}, ${daysJson}::jsonb, ${unscheduledJson}::jsonb, NOW())
    ON CONFLICT (user_id) DO UPDATE SET
      start_date   = EXCLUDED.start_date,
      days         = EXCLUDED.days,
      unscheduled  = EXCLUDED.unscheduled,
      generated_at = EXCLUDED.generated_at
  `;

  try {
    await doUpsert();
  } catch (err: any) {
    if (err.message?.includes('relation "schedules" does not exist')) {
      await createScheduleTable();
      await doUpsert();
    } else {
      throw err;
    }
  }

  // 6. Update tasks in the DB so Dashboard can filter by `t.state === 'scheduled'`
  const updatePromises = [];
  for (const day of output.days) {
    for (const section of day.sections) {
      for (const t of section.tasks) {
        if (t.isRecreational) continue;
        const newSlot = {
          day: day.dayOffset,
          startSlot: t.startSlot,
          endSlot: (t.startSlot || 0) + Math.ceil(t.completionTime / 15),
          fitnessScore: 5.0
        };
        updatePromises.push(sql`
          UPDATE tasks
          SET state = 'scheduled',
              scheduled_slot = ${JSON.stringify(newSlot)}::jsonb
          WHERE id = ${t.taskId} AND user_id = ${userId}
        `);
      }
    }
  }

  // Unscheduled tasks revert to null / 'unscheduled'
  for (const unscheduledId of output.unscheduled) {
    updatePromises.push(sql`
      UPDATE tasks
      SET state = 'unscheduled',
          scheduled_slot = NULL
      WHERE id = ${unscheduledId} AND user_id = ${userId}
    `);
  }

  if (updatePromises.length > 0) {
    await Promise.all(updatePromises);
  }

  return output;
}

// ── Public API ────────────────────────────────────────────

/**
 * Get all tasks for the current user.
 */
export async function getTasks(): Promise<{ tasks?: Task[]; error?: string }> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    const result = await sql`
      SELECT * FROM tasks WHERE user_id = ${userId} ORDER BY "sequence" ASC, created_at ASC;
    `;
    return { tasks: result.rows.map(rowToTask) };
  } catch (error: any) {
    if (error.message?.includes('relation "tasks" does not exist')) {
      await createTasksTables();
      return { tasks: [] };
    }
    console.error("Fetch tasks failed:", error);
    return { error: "Failed to fetch tasks" };
  }
}

/**
 * Add / upload a single task manually.
 */
export async function addTask(task: Task): Promise<{ success?: boolean; error?: string }> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    await sql`
      INSERT INTO tasks (
        id, user_id, name, type, difficulty, completion_time, priority, state, subject,
        deadline, etask, "sequence", urgency, score, days_remaining
      ) VALUES (
        ${task.id}, ${userId}, ${task.name}, ${task.type}, ${task.difficulty}, ${task.completionTime},
        ${task.priority}, ${task.state}, ${task.subject || null},
        ${task.deadline || null}, ${task.etask},
        ${task.sequence || 0}, ${task.urgency || 0}, ${task.score || 0}, ${task.daysRemaining || 0}
      )
      ON CONFLICT (id) DO UPDATE SET
        name           = EXCLUDED.name,
        type           = EXCLUDED.type,
        difficulty     = EXCLUDED.difficulty,
        completion_time = EXCLUDED.completion_time,
        priority       = EXCLUDED.priority,
        state          = EXCLUDED.state,
        subject        = EXCLUDED.subject,
        deadline       = EXCLUDED.deadline,
        etask          = EXCLUDED.etask,
        "sequence"     = EXCLUDED."sequence",
        urgency        = EXCLUDED.urgency,
        score          = EXCLUDED.score,
        days_remaining = EXCLUDED.days_remaining;
    `;
    return { success: true };
  } catch (error) {
    console.error("addTask failed:", error);
    return { error: "Failed to save task" };
  }
}

/**
 * Add a single manual task (from the Add Task modal), score it,
 * insert it into the DB, and run the scheduler on ALL active tasks.
 *
 * This bypasses the roadmap/AI generation pipeline entirely.
 */
export async function addAndScheduleTask(input: {
  name: string;
  type: string;
  difficulty: number;
  completionTime: number;
  priority: "high" | "normal" | "low";
  subject?: string;
  deadline?: string;
}): Promise<{
  success?: boolean;
  taskId?: string;
  unscheduled?: string[];
  schedulerLog?: string[];
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    const today = new Date().toISOString().split("T")[0];

    // ── 1. Build SubtaskInput[] with a single element ──────
    const subtaskInput: SubtaskInput[] = [
      {
        name: input.name,
        priority: input.priority,
        difficulty: input.difficulty,
        sequence: 1,
        completion_time: input.completionTime,
        deadline: input.deadline,
      },
    ];

    // ── 2. Score the task using the existing scoring engine ─
    const scored = scoreSubtaskArray(subtaskInput, today);
    const scoredTask = scored[0];

    // ── 3. Insert the scored task into the DB ──────────────
    const taskId = crypto.randomUUID();
    await addTask({
      id: taskId,
      name: input.name,
      type: input.type,
      difficulty: input.difficulty,
      completionTime: input.completionTime,
      priority: input.priority,
      state: "unscheduled",
      subject: input.subject,
      deadline: input.deadline,
      etask: scoredTask.etask,
      sequence: scoredTask.sequence,
      urgency: scoredTask.urgency,
      score: scoredTask.score,
      daysRemaining: scoredTask.days_remaining,
      createdAt: new Date().toISOString(),
    } as Task);

    // ── 4. Run scheduler on ALL active tasks (including the new one) ─
    const output = await runAndSaveSchedule(userId);

    return {
      success: true,
      taskId,
      unscheduled: output?.unscheduled ?? [],
      schedulerLog: output?.reasoningLog ?? [],
    };
  } catch (err: any) {
    console.error("addAndScheduleTask failed:", err);
    return { error: err.message || "Failed to add and schedule task" };
  }
}

/**
 * Add a single manual task (from Add Task modal), score it, and keep it in
 * the unscheduled pool. Scheduling happens only when user explicitly triggers
 * scheduler actions.
 */
export async function addTaskToPool(input: {
  name: string;
  type: string;
  difficulty: number;
  completionTime: number;
  priority: "high" | "normal" | "low";
  subject?: string;
  deadline?: string;
}): Promise<{
  success?: boolean;
  taskId?: string;
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    const today = new Date().toISOString().split("T")[0];
    const subtaskInput: SubtaskInput[] = [
      {
        name: input.name,
        priority: input.priority,
        difficulty: input.difficulty,
        sequence: 1,
        completion_time: input.completionTime,
        deadline: input.deadline,
      },
    ];

    const scored = scoreSubtaskArray(subtaskInput, today);
    const scoredTask = scored[0];
    const taskId = crypto.randomUUID();

    await addTask({
      id: taskId,
      name: input.name,
      type: input.type,
      difficulty: input.difficulty,
      completionTime: input.completionTime,
      priority: input.priority,
      state: "unscheduled",
      subject: input.subject,
      deadline: input.deadline,
      etask: scoredTask.etask,
      sequence: scoredTask.sequence,
      urgency: scoredTask.urgency,
      score: scoredTask.score,
      daysRemaining: scoredTask.days_remaining,
      createdAt: new Date().toISOString(),
    } as Task);

    return { success: true, taskId };
  } catch (err: any) {
    console.error("addTaskToPool failed:", err);
    return { error: err.message || "Failed to add task to pool" };
  }
}

/**
 * Process AI-generated subtasks, score them, save to DB, then
 * automatically run the scheduler and persist the new schedule.
 */
export async function processAndSaveGeneratedSubtasks(
  subtasks: SubtaskInput[],
  subject?: string,
): Promise<{
  success?: boolean;
  unscheduled?: string[];
  schedulerLog?: string[];
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    // ── 1. Score all subtasks ───────────────────────────────
    const today = new Date().toISOString().split("T")[0];
    const scoredSubtasks = scoreSubtaskArray(subtasks, today);

    // ── 2. Persist each scored subtask ─────────────────────
    for (let i = 0; i < subtasks.length; i++) {
      const input = subtasks[i];
      const scored = scoredSubtasks[i];

      await addTask({
        id: crypto.randomUUID(),
        name: input.name,
        type: (input as any).type || "learning",
        difficulty: input.difficulty,
        completionTime: input.completion_time,
        priority: input.priority,
        state: "unscheduled",
        subject,
        deadline: input.deadline,
        etask: scored.etask,
        sequence: scored.sequence,
        urgency: scored.urgency,
        score: scored.score,
        daysRemaining: scored.days_remaining,
        createdAt: new Date().toISOString(),
      } as Task);
    }

    // ── 3. Auto-trigger scheduler on ALL active tasks ──────
    const output = await runAndSaveSchedule(userId);

    return {
      success: true,
      unscheduled: output?.unscheduled ?? [],
      schedulerLog: output?.reasoningLog ?? [],
    };
  } catch (err: any) {
    console.error("processAndSaveGeneratedSubtasks failed:", err);
    return { error: err.message || "Failed to process subtasks" };
  }
}

/**
 * Returns the stored DaySchedule[] for the current user.
 * The matrix page calls this to paint the 96×7 grid.
 */
export async function getLatestSchedule(): Promise<{
  days?: any[];
  unscheduled?: string[];
  generatedAt?: string;
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    const result = await sql`
      SELECT days, unscheduled, generated_at
      FROM schedules
      WHERE user_id = ${userId}
      LIMIT 1
    `;

    if (result.rows.length === 0) {
      return { days: [], unscheduled: [] };
    }

    const row = result.rows[0];
    return {
      days: row.days as any[],
      unscheduled: row.unscheduled as string[],
      generatedAt: new Date(row.generated_at as string).toISOString(),
    };
  } catch (err: any) {
    if (err.message?.includes('relation "schedules" does not exist')) {
      await createScheduleTable();
      return { days: [], unscheduled: [] };
    }
    console.error("getLatestSchedule failed:", err);
    return { error: "Failed to fetch schedule" };
  }
}

/**
 * Manually force a fresh scheduler run for the current user.
 * Useful for a "Reschedule" button in the UI.
 */
export async function triggerReschedule(): Promise<{
  success?: boolean;
  unscheduled?: string[];
  schedulerLog?: string[];
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    const output = await runAndSaveSchedule(userId);
    return {
      success: true,
      unscheduled: output?.unscheduled ?? [],
      schedulerLog: output?.reasoningLog ?? [],
    };
  } catch (err: any) {
    console.error("triggerReschedule failed:", err);
    return { error: err.message || "Failed to reschedule" };
  }
}

// ══════════════════════════════════════════════════════════
// ADDITIONAL ACTIONS (required by TasksView & friend's UI)
// ══════════════════════════════════════════════════════════

/**
 * Alias for addTask — upsert an existing task object.
 * The friend's TasksView calls saveTask() after editing a task in the sidebar.
 */
export const saveTask = addTask;

/**
 * Delete a single task by ID.
 */
export async function deleteTask(
  taskId: string,
): Promise<{ success?: boolean; error?: string }> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };
  try {
    await sql`DELETE FROM tasks WHERE id = ${taskId} AND user_id = ${userId}`;
    return { success: true };
  } catch (err: any) {
    console.error("deleteTask failed:", err);
    return { error: err.message || "Failed to delete task" };
  }
}

/**
 * Update a task's state and optionally its scheduled_slot.
 * Called after drag-and-drop or marking complete.
 */
export async function updateTaskStateAndSlot(
  taskId: string,
  state: string,
  slot?: any,
): Promise<{ success?: boolean; error?: string }> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };
  try {
    const slotJson = slot ? JSON.stringify(slot) : null;
    await sql`
      UPDATE tasks
      SET state = ${state}, scheduled_slot = ${slotJson}::jsonb
      WHERE id = ${taskId} AND user_id = ${userId}
    `;
    return { success: true };
  } catch (err: any) {
    console.error("updateTaskStateAndSlot failed:", err);
    return { error: err.message || "Failed to update task" };
  }
}

/**
 * Bulk upsert an array of Task objects.
 * Called by TasksView after scheduling to sync the store with the DB.
 */
export async function syncTasks(
  tasks: Task[],
): Promise<{ success?: boolean; error?: string }> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };
  try {
    for (const task of tasks) {
      await addTask(task);
    }
    return { success: true };
  } catch (err: any) {
    console.error("syncTasks failed:", err);
    return { error: err.message || "Failed to sync tasks" };
  }
}

/**
 * Delete all tasks in the 'unscheduled' state for the current user.
 * Used to flush the pool before a fresh RAG extraction.
 */
export async function clearUnscheduledTasks(): Promise<{
  success?: boolean;
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };
  try {
    await sql`DELETE FROM tasks WHERE user_id = ${userId} AND state = 'unscheduled'`;
    return { success: true };
  } catch (err: any) {
    console.error("clearUnscheduledTasks failed:", err);
    return { error: err.message || "Failed to clear tasks" };
  }
}

// ── Section Weights (public API) ──────────────────────────

/**
 * Public version of fetchSectionWeightsById — reads from the current user.
 */
export async function getSectionWeights(): Promise<{
  weights?: { morning: number; afternoon: number; evening: number };
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };
  const weights = await fetchSectionWeightsById(userId);
  return { weights };
}

/**
 * Returns all scheduler-influencing inputs for the current user.
 * This is the canonical server action payload for allocation diagnostics.
 */
export async function getSchedulerInfluenceData(): Promise<{
  data?: {
    timezone: string | null;
    wake_time: number;
    sleep_time: number;
    session_style: number;
    switch_buffer: number;
    deadline_style: number;
    peak_focus_windows: any[];
    low_energy_windows: any[];
    fixed_commitments: any[];
    hard_exclusions: any[];
    recovery_activities: any[];
    section_weights: { morning: number; afternoon: number; evening: number };
  };
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    await createTasksTables();

    const [userResult, sectionWeights] = await Promise.all([
      sql`
        SELECT
          timezone,
          wake_time,
          sleep_time,
          session_style,
          switch_buffer,
          deadline_style,
          peak_focus_windows,
          low_energy_windows,
          fixed_commitments,
          hard_exclusions,
          recovery_activities
        FROM users
        WHERE id = ${userId}
        LIMIT 1
      `,
      fetchSectionWeightsById(userId),
    ]);

    if (userResult.rows.length === 0) {
      return { error: "User profile not found" };
    }

    const userRow = userResult.rows[0];
    return {
      data: {
        timezone: (userRow.timezone as string | null) ?? null,
        wake_time: (userRow.wake_time as number) ?? 420,
        sleep_time: (userRow.sleep_time as number) ?? 1320,
        session_style: (userRow.session_style as number) ?? 0,
        switch_buffer: (userRow.switch_buffer as number) ?? 0,
        deadline_style: (userRow.deadline_style as number) ?? 0,
        peak_focus_windows: normalizeJsonArray(userRow.peak_focus_windows),
        low_energy_windows: normalizeJsonArray(userRow.low_energy_windows),
        fixed_commitments: normalizeJsonArray(userRow.fixed_commitments),
        hard_exclusions: normalizeJsonArray(userRow.hard_exclusions),
        recovery_activities: normalizeJsonArray(userRow.recovery_activities),
        section_weights: sectionWeights,
      },
    };
  } catch (err: any) {
    console.error("getSchedulerInfluenceData failed:", err);
    return { error: err.message || "Failed to fetch scheduler influence data" };
  }
}

/**
 * Persist updated section weights for the current user.
 */
export async function updateSectionWeightsDB(weights: {
  morning: number;
  afternoon: number;
  evening: number;
}): Promise<{ success?: boolean; error?: string }> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };
  try {
    await sql`
      INSERT INTO section_weights (user_id, morning, afternoon, evening)
      VALUES (${userId}, ${weights.morning}, ${weights.afternoon}, ${weights.evening})
      ON CONFLICT (user_id) DO UPDATE SET
        morning   = EXCLUDED.morning,
        afternoon = EXCLUDED.afternoon,
        evening   = EXCLUDED.evening,
        updated_at = NOW()
    `;
    return { success: true };
  } catch (err: any) {
    console.error("updateSectionWeightsDB failed:", err);
    return { error: err.message || "Failed to update weights" };
  }
}

/**
 * Record observed section performance (actual vs scheduled axioms).
 */
export async function recordSectionPerformance(
  sectionName: string,
  scheduledAxioms: number,
  actualAxioms: number,
): Promise<{ success?: boolean; error?: string }> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };
  try {
    const efficiencyRatio =
      scheduledAxioms > 0 ? actualAxioms / scheduledAxioms : 1;
    await sql`
      INSERT INTO section_performance
        (user_id, section_name, scheduled_axioms, actual_axioms, efficiency_ratio)
      VALUES
        (${userId}, ${sectionName}, ${scheduledAxioms}, ${actualAxioms}, ${efficiencyRatio})
    `;
    return { success: true };
  } catch (err: any) {
    console.error("recordSectionPerformance failed:", err);
    return { error: err.message || "Failed to record performance" };
  }
}

/**
 * Mark a section complete: records its performance and triggers a weight update.
 */
export async function markSectionComplete(
  sectionName: string,
  scheduledAxioms: number,
  actualAxioms: number,
): Promise<{ success?: boolean; updatedWeights?: any; error?: string }> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };
  try {
    await recordSectionPerformance(sectionName, scheduledAxioms, actualAxioms);
    // Re-read current weights and return them so the caller can dispatch
    const weights = await fetchSectionWeightsById(userId);
    return { success: true, updatedWeights: weights };
  } catch (err: any) {
    console.error("markSectionComplete failed:", err);
    return { error: err.message || "Failed to mark section complete" };
  }
}

// ── RAG Source Persistence ────────────────────────────────

async function ensureSourcesTable(): Promise<void> {
  await sql`
    CREATE TABLE IF NOT EXISTS sources (
      id SERIAL PRIMARY KEY,
      user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      topic VARCHAR(255),
      content TEXT NOT NULL,
      created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )
  `;
}

/**
 * Persist RAG context text for a topic so it survives page reloads.
 */
export async function saveSource(
  content: string,
  topic?: string,
): Promise<{ success?: boolean; error?: string }> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };
  try {
    await ensureSourcesTable();
    await sql`
      INSERT INTO sources (user_id, topic, content)
      VALUES (${userId}, ${topic ?? null}, ${content})
    `;
    return { success: true };
  } catch (err: any) {
    console.error("saveSource failed:", err);
    return { error: err.message || "Failed to save source" };
  }
}

/**
 * Retrieve the most recent RAG context for the current user.
 */
export async function getSource(): Promise<{
  content?: string;
  topic?: string;
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };
  try {
    await ensureSourcesTable();
    const result = await sql`
      SELECT content, topic FROM sources
      WHERE user_id = ${userId}
      ORDER BY created_at DESC
      LIMIT 1
    `;
    if (result.rows.length === 0) return {};
    return {
      content: result.rows[0].content as string,
      topic: result.rows[0].topic as string | undefined,
    };
  } catch (err: any) {
    console.error("getSource failed:", err);
    return { error: err.message || "Failed to get source" };
  }
}

/**
 * Delete all RAG context rows for the current user.
 */
export async function clearSources(): Promise<{
  success?: boolean;
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };
  try {
    await ensureSourcesTable();
    await sql`DELETE FROM sources WHERE user_id = ${userId}`;
    return { success: true };
  } catch (err: any) {
    console.error("clearSources failed:", err);
    return { error: err.message || "Failed to clear sources" };
  }
}

// ── Chunk Persistence (task_chunks table) ─────────────────

/**
 * Persist a batch of schedule chunks to the task_chunks table.
 * In the atomic model this is usually a no-op (empty array),
 * but kept for compatibility with the chunked scheduling path.
 */
export async function saveChunks(
  chunks: Array<{
    id: string;
    parentTaskId: string;
    chunkIndex: number;
    totalChunks: number;
    completionTime: number;
    scheduledDay: number;
    section: string;
    axiomCost: number;
    state: string;
  }>,
): Promise<{ success?: boolean; error?: string }> {
  if (chunks.length === 0) return { success: true };
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };
  try {
    for (const chunk of chunks) {
      await sql`
        INSERT INTO task_chunks
          (id, parent_task_id, user_id, chunk_index, total_chunks,
           duration, scheduled_day, section, axiom_cost, state)
        VALUES
          (${chunk.id}, ${chunk.parentTaskId}, ${userId},
           ${chunk.chunkIndex}, ${chunk.totalChunks},
           ${chunk.completionTime}, ${chunk.scheduledDay},
           ${chunk.section}, ${chunk.axiomCost}, ${chunk.state})
        ON CONFLICT (id) DO UPDATE SET
          state = EXCLUDED.state,
          scheduled_day = EXCLUDED.scheduled_day,
          section = EXCLUDED.section
      `;
    }
    return { success: true };
  } catch (err: any) {
    // task_chunks table may not exist — non-fatal in atomic mode
    console.warn("saveChunks skipped:", err.message);
    return { success: true };
  }
}

// ── Primary Scheduler Bridge ──────────────────────────────

/**
 * Client-facing scheduler action.
 * Accepts the Task[] currently in the React store, converts them to TaskRows,
 * runs the deterministic scheduler, persists the result to the schedules table,
 * and returns RunSchedulerResult so the caller can dispatch SET_SECTIONS.
 *
 * This is the function TasksView calls on load and on "Re-Schedule".
 */
export async function runSchedulerAction(
  tasks: Task[],
  startDate?: string,
): Promise<RunSchedulerResult> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    // 1. Fetch section weights and profile in parallel
    const [sectionWeights, profile] = await Promise.all([
      fetchSectionWeightsById(userId),
      fetchProfileById(userId),
    ]);

    // 2. Filter out terminal states, then convert Task → TaskRow
    const activeRows: TaskRow[] = tasks
      .filter((t) => !TERMINAL_STATES.has(t.state))
      .map(taskToRow);

    if (activeRows.length === 0) {
      return { days: [], unscheduled: [], reasoningLog: ["No active tasks to schedule."] };
    }

    // 3. Run the pure deterministic scheduler
    const today = startDate ?? new Date().toISOString().split("T")[0];
    const output = runScheduler({
      tasks: activeRows,
      timezone: profile?.timezone ?? undefined,
      wake_time: profile?.wake_time ?? 420,
      sleep_time: profile?.sleep_time ?? 1320,
      session_style: profile?.session_style ?? 0,
      switch_buffer: profile?.switch_buffer ?? 0,
      deadline_style: profile?.deadline_style ?? 0,
      peak_focus_windows: (profile?.peak_focus_windows ?? []) as any,
      low_energy_windows: (profile?.low_energy_windows ?? []) as any,
      fixed_commitments: (profile?.fixed_commitments ?? []) as any,
      hard_exclusions: (profile?.hard_exclusions ?? []) as any,
      recovery_activities: (profile?.recovery_activities ?? []) as any,
      section_weights: sectionWeights,
      start_date: today,
    });

    // 4. Persist JSONB blob into schedules table
    const daysJson = JSON.stringify(output.days);
    const unscheduledJson = JSON.stringify(output.unscheduled);

    const doUpsert = () => sql`
      INSERT INTO schedules (user_id, start_date, days, unscheduled, generated_at)
      VALUES (${userId}, ${today}, ${daysJson}::jsonb, ${unscheduledJson}::jsonb, NOW())
      ON CONFLICT (user_id) DO UPDATE SET
        start_date   = EXCLUDED.start_date,
        days         = EXCLUDED.days,
        unscheduled  = EXCLUDED.unscheduled,
        generated_at = EXCLUDED.generated_at
    `;

    try {
      await doUpsert();
    } catch (err: any) {
      if (err.message?.includes('relation "schedules" does not exist')) {
        await createScheduleTable();
        await doUpsert();
      } else {
        throw err;
      }
    }

    return {
      days: output.days,
      unscheduled: output.unscheduled,
      reasoningLog: output.reasoningLog,
      updatedWeights: sectionWeights,
    };
  } catch (err: any) {
    console.error("runSchedulerAction failed:", err);
    return { error: err.message || "Failed to run scheduler" };
  }
}

// ── Primary AI-Powered Task Generator ────────────────────────
// This is the COMPLETE pipeline entry point:
//   User prompt → Gemini (RoadmapDAGNode[]) → SubtaskInput[]
//              → scoreSubtaskArray → INSERT tasks → runAndSaveSchedule

import { generateRoadmapAction } from "./roadmap";
import type { RoadmapDAGNode } from "@/lib/types";

function nodeToSubtask(
  node: RoadmapDAGNode,
  idx: number,
  deadline?: string,
): import("@/lib/scoring").SubtaskInput {
  // Map RoadmapDAGNode priority → SubtaskInput priority
  const priority: "high" | "normal" | "low" =
    node.priority === "high"
      ? "high"
      : node.priority === "low"
        ? "low"
        : "normal";

  let parsedDeadline: string | undefined = undefined;
  if (deadline) {
    const timestamp = Date.parse(deadline);
    if (!isNaN(timestamp)) {
      parsedDeadline = new Date(timestamp).toISOString();
    }
  }

  return {
    name: node.title,
    priority,
    difficulty: Math.min(10, Math.max(1, node.difficulty ?? 5)),
    // sequence from the DAG node (already set by roadmap-agent)
    sequence: node.sequence ?? idx + 1,
    // estimatedMinutes is the DAG field; maps to completion_time
    completion_time: node.estimatedMinutes ?? 60,
    deadline: parsedDeadline,
  };
}

/**
 * THE FULL PIPELINE:
 *   User goal string → Gemini API → RoadmapDAGNode[] → SubtaskInput[]
 *   → scoreSubtaskArray → INSERT tasks → runAndSaveSchedule → UPSERT schedules
 *
 * Called by TasksView "Generate Tasks" button.
 * Returns the number of tasks inserted so the UI can show confirmation.
 */
export async function generateTasksFromRoadmapGoal(
  goal: string,
  deadline?: string,
  syllabus?: string,
): Promise<{
  count?: number;
  roadmapId?: string;
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    // 1. Call Gemini via the existing roadmap action (handles rate-limit + retry)
    const { roadmap, error: genErr } = await generateRoadmapAction({
      goal,
      deadline: deadline || "in 4 weeks",
      syllabus: syllabus || undefined,
    });

    if (genErr || !roadmap) {
      return { error: genErr || "Gemini returned no roadmap" };
    }

    // 2. Map DAG nodes → SubtaskInput[]
    //    Sort by sequence so prerequisites are inserted first
    const sorted = [...roadmap.nodes].sort(
      (a, b) => (a.sequence ?? 0) - (b.sequence ?? 0),
    );
    const subtasks = sorted.map((node, idx) =>
      nodeToSubtask(node, idx, deadline),
    );

    // 3. Run through scoring → INSERT tasks → runAndSaveSchedule
    const result = await processAndSaveGeneratedSubtasks(subtasks, goal);
    if (result.error) return { error: result.error };

    return { count: subtasks.length, roadmapId: roadmap.id };
  } catch (err: any) {
    console.error("generateTasksFromRoadmapGoal failed:", err);
    return { error: err.message || "Failed to generate tasks" };
  }
}
