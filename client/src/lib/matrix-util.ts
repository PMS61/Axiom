/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Matrix Utility
   Generates the 96×7 slot matrix for the schedule view.

   Data sources (in order of application):
     1. User profile  → sleep / peak / commitment / exclusion overlays
     2. schedules table → task placement painted from DaySchedule[]

   Day columns: dayOffset 0 (today) … 6 (today+6)
   Row slots:   0-95 (each = 15 min, 0 = midnight, 24 = 6 am …)
   ═══════════════════════════════════════════════════════════ */

import { getLatestSchedule } from "@/app/actions/tasks";
import { getUserProfile } from "@/app/actions/auth";

export interface MatrixSlot {
  /** Axiom cost of the task occupying this slot (0 = empty or recreational) */
  etask: number;
  /** ID of the task placed here, if any */
  taskId?: string;
  /** Display name of the task placed here */
  taskName?: string;
  /** True if this slot falls inside the user's peak-focus window */
  isPeak: boolean;
  /** True if this slot falls inside a low-energy window */
  isLow: boolean;
  /** True if this slot is inside the sleep window */
  isSleep: boolean;
  /** True if a fixed commitment occupies this slot */
  isFixedCommitment: boolean;
  /** True if a hard exclusion blocks this slot */
  isHardExclusion: boolean;
  /** True when none of sleep / commitment / exclusion apply */
  isAvailable: boolean;
  /** True when the task in this slot has been completed or skipped */
  actualCompletion: boolean;
}

/**
 * Builds a 7×96 matrix (7 days × 96 fifteen-minute slots) for the
 * current authenticated user.
 *
 * Returns `{ error }` on auth or DB failure.
 * Returns an empty matrix (all slots default) when no schedule exists yet.
 */
export async function getUserMatrix(): Promise<MatrixSlot[][] | { error: string }> {
  try {
    // Fetch profile and latest schedule in parallel
    const [profileRes, scheduleRes] = await Promise.all([
      getUserProfile(),
      getLatestSchedule(),
    ]);

    if ("error" in profileRes) return { error: profileRes.error as string };
    if (scheduleRes.error) return { error: scheduleRes.error };

    const profile = profileRes.user;

    // ── Initialise 7×96 empty matrix ─────────────────────────
    const matrix: MatrixSlot[][] = Array.from({ length: 7 }, () =>
      Array.from({ length: 96 }, (): MatrixSlot => ({
        etask: 0,
        isPeak: false,
        isLow: false,
        isSleep: false,
        isFixedCommitment: false,
        isHardExclusion: false,
        isAvailable: true,
        actualCompletion: false,
      })),
    );

    const today = new Date();
    today.setHours(0, 0, 0, 0);

    // ── 1. Apply profile overlays ─────────────────────────────
    for (let dayIdx = 0; dayIdx < 7; dayIdx++) {
      const currentDay = new Date(today);
      currentDay.setDate(today.getDate() + dayIdx);
      const dayOfWeek = currentDay.getDay(); // 0=Sun … 6=Sat

      for (let slot = 0; slot < 96; slot++) {
        const slotMins = slot * 15;
        const cell = matrix[dayIdx][slot];

        // Peak focus windows
        if (
          profile.peak_focus_windows?.some(
            (w: any) => slotMins >= w.start_min && slotMins < w.end_min,
          )
        ) {
          cell.isPeak = true;
        }

        // Low energy windows
        if (
          profile.low_energy_windows?.some(
            (w: any) => slotMins >= w.start_min && slotMins < w.end_min,
          )
        ) {
          cell.isLow = true;
        }

        // Sleep window
        const { wake_time, sleep_time } = profile;
        if (wake_time != null && sleep_time != null) {
          if (sleep_time > wake_time) {
            // Normal: awake from wake_time to sleep_time
            if (slotMins >= sleep_time || slotMins < wake_time) cell.isSleep = true;
          } else {
            // Inverted (e.g. nap schedule)
            if (slotMins >= sleep_time && slotMins < wake_time) cell.isSleep = true;
          }
        }

        // Fixed commitments (day-aware)
        if (
          profile.fixed_commitments?.some(
            (c: any) =>
              c.days.includes(dayOfWeek) &&
              slotMins >= c.start_min &&
              slotMins < c.end_min,
          )
        ) {
          cell.isFixedCommitment = true;
        }

        // Hard exclusions (day-aware)
        if (
          profile.hard_exclusions?.some(
            (c: any) =>
              c.days.includes(dayOfWeek) &&
              slotMins >= c.start_min &&
              slotMins < c.end_min,
          )
        ) {
          cell.isHardExclusion = true;
        }

        // Availability = not blocked by any hard constraint
        if (cell.isSleep || cell.isFixedCommitment || cell.isHardExclusion) {
          cell.isAvailable = false;
        }
      }
    }

    // ── 2. Paint tasks from schedule blob ─────────────────────
    //    The schedule's dayOffset maps directly to the matrix column.
    //    We only paint dayOffset 0–6 (the visible 7-day window).
    const days: any[] = scheduleRes.days ?? [];

    for (const day of days) {
      const dayOffset: number = day.dayOffset;
      if (dayOffset < 0 || dayOffset > 6) continue; // outside visible window

      for (const section of day.sections ?? []) {
        for (const item of section.tasks ?? []) {
          if (item.startSlot == null) continue;

          const startSlot: number = item.startSlot;
          const slotsNeeded = Math.max(1, Math.ceil(item.completionTime / 15));

          for (let s = startSlot; s < startSlot + slotsNeeded && s < 96; s++) {
            const cell = matrix[dayOffset][s];
            cell.taskId = item.taskId;
            cell.taskName = item.taskName;
            cell.etask = item.axiomCost;
            cell.actualCompletion = false; // updated when task state changes
          }
        }
      }
    }

    return matrix;
  } catch (err) {
    console.error("getUserMatrix failed:", err);
    return { error: "Matrix generation failed: " + (err as Error).message };
  }
}
