/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Energy / Axiom Model
   Deterministic axiom budget system with dynamic section weights.

   Constants:
     BASE_AXIOMS = 50  (daily budget)
     Section weights = [0.40, 0.35, 0.25]  (morning / afternoon / evening)

   Formulas:
     E_task = 0.6 * difficulty² + 0.8 * priority_num
     E_gain  = 0.25 * completionTime (minutes)
     new_weight = old_weight * (1 + α * (R − 1)), α = 0.2
   ═══════════════════════════════════════════════════════════ */

import type { SectionName, TimeSection } from "./types";

// ── Constants ─────────────────────────────────────────────

export const BASE_AXIOMS = 50;
export const FEEDBACK_ALPHA = 0.2;

/** Default fractional weights for each section — must sum to 1. */
const DEFAULT_WEIGHTS: Record<SectionName, number> = {
  morning: 0.40,
  afternoon: 0.35,
  evening: 0.25,
};

/** Hour ranges: [startHour, endHour) */
const SECTION_HOURS: Record<SectionName, [number, number]> = {
  morning: [6, 12],
  afternoon: [12, 18],
  evening: [18, 24],
};

const SECTION_LABELS: Record<SectionName, string> = {
  morning: "Morning",
  afternoon: "Afternoon",
  evening: "Evening",
};

// ── Section Builder ────────────────────────────────────────

/**
 * Builds the 3-section array with axiom budgets derived from weights.
 * Weights are accepted as a plain record; missing keys fall back to defaults.
 */
export function buildSections(
  weights: Partial<Record<SectionName, number>> = {},
): TimeSection[] {
  const merged: Record<SectionName, number> = {
    morning: weights.morning ?? DEFAULT_WEIGHTS.morning,
    afternoon: weights.afternoon ?? DEFAULT_WEIGHTS.afternoon,
    evening: weights.evening ?? DEFAULT_WEIGHTS.evening,
  };

  // Normalise so they always sum to 1
  const total = merged.morning + merged.afternoon + merged.evening;
  if (total <= 0) throw new Error("Section weights must be positive");
  const normalized: Record<SectionName, number> = {
    morning: merged.morning / total,
    afternoon: merged.afternoon / total,
    evening: merged.evening / total,
  };

  const names: SectionName[] = ["morning", "afternoon", "evening"];
  return names.map((name) => ({
    name,
    label: SECTION_LABELS[name],
    startHour: SECTION_HOURS[name][0],
    endHour: SECTION_HOURS[name][1],
    weight: normalized[name],
    axiomBudget: +(BASE_AXIOMS * normalized[name]).toFixed(2),
  }));
}

// ── Axiom Cost / Gain ──────────────────────────────────────

/**
 * E_task = 0.6 × difficulty² + 0.8 × priority_num
 * priority_num: high→9, normal→5, low→2
 */
export function computeAxiomCost(
  difficulty: number,
  priorityNum: number,
): number {
  return +(0.6 * Math.pow(difficulty, 2) + 0.8 * priorityNum).toFixed(4);
}

/**
 * E_gain = 0.25 × completionTime (minutes)
 * Used for recreational tasks that RESTORE axioms.
 */
export function computeAxiomGain(durationMinutes: number): number {
  return +(0.25 * durationMinutes).toFixed(4);
}

// ── Feedback Weight Update ─────────────────────────────────

/**
 * Updates a single section's weight via the feedback rule:
 *   new_weight = old_weight × (1 + α × (R − 1))
 * where R = efficiencyRatio (actual / scheduled axioms).
 * Result is the raw (un-normalised) updated weight.
 */
export function updatedRawWeight(
  oldWeight: number,
  efficiencyRatio: number,
): number {
  return oldWeight * (1 + FEEDBACK_ALPHA * (efficiencyRatio - 1));
}

/**
 * Applies feedback ratios to all three section weights and normalises
 * so they still sum to 1.
 *
 * @param currentWeights  — current weights for each section
 * @param ratios          — efficiency ratio per section (actual/scheduled)
 * @returns normalised updated weights
 */
export function updateSectionWeights(
  currentWeights: Record<SectionName, number>,
  ratios: Partial<Record<SectionName, number>>,
): Record<SectionName, number> {
  const names: SectionName[] = ["morning", "afternoon", "evening"];

  const raw: Record<SectionName, number> = {
    morning: updatedRawWeight(currentWeights.morning, ratios.morning ?? 1),
    afternoon: updatedRawWeight(currentWeights.afternoon, ratios.afternoon ?? 1),
    evening: updatedRawWeight(currentWeights.evening, ratios.evening ?? 1),
  };

  // Clamp each to [0.05, 0.90] to prevent degenerate schedules
  for (const n of names) {
    raw[n] = Math.min(0.90, Math.max(0.05, raw[n]));
  }

  const total = raw.morning + raw.afternoon + raw.evening;
  return {
    morning: +(raw.morning / total).toFixed(6),
    afternoon: +(raw.afternoon / total).toFixed(6),
    evening: +(raw.evening / total).toFixed(6),
  };
}

// ── Section Lookup ─────────────────────────────────────────

/** Returns which section a given hour (0-23) belongs to, or null if outside. */
export function sectionForHour(hour: number): SectionName | null {
  if (hour >= 6 && hour < 12) return "morning";
  if (hour >= 12 && hour < 18) return "afternoon";
  if (hour >= 18 && hour < 24) return "evening";
  return null;
}

/** Maps priority string to numeric value used in axiom formulas. */
export function priorityToNum(priority: "high" | "normal" | "low"): number {
  const map: Record<string, number> = { high: 9, normal: 5, low: 2 };
  return map[priority] ?? 5;
}
