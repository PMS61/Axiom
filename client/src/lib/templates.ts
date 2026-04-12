/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Rule-Based Report Templates
   Deterministic string templates for daily and weekly analysis.
   ═══════════════════════════════════════════════════════════ */

export interface DailyReportData {
  adherencePercentage: number;
  totalEtask: number;
  highCLTasksPlacedInPeakCount: number;
  unresolvedConflictsCount: number;
  contextSwitchPenalty: number;
  burnoutRisk: "safe" | "watch" | "warning" | "critical";
  energyDeficit: number;
}

export interface WeeklyReportData {
  compositeScore: number;
  deadlineHitRate: number;
  averageCLBalance: number;
  maxCLSpikeDay: string;
  maxCLSpikeValue: number;
  lowEnergyDaysCount: number;
  recommendedFocusShift: string;
}

/**
 * Generates an insight trace log for the Daily Report.
 * Uses strict rule-based phrasing suitable for the Axiom aesthetic.
 */
export function generateDailyInsight(data: DailyReportData): string {
  const templates: string[] = [];

  // 1. Adherence Evaluation
  if (data.adherencePercentage >= 90) {
    templates.push(`Schedule execution optimal (${data.adherencePercentage}% adherence).`);
  } else if (data.adherencePercentage >= 70) {
    templates.push(`Execution variance detected (${data.adherencePercentage}% adherence).`);
  } else {
    templates.push(`Severe schedule deviation (${data.adherencePercentage}% adherence).`);
  }

  // 2. High Etask & Peak Bandwidth Mapping
  if (data.highCLTasksPlacedInPeakCount > 0) {
    templates.push(`Successfully mapped ${data.highCLTasksPlacedInPeakCount} high-Etask tasks to peak bandwidth windows.`);
  } else if (data.totalEtask > 15) {
    templates.push(`Warning: High cumulative Etask (${data.totalEtask.toFixed(1)}) without alignment to peak energy windows.`);
  }

  // 3. Penalty & Conflicts
  if (data.contextSwitchPenalty > 5) {
    templates.push(`High context-switching penalty logged (${data.contextSwitchPenalty.toFixed(1)}). Group similar workspace topologies.`);
  }
  if (data.unresolvedConflictsCount > 0) {
    templates.push(`[!] ${data.unresolvedConflictsCount} scheduling paths remain unresolved. Manual intervention required.`);
  }

  // 4. Burnout State
  if (data.burnoutRisk === "critical" || data.energyDeficit > 10) {
    templates.push(`CRITICAL: Energy deficit (${data.energyDeficit}). System enforcing hard constraint on deep work for T+24h.`);
  }

  return templates.join(" ");
}

/**
 * Generates an insight trace log for the Weekly Report.
 */
export function generateWeeklyInsight(data: WeeklyReportData): string {
  const templates: string[] = [];

  // 1. Composite & Deadlines
  templates.push(`7-Day Vector finalized with Composite Score [${data.compositeScore}].`);
  
  if (data.deadlineHitRate < 80) {
    templates.push(`Deadline hit rate at ${data.deadlineHitRate}%. Recommend redistributing constraints or sacrificing task depth.`);
  } else {
    templates.push(`Deadline constraints resolved (${data.deadlineHitRate}% hit rate).`);
  }

  // 2. Volatility Tracking
  if (data.maxCLSpikeValue > 35) {
    templates.push(`Spike volatility detected: ${data.maxCLSpikeDay} Etask exceeded threshold (${data.maxCLSpikeValue}).`);
    templates.push(`Recommendation: Distribute deep work toward ${data.recommendedFocusShift}.`);
  } else {
    templates.push(`Etask variance remained within stable systemic thresholds (Avg: ${data.averageCLBalance.toFixed(1)}).`);
  }

  // 3. Systemic Burnout Prevention
  if (data.lowEnergyDaysCount >= 3) {
    templates.push(`[!] Slump patterns observed on ${data.lowEnergyDaysCount} days. Re-calibrate bandwidth array parameters.`);
  }

  return templates.join(" ");
}
