export const variants = [
  "full",
  "no_practice",
  "no_hierarchy",
  "no_dynamics",
] as const;
export const scenarios = [
  "full",
  "no_practice",
  "no_hierarchy",
  "no_dynamics",
  "short_gaps",
] as const;
export type RunConfig = {
  source: "synthetic" | "upload";
  scenarios: string[];
  variants: string[];
  replicates: number;
  chains: number;
  iterations: number;
  warmup: number;
  seed: number;
  users: number;
  items: number;
  events: number;
  holdout: number;
  workers: number;
  sequences?: unknown;
};
export const defaultConfig: RunConfig = {
  source: "synthetic",
  scenarios: ["full"],
  variants: [...variants],
  replicates: 1,
  chains: 4,
  iterations: 2000,
  warmup: 1000,
  seed: 20261027,
  users: 4,
  items: 5,
  events: 32,
  holdout: 0.25,
  workers: 2,
};
export type RunStatus = {
  state: "queued" | "running" | "complete" | "failed" | "cancelled";
  completed?: number;
  total?: number;
  message: string;
  elapsed_seconds?: number;
};
export type RunListItem = { id: string; status: RunStatus; created: string };
export type Diagnostic = {
  scenario: string;
  replicate: number;
  variant: string;
  parameter: string;
  mean: number;
  sd: number;
  low: number;
  high: number;
  rhat: number;
  ess_bulk: number;
  ess_tail: number;
  mcse_mean: number;
  ess_per_cpu_second: number;
  passed: boolean;
};
export type ResultRow = {
  scenario: string;
  replicate: number;
  variant: string;
  log_loss: number;
  brier: number;
  mae: number;
  auc: number | null;
  calibration_bias: number;
  test_events: number;
  max_rhat: number;
  min_ess_bulk: number;
  min_ess_tail: number;
  passed: boolean;
  failed_parameters: number;
  fit_cpu_seconds: number;
  sum_chain_wall_seconds: number;
  prediction_cpu_seconds: number;
};
export type RecoveryRow = {
  scenario: string;
  replicate: number;
  variant: string;
  parameter: string;
  truth: number;
  mean: number;
  low: number;
  high: number;
  error: number;
  covered: boolean;
  passed: boolean;
  matched: boolean;
};
export type Comparison = {
  scenario: string;
  replicate: number;
  variant: string;
  delta_log_loss: number;
  low: number;
  high: number;
  clusters: number;
  converged: boolean;
};
export type Summary = {
  config: RunConfig;
  results: ResultRow[];
  diagnostics: Diagnostic[];
  recovery?: RecoveryRow[];
  comparisons?: Comparison[];
  wall_seconds?: number;
};
export type RunDetail = {
  id: string;
  status: RunStatus;
  summary: Summary | null;
  artifacts: string[];
  config: RunConfig | null;
};
export const humanize = (name: string) => name.replaceAll("_", " ");
