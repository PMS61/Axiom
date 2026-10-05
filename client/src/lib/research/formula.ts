export const FORMULA_VERSION = "m1-stub-v0";
export const FORMULA_NOTICE =
  "Provisional load proxy; M1's research formula is not defined yet.";
export function estimateCognitiveLoadProxy(
  difficulty: number,
  priority: "high" | "normal" | "low",
): number {
  const priorityValue = { high: 9, normal: 5, low: 2 }[priority];
  return Number((0.6 * difficulty ** 2 + 0.8 * priorityValue).toFixed(2));
}
