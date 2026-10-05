export const FIXED_INTERVAL_BASELINE = {
  name: "Fixed-interval baseline stub",
  focusMinutes: 50,
  breakMinutes: 10,
};
export function fixedIntervalSessionCount(totalMinutes: number): number {
  const cycle =
    FIXED_INTERVAL_BASELINE.focusMinutes + FIXED_INTERVAL_BASELINE.breakMinutes;
  return totalMinutes <= 0 ? 0 : Math.ceil(totalMinutes / cycle);
}
