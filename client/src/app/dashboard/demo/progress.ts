import { initialStudy } from "@/app/learn/scheduled-reviews";
import { study } from "./demo-data";

export const progressKey = "axiom-dashboard-concepts-online-v1";

export function readProgress() {
  const stored = localStorage.getItem(progressKey);
  if (!stored) return { state: initialStudy() };
  const data = JSON.parse(stored);
  const date = new Date(`${data.start}T12:00:00Z`);
  if (
    !Number.isFinite(date.getTime()) ||
    date.toISOString().slice(0, 10) !== data.start
  ) {
    return { state: initialStudy() };
  }
  return {
    state: study.restoreStudy(data.progress),
    start: data.start as string,
  };
}

export function localToday() {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Kolkata",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
}

export function epochDay(isoDate: string) {
  return Math.floor(Date.parse(`${isoDate}T00:00:00Z`) / 86400000);
}
