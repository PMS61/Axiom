/* ═══════════════════════════════════════════════════════════
   Trend Engine — Scoring
   Pure math. No I/O. No LLM.

   trend_score = frequency * 0.4 + engagement * 0.4 + recency * 0.2

   All inputs are normalized to 0–1 before combining.
   Final score is scaled to 0–100.
   ═══════════════════════════════════════════════════════════ */

import type { RawSignal, TopicSignals } from "./types";

const WEIGHT_FREQUENCY  = 0.4;
const WEIGHT_ENGAGEMENT = 0.4;
const WEIGHT_RECENCY    = 0.2;

// Half-life for recency: 24 hours
const RECENCY_HALF_LIFE_MS = 24 * 60 * 60 * 1000;

function recencyScore(publishedAt: string): number {
  const age = Date.now() - new Date(publishedAt).getTime();
  // Exponential decay: score = e^(-age / half_life)
  return Math.exp(-age / RECENCY_HALF_LIFE_MS);
}

/** Normalize an array to [0, 1] using min-max scaling */
function normalizeArray(values: number[]): number[] {
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const range = max - min;
  if (range === 0) return values.map(() => 0.5);
  return values.map((v) => (v - min) / range);
}

/** Group raw signals by topic and compute per-topic aggregates */
export function aggregateByTopic(
  signals: Array<{ signal: RawSignal; topic: string }>,
): Record<string, TopicSignals> {
  const map: Record<string, TopicSignals> = {};

  for (const { signal, topic } of signals) {
    if (!map[topic]) {
      map[topic] = {
        topic,
        news_mentions: 0,
        reddit_posts: 0,
        hackernews_posts: 0,
        avg_engagement: 0,
        total_engagement: 0,
        recency_score: 0,
        raw: [],
      };
    }

    const entry = map[topic];
    entry.raw.push(signal);

    if (signal.source === "news") entry.news_mentions++;
    if (signal.source === "reddit") entry.reddit_posts++;
    if (signal.source === "hackernews") entry.hackernews_posts++;

    entry.total_engagement += signal.engagement;
    entry.recency_score = Math.max(entry.recency_score, recencyScore(signal.publishedAt));
  }

  // Compute avg_engagement
  for (const topic of Object.keys(map)) {
    const entry = map[topic];
    entry.avg_engagement = entry.total_engagement / entry.raw.length;
  }

  return map;
}

/** Score all topics and return sorted by trend_score descending */
export function scoreTopics(
  topicMap: Record<string, TopicSignals>,
): Array<TopicSignals & { trend_score: number }> {
  const topics = Object.values(topicMap);
  if (topics.length === 0) return [];

  const mentionCounts = topics.map(
    (t) => t.news_mentions + t.reddit_posts + t.hackernews_posts,
  );
  const engagements = topics.map((t) => t.avg_engagement);
  const recencies = topics.map((t) => t.recency_score);

  const normMentions = normalizeArray(mentionCounts);
  const normEngagements = normalizeArray(engagements);
  const normRecencies = normalizeArray(recencies);

  const scored = topics.map((topic, i) => {
    const rawScore =
      normMentions[i] * WEIGHT_FREQUENCY +
      normEngagements[i] * WEIGHT_ENGAGEMENT +
      normRecencies[i] * WEIGHT_RECENCY;

    return { ...topic, trend_score: Math.round(rawScore * 100) };
  });

  return scored.sort((a, b) => b.trend_score - a.trend_score);
}

/**
 * Determine trend direction by comparing last 7 signals vs previous 7 signals.
 * Returns "rising" | "stable" | "declining".
 */
export function inferDirection(raw: RawSignal[]): "rising" | "stable" | "declining" {
  if (raw.length < 4) return "stable";

  const sorted = [...raw].sort(
    (a, b) => new Date(a.publishedAt).getTime() - new Date(b.publishedAt).getTime(),
  );

  const half = Math.floor(sorted.length / 2);
  const older = sorted.slice(0, half);
  const newer = sorted.slice(half);

  const olderAvgEng = older.reduce((s, r) => s + r.engagement, 0) / older.length;
  const newerAvgEng = newer.reduce((s, r) => s + r.engagement, 0) / newer.length;

  const ratio = olderAvgEng === 0 ? 2 : newerAvgEng / olderAvgEng;

  if (ratio > 1.2) return "rising";
  if (ratio < 0.8) return "declining";
  return "stable";
}

export function inferMomentum(raw: RawSignal[]): "accelerating" | "steady" | "slowing" {
  if (raw.length < 6) return "steady";

  const sorted = [...raw].sort(
    (a, b) => new Date(a.publishedAt).getTime() - new Date(b.publishedAt).getTime(),
  );

  const thirds = Math.floor(sorted.length / 3);
  const early = sorted.slice(0, thirds);
  const mid = sorted.slice(thirds, thirds * 2);
  const late = sorted.slice(thirds * 2);

  const avg = (arr: RawSignal[]) =>
    arr.length ? arr.reduce((s, r) => s + r.engagement, 0) / arr.length : 0;

  const trend1 = avg(mid) - avg(early);
  const trend2 = avg(late) - avg(mid);

  if (trend2 > trend1 + 5) return "accelerating";
  if (trend2 < trend1 - 5) return "slowing";
  return "steady";
}

/**
 * Build a 7-point time series from raw signals.
 * Divides the last 48h into 7 bins (~7h each).
 */
export function buildTimeSeries(
  raw: RawSignal[],
  metric: "mentions" | "engagement",
): number[] {
  const now = Date.now();
  const windowMs = 48 * 60 * 60 * 1000;
  const binMs = windowMs / 7;
  const bins = Array(7).fill(0) as number[];

  for (const signal of raw) {
    const age = now - new Date(signal.publishedAt).getTime();
    if (age > windowMs) continue;
    const bin = Math.min(6, Math.floor(age / binMs));
    const idx = 6 - bin; // reverse so index 0 = oldest
    bins[idx] += metric === "mentions" ? 1 : signal.engagement;
  }

  return bins;
}
