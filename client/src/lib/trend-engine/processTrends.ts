/* ═══════════════════════════════════════════════════════════
   Trend Engine — Orchestrator
   Fetches all signals → maps to topics → scores → returns top N.
   No LLM calls here. LLM insights are generated separately in
   the Server Action (so we can cache raw trends independently).
   ═══════════════════════════════════════════════════════════ */

import { fetchAllNewsSignals } from "./fetchNews";
import { fetchAllRedditSignals } from "./fetchReddit";
import { fetchHackerNewsSignals } from "./fetchHackerNews";
import { mapTitleToTopic } from "./topicMapper";
import {
  aggregateByTopic,
  scoreTopics,
  inferDirection,
  inferMomentum,
  buildTimeSeries,
} from "./scoring";
import type { RawSignal, TrendResult } from "./types";

interface ProcessedTrends {
  trends: Omit<TrendResult, "insight" | "future_outlook">[];
  fetchedAt: string;
}

export async function processTrends(topN = 10): Promise<ProcessedTrends> {
  // Fetch all sources in parallel
  const [newsSignals, redditSignals, hnSignals] = await Promise.all([
    fetchAllNewsSignals(),
    fetchAllRedditSignals(),
    fetchHackerNewsSignals(),
  ]);

  const allSignals: RawSignal[] = [...newsSignals, ...redditSignals, ...hnSignals];

  // Map each signal to a topic
  const mapped: Array<{ signal: RawSignal; topic: string }> = [];
  for (const signal of allSignals) {
    const topic = mapTitleToTopic(signal.title);
    if (topic) mapped.push({ signal, topic });
  }

  // Aggregate and score
  const topicMap = aggregateByTopic(mapped);
  const scored = scoreTopics(topicMap).slice(0, topN);

  const trends = scored.map((t) => ({
    topic: t.topic,
    trend_score: t.trend_score,
    direction: inferDirection(t.raw),
    momentum: inferMomentum(t.raw),
    signals: {
      news_mentions: t.news_mentions,
      reddit_posts: t.reddit_posts,
      hackernews_posts: t.hackernews_posts,
      avg_engagement: Math.round(t.avg_engagement),
    },
    mentions_over_time: buildTimeSeries(t.raw, "mentions"),
    engagement_over_time: buildTimeSeries(t.raw, "engagement"),
  }));

  return { trends, fetchedAt: new Date().toISOString() };
}

/** Process trends for a specific query topic */
export async function processSingleTopic(
  query: string,
): Promise<Omit<TrendResult, "insight" | "future_outlook"> | null> {
  const [newsSignals, redditSignals, hnSignals] = await Promise.all([
    fetchAllNewsSignals(),
    fetchAllRedditSignals(),
    fetchHackerNewsSignals(),
  ]);

  const allSignals = [...newsSignals, ...redditSignals, ...hnSignals];

  // Filter signals relevant to this query
  const q = query.toLowerCase();
  const relevant = allSignals.filter((s) =>
    s.title.toLowerCase().includes(q),
  );

  if (relevant.length === 0) return null;

  const mapped = relevant.map((s) => ({ signal: s, topic: query }));
  const topicMap = aggregateByTopic(mapped);
  const scored = scoreTopics(topicMap);

  if (scored.length === 0) return null;
  const t = scored[0];

  return {
    topic: query,
    trend_score: t.trend_score,
    direction: inferDirection(t.raw),
    momentum: inferMomentum(t.raw),
    signals: {
      news_mentions: t.news_mentions,
      reddit_posts: t.reddit_posts,
      hackernews_posts: t.hackernews_posts,
      avg_engagement: Math.round(t.avg_engagement),
    },
    mentions_over_time: buildTimeSeries(t.raw, "mentions"),
    engagement_over_time: buildTimeSeries(t.raw, "engagement"),
  };
}
