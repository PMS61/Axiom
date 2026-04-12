/* ═══════════════════════════════════════════════════════════
   Trend Engine — Hacker News Fetcher
   Uses the public Firebase HN API — no key required.
   https://github.com/HackerNews/API
   ═══════════════════════════════════════════════════════════ */

import type { RawSignal } from "./types";

const HN_BASE = "https://hacker-news.firebaseio.com/v0";

interface HNItem {
  id: number;
  title: string;
  url?: string;
  score: number;
  descendants?: number; // comment count
  time: number; // unix timestamp
  type: string;
}

async function fetchHNItem(id: number): Promise<HNItem | null> {
  try {
    const res = await fetch(`${HN_BASE}/item/${id}.json`, {
      next: { revalidate: 3600 },
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function fetchHackerNewsSignals(): Promise<RawSignal[]> {
  try {
    // Get top 100 story IDs
    const res = await fetch(`${HN_BASE}/topstories.json`, {
      next: { revalidate: 3600 },
    });
    if (!res.ok) return [];

    const ids: number[] = await res.json();
    const top50 = ids.slice(0, 50);

    // Fetch items in batches of 10
    const signals: RawSignal[] = [];
    const cutoff = Date.now() / 1000 - 48 * 3600;

    const batches = [];
    for (let i = 0; i < top50.length; i += 10) {
      batches.push(top50.slice(i, i + 10));
    }

    for (const batch of batches) {
      const items = await Promise.all(batch.map(fetchHNItem));
      for (const item of items) {
        if (!item || item.type !== "story" || item.time < cutoff) continue;
        if (!item.title) continue;

        signals.push({
          title: item.title,
          source: "hackernews",
          publishedAt: new Date(item.time * 1000).toISOString(),
          engagement: item.score + (item.descendants ?? 0),
          url: item.url ?? `https://news.ycombinator.com/item?id=${item.id}`,
        });
      }
    }

    return signals;
  } catch (err) {
    console.error("fetchHackerNewsSignals failed:", err);
    return [];
  }
}
