/* ═══════════════════════════════════════════════════════════
   Trend Engine — Reddit Fetcher
   Uses Reddit public JSON API (no OAuth needed for read-only).
   If REDDIT_CLIENT_ID + REDDIT_SECRET are set, uses OAuth for
   higher rate limits (60 req/min vs 10 req/min).
   ═══════════════════════════════════════════════════════════ */

import type { RawSignal } from "./types";

const SUBREDDITS = [
  "programming",
  "webdev",
  "MachineLearning",
  "learnprogramming",
  "cscareerquestions",
  "startups",
  "technology",
  "artificial",
  "datascience",
  "devops",
];

interface RedditPost {
  data: {
    title: string;
    url: string;
    score: number;
    num_comments: number;
    created_utc: number;
    selftext: string;
  };
}

interface RedditListing {
  data: {
    children: RedditPost[];
  };
}

// Get Reddit OAuth token using client credentials (app-only auth)
let redditToken: { value: string; expiresAt: number } | null = null;

async function getRedditToken(): Promise<string | null> {
  const clientId = process.env.REDDIT_CLIENT_ID;
  const secret = process.env.REDDIT_SECRET;
  if (!clientId || !secret) return null;

  if (redditToken && Date.now() < redditToken.expiresAt) return redditToken.value;

  try {
    const credentials = Buffer.from(`${clientId}:${secret}`).toString("base64");
    const res = await fetch("https://www.reddit.com/api/v1/access_token", {
      method: "POST",
      headers: {
        Authorization: `Basic ${credentials}`,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "AxiomTrendEngine/1.0 by axiom_app",
      },
      body: "grant_type=client_credentials",
    });

    if (!res.ok) return null;
    const data = await res.json();
    redditToken = {
      value: data.access_token,
      expiresAt: Date.now() + (data.expires_in - 60) * 1000,
    };
    return redditToken.value;
  } catch {
    return null;
  }
}

async function fetchSubreddit(subreddit: string): Promise<RawSignal[]> {
  const token = await getRedditToken();
  const baseUrl = token
    ? `https://oauth.reddit.com/r/${subreddit}/hot`
    : `https://www.reddit.com/r/${subreddit}/hot.json`;

  const headers: Record<string, string> = {
    "User-Agent": "AxiomTrendEngine/1.0 by axiom_app",
  };
  if (token) headers.Authorization = `Bearer ${token}`;

  const url = `${baseUrl}?limit=25${token ? "" : "&raw_json=1"}`;

  try {
    const res = await fetch(url, {
      headers,
      next: { revalidate: 3600 }, // 1-hour Next.js cache per subreddit
    });

    if (res.status === 429) {
      console.warn(`Reddit rate limit on r/${subreddit}`);
      return [];
    }
    if (!res.ok) {
      console.error(`Reddit fetch failed for r/${subreddit}: ${res.status}`);
      return [];
    }

    const data: RedditListing = await res.json();
    const cutoff = Date.now() / 1000 - 48 * 3600; // last 48h

    return data.data.children
      .filter((p) => p.data.created_utc > cutoff)
      .map((p) => ({
        title: p.data.title,
        source: "reddit" as const,
        publishedAt: new Date(p.data.created_utc * 1000).toISOString(),
        engagement: p.data.score + p.data.num_comments * 2,
        url: `https://reddit.com${p.data.url}`,
      }));
  } catch (err) {
    console.error(`fetchSubreddit r/${subreddit} error:`, err);
    return [];
  }
}

export async function fetchAllRedditSignals(): Promise<RawSignal[]> {
  const results = await Promise.allSettled(
    SUBREDDITS.map((s) => fetchSubreddit(s)),
  );

  return results
    .filter((r): r is PromiseFulfilledResult<RawSignal[]> => r.status === "fulfilled")
    .flatMap((r) => r.value);
}
