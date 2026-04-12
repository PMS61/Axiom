/* ═══════════════════════════════════════════════════════════
   Trend Engine — NewsAPI Fetcher
   Docs: https://newsapi.org/docs
   ═══════════════════════════════════════════════════════════ */

import type { RawSignal } from "./types";

const NEWS_API_BASE = "https://newsapi.org/v2";
const API_KEY = process.env.NEWS_API_KEY;

// Tech keywords to search. Each becomes a separate query batched in parallel.
const TECH_QUERIES = [
  "artificial intelligence machine learning",
  "web development React Next.js",
  "cloud computing AWS Kubernetes",
  "data science Python",
  "cybersecurity",
  "blockchain cryptocurrency",
  "mobile development",
  "programming software engineering",
  "startup technology",
];

interface NewsArticle {
  title: string;
  description: string | null;
  url: string;
  publishedAt: string;
  source: { name: string };
}

interface NewsApiResponse {
  status: string;
  totalResults: number;
  articles: NewsArticle[];
}

export async function fetchNewsSignals(query: string): Promise<RawSignal[]> {
  if (!API_KEY) {
    console.warn("NEWS_API_KEY not set — skipping NewsAPI fetch");
    return [];
  }

  const from = new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString().split("T")[0];

  const url = new URL(`${NEWS_API_BASE}/everything`);
  url.searchParams.set("q", query);
  url.searchParams.set("language", "en");
  url.searchParams.set("sortBy", "popularity");
  url.searchParams.set("pageSize", "20");
  url.searchParams.set("from", from);
  url.searchParams.set("apiKey", API_KEY);

  try {
    const res = await fetch(url.toString(), {
      headers: { "User-Agent": "AxiomTrendEngine/1.0" },
      next: { revalidate: 7200 }, // 2-hour Next.js cache
    });

    if (!res.ok) {
      console.error(`NewsAPI error ${res.status} for query: ${query}`);
      return [];
    }

    const data: NewsApiResponse = await res.json();
    if (data.status !== "ok") return [];

    return data.articles.map((a) => ({
      title: a.title,
      source: "news" as const,
      publishedAt: a.publishedAt,
      engagement: 0, // NewsAPI free tier has no engagement data
      url: a.url,
    }));
  } catch (err) {
    console.error("fetchNewsSignals failed:", err);
    return [];
  }
}

export async function fetchAllNewsSignals(): Promise<RawSignal[]> {
  const results = await Promise.allSettled(
    TECH_QUERIES.map((q) => fetchNewsSignals(q)),
  );

  return results
    .filter((r): r is PromiseFulfilledResult<RawSignal[]> => r.status === "fulfilled")
    .flatMap((r) => r.value);
}
