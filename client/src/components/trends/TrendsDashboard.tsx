"use client";

import { useState, useTransition } from "react";
import { addTopicToRoadmap, fetchTopTrends } from "@/app/actions/trends";
import Header from "@/components/Header";
import type { RoadmapPayload, TrendResult } from "@/lib/trend-engine/types";
import TopicSearch from "./TopicSearch";
import TrendCard from "./TrendCard";

interface TrendsDashboardProps {
  initialTrends: TrendResult[];
  cached: boolean;
  fetchedAt?: string;
}

export default function TrendsDashboard({
  initialTrends,
  cached,
  fetchedAt,
}: TrendsDashboardProps) {
  const [trends, setTrends] = useState<TrendResult[]>(initialTrends);
  const [isCached, setIsCached] = useState(cached);
  const [roadmapQueue, setRoadmapQueue] = useState<RoadmapPayload[]>([]);
  const [notification, setNotification] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  function showNotification(msg: string) {
    setNotification(msg);
    setTimeout(() => setNotification(null), 4000);
  }

  async function handleAddToRoadmap(topic: string, trendScore: number) {
    const res = await addTopicToRoadmap(topic, trendScore);
    if (res.payload) {
      setRoadmapQueue((q) => {
        const exists = q.find((p) => p.topic === res.payload?.topic);
        if (exists) return q;
        return [...q, res.payload!];
      });
      showNotification(
        `"${topic}" queued — priority ${res.payload.priority}/10, depth: ${res.payload.recommended_depth}`,
      );
    }
  }

  function handleRefresh() {
    startTransition(async () => {
      const res = await fetchTopTrends(true);
      if (res.trends) {
        setTrends(res.trends);
        setIsCached(false);
        showNotification("Signals refreshed.");
      }
    });
  }

  // Stats
  const risingCount = trends.filter((t) => t.direction === "rising").length;
  const decliningCount = trends.filter(
    (t) => t.direction === "declining",
  ).length;
  const stableCount = trends.filter((t) => t.direction === "stable").length;
  const avgScore =
    trends.length > 0
      ? Math.round(
          trends.reduce((s, t) => s + t.trend_score, 0) / trends.length,
        )
      : 0;
  const topTrend = trends[0] ?? null;

  return (
    <div style={{ minHeight: "100vh" }}>
      <Header />

      {/* Notification toast */}
      {notification && (
        <div
          style={{
            position: "fixed",
            top: 16,
            right: 16,
            zIndex: 99999,
            background: "var(--card-bg)",
            border: "0.5px solid var(--rule)",
            padding: "12px 16px",
            fontSize: 12,
            fontFamily: "var(--mono)",
            maxWidth: 340,
            boxShadow: "0 4px 24px rgba(0,0,0,0.08)",
          }}
        >
          {notification}
        </div>
      )}

      {/* ── Hero Section ── */}
      <section
        className="container section-rule"
        style={{ paddingTop: 60, paddingBottom: 40 }}
      >
        <div
          className="responsive-grid-hero"
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1.5fr",
            gap: 60,
            alignItems: "start",
          }}
        >
          <div>
            <div className="meta-text" style={{ marginBottom: 16 }}>
              TREND INTELLIGENCE
            </div>
            <h1 style={{ fontSize: 42, marginBottom: 24 }}>
              What to Learn Next
            </h1>
            <p
              style={{ color: "var(--muted)", maxWidth: 380, marginBottom: 24 }}
            >
              {trends.length} topics tracked · live signals from News, Reddit
              &amp; Hacker News.
            </p>

            <div
              className="meta-text responsive-grid-stats"
              style={{
                borderTop: "0.5px solid var(--rule)",
                paddingTop: 16,
                display: "grid",
                gridTemplateColumns: "1fr 1fr 1fr 1fr",
                gap: 16,
              }}
            >
              <div>
                <div style={{ fontSize: 10, color: "var(--muted)" }}>
                  Rising
                </div>
                <div
                  style={{
                    fontWeight: 700,
                    fontSize: 20,
                    marginTop: 4,
                    color: "var(--safe)",
                  }}
                >
                  {risingCount}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 10, color: "var(--muted)" }}>
                  Stable
                </div>
                <div
                  style={{
                    fontWeight: 700,
                    fontSize: 20,
                    marginTop: 4,
                    color: "var(--ink)",
                  }}
                >
                  {stableCount}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 10, color: "var(--muted)" }}>
                  Declining
                </div>
                <div
                  style={{
                    fontWeight: 700,
                    fontSize: 20,
                    marginTop: 4,
                    color: "var(--vermillion)",
                  }}
                >
                  {decliningCount}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 10, color: "var(--muted)" }}>
                  Avg Score
                </div>
                <div
                  style={{
                    fontWeight: 700,
                    fontSize: 20,
                    marginTop: 4,
                    color: "var(--ink)",
                  }}
                >
                  {avgScore}
                </div>
              </div>
            </div>
          </div>

          {/* Top trending topic callout */}
          <div>
            <div
              className="meta-text"
              style={{
                marginBottom: 12,
                display: "flex",
                justifyContent: "space-between",
              }}
            >
              <span>Top Signal Now</span>
              <span style={{ color: "var(--vermillion)" }}>Trend Score</span>
            </div>
            {topTrend ? (
              <div
                style={{
                  background: "var(--card-bg)",
                  border: "0.5px solid var(--rule)",
                  borderLeft: `4px solid ${
                    topTrend.direction === "rising"
                      ? "var(--safe)"
                      : topTrend.direction === "declining"
                        ? "var(--vermillion)"
                        : "var(--ink)"
                  }`,
                  padding: "24px",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 12,
                  }}
                >
                  <h3 style={{ margin: 0, fontSize: 18, fontWeight: 700 }}>
                    {topTrend.topic}
                  </h3>
                  <span
                    style={{
                      fontWeight: 700,
                      fontSize: 24,
                      fontFamily: "var(--mono)",
                    }}
                  >
                    {topTrend.trend_score}
                  </span>
                </div>
                <p
                  style={{
                    color: "var(--muted)",
                    fontSize: 12,
                    marginBottom: 16,
                  }}
                >
                  {topTrend.insight}
                </p>
                <div className="meta-text" style={{ display: "flex", gap: 16 }}>
                  <span>{topTrend.signals.news_mentions} news</span>
                  <span>· {topTrend.signals.reddit_posts} reddit</span>
                  <span>· {topTrend.signals.hackernews_posts} HN</span>
                </div>
              </div>
            ) : (
              <div
                style={{
                  border: "0.5px dashed var(--muted)",
                  padding: "32px 24px",
                  textAlign: "center",
                }}
              >
                <p className="meta-text">No trends loaded.</p>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* ── Controls ── */}
      <section
        className="container section-rule"
        style={{
          padding: "16px 24px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 12,
        }}
      >
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          <button
            type="button"
            className="btn btn-sm btn-primary"
            onClick={handleRefresh}
            disabled={isPending}
            style={{
              opacity: isPending ? 0.6 : 1,
              cursor: isPending ? "not-allowed" : "pointer",
            }}
          >
            {isPending ? "Refreshing…" : "↺ Refresh Signals"}
          </button>
          {isCached && (
            <span className="meta-text" style={{ color: "var(--muted)" }}>
              Cached results
            </span>
          )}
        </div>

        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          {roadmapQueue.length > 0 && (
            <div
              style={{
                display: "flex",
                gap: 8,
                alignItems: "center",
                flexWrap: "wrap",
              }}
            >
              <span className="meta-text">
                Roadmap Queue ({roadmapQueue.length}):
              </span>
              {roadmapQueue.map((p) => (
                <div
                  key={p.topic}
                  style={{
                    fontFamily: "var(--mono)",
                    fontSize: 10,
                    border: "0.5px solid var(--rule)",
                    padding: "2px 8px",
                    display: "inline-flex",
                    gap: 8,
                    background: "var(--card-bg)",
                  }}
                >
                  <span>{p.topic}</span>
                  <span style={{ color: "var(--muted)" }}>P{p.priority}</span>
                </div>
              ))}
            </div>
          )}
          {fetchedAt && (
            <span className="meta-text" style={{ color: "var(--muted)" }}>
              {new Date(fetchedAt).toLocaleTimeString()}
            </span>
          )}
        </div>
      </section>

      {/* ── Main Content ── */}
      <section
        className="container"
        style={{ paddingTop: 40, paddingBottom: 80 }}
      >
        {/* Trending Topics Table */}
        <div style={{ marginBottom: 60 }}>
          <div
            style={{
              display: "flex",
              alignItems: "baseline",
              gap: 16,
              marginBottom: 24,
              borderBottom: "0.5px solid var(--rule)",
              paddingBottom: 8,
            }}
          >
            <h2 style={{ fontSize: 18 }}>
              Top {trends.length} Trending Topics
            </h2>
            <span className="meta-text">Last 48h · All sources</span>
          </div>

          {/* Table header */}
          {trends.length > 0 && (
            <div
              className="meta-text"
              style={{
                display: "grid",
                gridTemplateColumns: "40px 1fr 80px 80px 80px 60px",
                gap: 12,
                padding: "8px 16px",
                borderBottom: "0.5px solid var(--ink)",
                marginBottom: 4,
              }}
            >
              <span>#</span>
              <span>Topic</span>
              <span style={{ textAlign: "right" }}>News</span>
              <span style={{ textAlign: "right" }}>Reddit</span>
              <span style={{ textAlign: "right" }}>HN</span>
              <span style={{ textAlign: "right" }}>Score</span>
            </div>
          )}

          {trends.length === 0 ? (
            <div
              style={{
                padding: "40px 0",
                textAlign: "center",
                color: "var(--muted)",
              }}
            >
              No trends loaded. Check API keys and try refreshing.
            </div>
          ) : (
            <div>
              {trends.map((trend, i) => (
                <TrendCard
                  key={trend.topic}
                  trend={trend}
                  rank={i + 1}
                  onAddToRoadmap={handleAddToRoadmap}
                />
              ))}
            </div>
          )}
        </div>

        {/* Topic Search */}
        <div>
          <div
            style={{
              borderBottom: "0.5px solid var(--rule)",
              paddingBottom: 8,
              marginBottom: 24,
            }}
          >
            <h2 style={{ fontSize: 18 }}>Topic Search</h2>
            <p className="meta-text" style={{ marginTop: 4 }}>
              Ask anything — get relevance score, ROI analysis, and a learning
              recommendation.
            </p>
          </div>
          <TopicSearch />
        </div>

        {/* Attribution */}
        <div
          className="meta-text"
          style={{
            display: "flex",
            gap: 24,
            paddingTop: 24,
            borderTop: "0.5px solid var(--rule)",
            marginTop: 40,
            flexWrap: "wrap",
          }}
        >
          <span>Sources: NewsAPI</span>
          <span>Reddit</span>
          <span>Hacker News</span>
          <span>Insights: Gemma 4</span>
        </div>
      </section>
    </div>
  );
}
