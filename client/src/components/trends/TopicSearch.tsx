"use client";

import { useState } from "react";
import { searchTopic } from "@/app/actions/trends";
import type { TopicSearchResult } from "@/lib/trend-engine/types";

const DEPTH_LABELS: Record<string, string> = {
  overview: "Overview",
  working_knowledge: "Working Knowledge",
  proficient: "Proficient",
  expert: "Expert",
};

const ROI_COLORS: Record<string, string> = {
  low: "var(--vermillion)",
  medium: "var(--watch)",
  high: "var(--safe)",
  very_high: "var(--safe)",
};

const EFFORT_COLORS: Record<string, string> = {
  low: "var(--safe)",
  medium: "var(--watch)",
  high: "var(--watch)",
  very_high: "var(--vermillion)",
};

export default function TopicSearch() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<TopicSearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setResult(null);
    setError(null);

    const res = await searchTopic(query);
    setLoading(false);

    if (res.error) {
      setError(res.error);
    } else if (res.result) {
      setResult(res.result);
    }
  }

  return (
    <div style={{ maxWidth: 680 }}>
      <form onSubmit={handleSearch} style={{ display: "flex", gap: 8, marginBottom: 20 }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder='e.g. "Should I learn Rust?" or "web development"'
          style={{
            flex: 1,
            fontFamily: "var(--mono)",
            fontSize: 13,
            color: "var(--ink)",
            background: "transparent",
            border: "none",
            borderBottom: "0.5px solid var(--rule)",
            padding: "8px 0",
            outline: "none",
          }}
          onFocus={(e) => (e.currentTarget.style.borderBottomColor = "var(--ink)")}
          onBlur={(e) => (e.currentTarget.style.borderBottomColor = "var(--rule)")}
        />
        <button
          type="submit"
          className="btn btn-sm btn-primary"
          disabled={loading || !query.trim()}
          style={{ opacity: loading || !query.trim() ? 0.5 : 1, whiteSpace: "nowrap" }}
        >
          {loading ? "Analyzing…" : "Analyze →"}
        </button>
      </form>

      {error && (
        <div
          style={{
            padding: "12px 16px",
            border: "0.5px solid var(--vermillion)",
            marginBottom: 16,
          }}
        >
          <p className="meta-text" style={{ color: "var(--vermillion)" }}>
            [!] {error}
          </p>
        </div>
      )}

      {loading && (
        <div
          style={{
            border: "0.5px solid var(--rule)",
            padding: "24px",
            background: "var(--card-bg)",
          }}
        >
          <div className="meta-text" style={{ color: "var(--muted)" }}>
            Analyzing…
          </div>
          <div
            style={{
              marginTop: 12,
              height: 2,
              background: "var(--rule)",
              overflow: "hidden",
              position: "relative",
            }}
          >
            <div
              style={{
                position: "absolute",
                height: "100%",
                width: "40%",
                background: "var(--ink)",
                animation: "slide 1.2s ease-in-out infinite",
              }}
            />
          </div>
        </div>
      )}

      {result && !loading && (
        <div
          style={{
            border: "0.5px solid var(--rule)",
            background: "var(--card-bg)",
            overflow: "hidden",
          }}
        >
          {/* Header */}
          <div
            style={{
              padding: "16px 20px",
              borderBottom: "0.5px solid var(--rule)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
            }}
          >
            <div>
              <h3 style={{ fontSize: 16, margin: 0, fontWeight: 700 }}>{result.topic}</h3>
              <p className="meta-text" style={{ marginTop: 4 }}>
                Learning depth: {DEPTH_LABELS[result.suggested_depth] ?? result.suggested_depth}
              </p>
            </div>
            <div style={{ textAlign: "right" }}>
              <span
                style={{ fontFamily: "var(--mono)", fontSize: 28, fontWeight: 700 }}
              >
                {result.relevance_score}
              </span>
              <p className="meta-text" style={{ color: "var(--muted)" }}>relevance</p>
            </div>
          </div>

          <div style={{ padding: "16px 20px" }}>
            {/* Metrics */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr 1fr",
                gap: 12,
                marginBottom: 16,
              }}
            >
              {[
                {
                  label: "ROI",
                  value: result.roi_estimate.replace("_", " "),
                  color: ROI_COLORS[result.roi_estimate] ?? "var(--ink)",
                },
                {
                  label: "Effort",
                  value: result.effort_estimate.replace("_", " "),
                  color: EFFORT_COLORS[result.effort_estimate] ?? "var(--ink)",
                },
                {
                  label: "Competition",
                  value: result.competition_level,
                  color:
                    result.competition_level === "high"
                      ? "var(--vermillion)"
                      : result.competition_level === "medium"
                        ? "var(--watch)"
                        : "var(--safe)",
                },
              ].map((m) => (
                <div
                  key={m.label}
                  style={{
                    padding: "12px",
                    border: "0.5px solid var(--rule)",
                    background: "var(--bg)",
                  }}
                >
                  <div className="meta-text" style={{ marginBottom: 6 }}>
                    {m.label}
                  </div>
                  <div
                    style={{
                      fontFamily: "var(--mono)",
                      fontSize: 13,
                      fontWeight: 700,
                      textTransform: "capitalize",
                      color: m.color,
                    }}
                  >
                    {m.value}
                  </div>
                </div>
              ))}
            </div>

            {/* Recommendation */}
            <div className="trace-log" style={{ padding: "12px 16px", marginBottom: 12 }}>
              <div className="log-line rule">
                <span style={{ fontSize: 12 }}>{result.recommendation}</span>
              </div>
            </div>

            {/* Analysis */}
            <div>
              <div className="meta-text" style={{ marginBottom: 6 }}>
                Analysis
              </div>
              <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.7 }}>
                {result.reasoning}
              </p>
            </div>

            {/* Live signals */}
            {result.trend_data && (
              <div
                className="meta-text"
                style={{
                  display: "flex",
                  gap: 20,
                  marginTop: 12,
                  paddingTop: 12,
                  borderTop: "0.5px solid var(--rule)",
                }}
              >
                <span>{result.trend_data.signals.news_mentions} news mentions</span>
                <span>{result.trend_data.signals.reddit_posts} reddit posts</span>
                <span>{result.trend_data.signals.hackernews_posts} HN posts</span>
              </div>
            )}
          </div>
        </div>
      )}

      <style>{`
        @keyframes slide {
          0% { left: -40%; }
          100% { left: 100%; }
        }
      `}</style>
    </div>
  );
}
