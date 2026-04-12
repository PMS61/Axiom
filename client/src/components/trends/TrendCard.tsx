"use client";

import { useState } from "react";
import TrendGraph from "./TrendGraph";
import InsightPanel from "./InsightPanel";
import type { TrendResult } from "@/lib/trend-engine/types";

interface TrendCardProps {
  trend: TrendResult;
  rank: number;
  onAddToRoadmap: (topic: string, trendScore: number) => void;
}

const DIRECTION_COLORS: Record<string, string> = {
  rising: "var(--safe)",
  stable: "var(--ink)",
  declining: "var(--vermillion)",
};

const DIRECTION_LABELS: Record<string, string> = {
  rising: "↑ Rising",
  stable: "→ Stable",
  declining: "↓ Declining",
};

const MOMENTUM_LABELS: Record<string, string> = {
  accelerating: "Accelerating",
  steady: "Steady",
  slowing: "Slowing",
};

export default function TrendCard({ trend, rank, onAddToRoadmap }: TrendCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [showInsight, setShowInsight] = useState(false);
  const [added, setAdded] = useState(false);

  const dirColor = DIRECTION_COLORS[trend.direction] ?? "var(--ink)";
  const scoreColor =
    trend.trend_score >= 70
      ? "var(--safe)"
      : trend.trend_score >= 40
        ? "var(--watch)"
        : "var(--vermillion)";

  function handleAdd() {
    onAddToRoadmap(trend.topic, trend.trend_score);
    setAdded(true);
    setTimeout(() => setAdded(false), 3000);
  }

  return (
    <div
      style={{
        borderBottom: "0.5px solid var(--rule)",
        background: expanded ? "var(--card-bg)" : "transparent",
        transition: "background 0.15s",
      }}
    >
      {/* Row */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "40px 1fr 80px 80px 80px 60px",
          gap: 12,
          padding: "12px 16px",
          cursor: "pointer",
          alignItems: "center",
          borderLeft: expanded ? `3px solid ${dirColor}` : "3px solid transparent",
          transition: "border-color 0.15s",
        }}
        onClick={() => setExpanded((e) => !e)}
        onKeyDown={(e) => e.key === "Enter" && setExpanded((v) => !v)}
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
      >
        {/* Rank */}
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: 12,
            color: "var(--muted)",
            fontWeight: 700,
          }}
        >
          {rank}
        </span>

        {/* Topic + direction */}
        <div style={{ minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <span style={{ fontWeight: 600, fontSize: 14 }}>{trend.topic}</span>
            <span
              className="meta-text"
              style={{ color: dirColor, letterSpacing: "0.1em" }}
            >
              {DIRECTION_LABELS[trend.direction]}
            </span>
            <span className="meta-text" style={{ color: "var(--muted)" }}>
              {MOMENTUM_LABELS[trend.momentum]}
            </span>
          </div>
          <p
            style={{
              fontSize: 11,
              color: "var(--muted)",
              marginTop: 2,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {trend.insight}
          </p>
        </div>

        {/* Signal counts */}
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: 12,
            textAlign: "right",
            color: "var(--muted)",
          }}
        >
          {trend.signals.news_mentions}
        </span>
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: 12,
            textAlign: "right",
            color: "var(--muted)",
          }}
        >
          {trend.signals.reddit_posts}
        </span>
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: 12,
            textAlign: "right",
            color: "var(--muted)",
          }}
        >
          {trend.signals.hackernews_posts}
        </span>

        {/* Score */}
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: 16,
            fontWeight: 700,
            textAlign: "right",
            color: scoreColor,
          }}
        >
          {trend.trend_score}
        </span>
      </div>

      {/* Expanded panel */}
      {expanded && (
        <div
          style={{
            padding: "16px 16px 24px 59px",
            borderTop: "0.5px solid var(--rule)",
          }}
        >
          {/* Graphs */}
          <TrendGraph
            mentionsOverTime={trend.mentions_over_time}
            engagementOverTime={trend.engagement_over_time}
            direction={trend.direction}
          />

          {/* Future outlook */}
          <div
            className="trace-log"
            style={{ padding: "16px 20px", marginTop: 16, marginBottom: 16 }}
          >
            <div className="log-line">
              <span className="meta-text">Future Outlook</span>
            </div>
            <p style={{ marginTop: 8, fontSize: 12, paddingLeft: 16 }}>{trend.future_outlook}</p>
          </div>

          {/* AI Insight toggle */}
          <button
            type="button"
            className="btn btn-sm"
            style={{ marginBottom: showInsight ? 12 : 0 }}
            onClick={() => setShowInsight((v) => !v)}
          >
            {showInsight ? "Hide Deep Insight" : "Generate Deep Insight →"}
          </button>

          {showInsight && (
            <div style={{ marginTop: 12 }}>
              <InsightPanel topic={trend.topic} />
            </div>
          )}

          {/* Actions */}
          <div style={{ display: "flex", gap: 8, marginTop: 16 }}>
            <button
              type="button"
              className={`btn btn-sm ${added ? "" : "btn-primary"}`}
              style={
                added
                  ? { color: "var(--safe)", borderColor: "var(--safe)" }
                  : {}
              }
              onClick={handleAdd}
              disabled={added}
            >
              {added ? "✓ Added to Roadmap" : "+ Add to Roadmap"}
            </button>
            <a
              href={`/dashboard/trends/explore?topic=${encodeURIComponent(trend.topic)}`}
              className="btn btn-sm"
            >
              Explore →
            </a>
          </div>
        </div>
      )}
    </div>
  );
}
