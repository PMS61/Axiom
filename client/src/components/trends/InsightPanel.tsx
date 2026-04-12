"use client";

import { useEffect, useState } from "react";
import { generateTopicInsight } from "@/app/actions/trends";

interface InsightPanelProps {
  topic: string;
}

interface Insight {
  why_it_matters: string;
  should_learn: boolean;
  effort_vs_roi: string;
  learning_path_hint: string;
}

export default function InsightPanel({ topic }: InsightPanelProps) {
  const [insight, setInsight] = useState<Insight | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    generateTopicInsight(topic).then((res) => {
      if (cancelled) return;
      if (res.error) {
        setError(res.error);
      } else if (res.insight) {
        setInsight(res.insight);
      }
      setLoading(false);
    });

    return () => {
      cancelled = true;
    };
  }, [topic]);

  if (loading) {
    return (
      <div
        style={{
          border: "0.5px solid var(--rule)",
          padding: "16px 20px",
          background: "var(--card-bg)",
        }}
      >
        <div className="meta-text" style={{ color: "var(--muted)" }}>
          Generating AI insight…
        </div>
        <div
          style={{
            marginTop: 8,
            display: "flex",
            flexDirection: "column",
            gap: 6,
          }}
        >
          {[0.6, 1, 0.75].map((w, i) => (
            <div
              key={i}
              style={{
                height: 8,
                width: `${w * 100}%`,
                background: "var(--rule)",
                borderRadius: 2,
              }}
            />
          ))}
        </div>
      </div>
    );
  }

  if (error || !insight) {
    return (
      <div
        style={{
          padding: "12px 16px",
          border: "0.5px solid var(--vermillion)",
        }}
      >
        <p className="meta-text" style={{ color: "var(--vermillion)" }}>
          [!] {error ?? "Failed to load insight."}
        </p>
      </div>
    );
  }

  return (
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
          padding: "10px 16px",
          borderBottom: "0.5px solid var(--rule)",
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}
      >
        <span className="meta-text">AI Insight</span>
        <span
          className="meta-text"
          style={{
            color: insight.should_learn ? "var(--safe)" : "var(--watch)",
            border: `0.5px solid ${insight.should_learn ? "var(--safe)" : "var(--watch)"}`,
            padding: "1px 8px",
          }}
        >
          {insight.should_learn ? "Recommended" : "Optional"}
        </span>
      </div>

      <div className="trace-log" style={{ padding: "16px 20px" }}>
        <div className="log-line rule" style={{ marginBottom: 12 }}>
          <span className="meta-text">Why It Matters</span>
          <p style={{ fontSize: 12, marginTop: 4, fontWeight: 400, color: "var(--ink)" }}>
            {insight.why_it_matters}
          </p>
        </div>
        <div className="log-line" style={{ marginBottom: 12 }}>
          <span className="meta-text">Effort vs ROI</span>
          <p style={{ fontSize: 12, marginTop: 4, color: "var(--muted)" }}>
            {insight.effort_vs_roi}
          </p>
        </div>
        <div className="log-line">
          <span className="meta-text">Where to Start</span>
          <p style={{ fontSize: 12, marginTop: 4, color: "var(--muted)" }}>
            {insight.learning_path_hint}
          </p>
        </div>
      </div>
    </div>
  );
}
