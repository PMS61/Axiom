/* ═══════════════════════════════════════════════════════════
   Trend Intelligence — Compact Summary Section
   Embedded in the main dashboard below the task schedule.
   Three columns: Rising / Hot Now / Declining.
   ═══════════════════════════════════════════════════════════ */

import type { TrendResult } from "@/lib/trend-engine/types";

interface TrendsSummaryProps {
  trends: TrendResult[];
}

const MOMENTUM_SYMBOL: Record<string, string> = {
  accelerating: "↑↑",
  steady: "→",
  slowing: "↓",
};

function MiniTrendRow({ trend, accentColor }: { trend: TrendResult; accentColor: string }) {
  return (
    <a
      href={`/dashboard/trends/explore?topic=${encodeURIComponent(trend.topic)}`}
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "10px 14px",
        borderBottom: "0.5px solid var(--rule)",
        textDecoration: "none",
        color: "inherit",
        transition: "background 0.1s",
        cursor: "pointer",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLElement).style.background = "var(--bg)";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLElement).style.background = "transparent";
      }}
    >
      <div style={{ minWidth: 0, flex: 1 }}>
        <div
          style={{
            fontWeight: 600,
            fontSize: 13,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            color: "var(--ink)",
          }}
        >
          {trend.topic}
        </div>
        <div
          className="meta-text"
          style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}
        >
          {MOMENTUM_SYMBOL[trend.momentum]} {trend.signals.news_mentions}n ·{" "}
          {trend.signals.reddit_posts}r · {trend.signals.hackernews_posts}hn
        </div>
      </div>
      <span
        style={{
          fontFamily: "var(--mono)",
          fontSize: 15,
          fontWeight: 700,
          color: accentColor,
          marginLeft: 12,
          flexShrink: 0,
        }}
      >
        {trend.trend_score}
      </span>
    </a>
  );
}

interface ColumnCardProps {
  label: string;
  sublabel: string;
  accentColor: string;
  borderColor: string;
  badgeColor: string;
  items: TrendResult[];
  emptyText: string;
}

function ColumnCard({
  label,
  sublabel,
  accentColor,
  borderColor,
  badgeColor,
  items,
  emptyText,
}: ColumnCardProps) {
  return (
    <div
      style={{
        border: "0.5px solid var(--rule)",
        borderTop: `3px solid ${borderColor}`,
        background: "var(--card-bg)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      {/* Card Header */}
      <div
        style={{
          padding: "16px 14px 12px",
          borderBottom: "0.5px solid var(--rule)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <div style={{ fontWeight: 700, fontSize: 13, color: "var(--ink)" }}>{label}</div>
          <div className="meta-text" style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
            {sublabel}
          </div>
        </div>
        <div
          style={{
            background: badgeColor,
            color: "var(--bg)",
            fontFamily: "var(--mono)",
            fontSize: 13,
            fontWeight: 700,
            padding: "2px 10px",
            borderRadius: 2,
            minWidth: 28,
            textAlign: "center",
          }}
        >
          {items.length}
        </div>
      </div>

      {/* Rows */}
      <div style={{ flex: 1 }}>
        {items.length === 0 ? (
          <div
            style={{
              padding: "24px 14px",
              textAlign: "center",
              color: "var(--muted)",
              fontSize: 12,
            }}
          >
            {emptyText}
          </div>
        ) : (
          items
            .slice(0, 5)
            .map((t) => (
              <MiniTrendRow key={t.topic} trend={t} accentColor={accentColor} />
            ))
        )}
      </div>
    </div>
  );
}

export default function TrendsSummary({ trends }: TrendsSummaryProps) {
  if (trends.length === 0) return null;

  const rising = trends.filter((t) => t.direction === "rising").sort((a, b) => b.trend_score - a.trend_score);
  const stable = trends.filter((t) => t.direction === "stable").sort((a, b) => b.trend_score - a.trend_score);
  const declining = trends.filter((t) => t.direction === "declining").sort((a, b) => b.trend_score - a.trend_score);

  const topTrend = trends[0];

  return (
    <section
      className="container"
      style={{
        paddingTop: 0,
        paddingBottom: 80,
        borderTop: "0.5px solid var(--rule)",
      }}
    >
      {/* Section Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          paddingTop: 40,
          paddingBottom: 24,
          borderBottom: "0.5px solid var(--rule)",
          marginBottom: 32,
        }}
      >
        <div>
          <div className="meta-text" style={{ marginBottom: 6 }}>
            TREND INTELLIGENCE
          </div>
          <h2 style={{ fontSize: 22, margin: 0, fontWeight: 700 }}>
            What to Learn Next
          </h2>
          <p style={{ color: "var(--muted)", fontSize: 12, marginTop: 6, marginBottom: 0 }}>
            Live signals from News, Reddit &amp; Hacker News ·{" "}
            {trends.length} topics tracked
            {topTrend && (
              <>
                {" "}· Top signal:{" "}
                <span style={{ fontWeight: 600, color: "var(--ink)" }}>
                  {topTrend.topic}
                </span>{" "}
                ({topTrend.trend_score})
              </>
            )}
          </p>
        </div>
        <a
          href="/dashboard/trends"
          className="btn btn-sm"
          style={{ flexShrink: 0, textDecoration: "none" }}
        >
          View All →
        </a>
      </div>

      {/* Three-column grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 20,
        }}
      >
        <ColumnCard
          label="Rising"
          sublabel="Gaining traction fast"
          accentColor="var(--safe)"
          borderColor="var(--safe)"
          badgeColor="var(--safe)"
          items={rising}
          emptyText="No rising topics detected."
        />
        <ColumnCard
          label="Stable / Hot Now"
          sublabel="Consistently active signals"
          accentColor="var(--ink)"
          borderColor="var(--ink)"
          badgeColor="var(--ink)"
          items={stable}
          emptyText="No stable topics detected."
        />
        <ColumnCard
          label="Declining"
          sublabel="Losing momentum — deprioritise"
          accentColor="var(--vermillion)"
          borderColor="var(--vermillion)"
          badgeColor="var(--vermillion)"
          items={declining}
          emptyText="No declining topics detected."
        />
      </div>

      {/* Footer */}
      <div
        className="meta-text"
        style={{
          display: "flex",
          gap: 24,
          paddingTop: 20,
          marginTop: 24,
          borderTop: "0.5px solid var(--rule)",
          color: "var(--muted)",
          flexWrap: "wrap",
        }}
      >
        <span>Sources: NewsAPI · Reddit · Hacker News</span>
        <span>Insights: Gemini 2.5 Flash Lite</span>
      </div>
    </section>
  );
}
