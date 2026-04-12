"use client";

/* Simple SVG sparkline — no external chart library */

interface TrendGraphProps {
  mentionsOverTime: number[];
  engagementOverTime: number[];
  direction: "rising" | "stable" | "declining";
}

function Sparkline({
  data,
  color,
  label,
}: {
  data: number[];
  color: string;
  label: string;
}) {
  const max = Math.max(...data, 1);
  const W = 200;
  const H = 48;
  const pad = 4;

  const points = data
    .map((v, i) => {
      const x = pad + (i / (data.length - 1)) * (W - pad * 2);
      const y = H - pad - ((v / max) * (H - pad * 2));
      return `${x},${y}`;
    })
    .join(" ");

  // Area fill path
  const firstX = pad;
  const lastX = W - pad;
  const areaPath = `M${firstX},${H - pad} L ${points} L${lastX},${H - pad} Z`;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <p className="meta-text">{label}</p>
      <svg width={W} height={H} style={{ width: "100%" }} viewBox={`0 0 ${W} ${H}`}>
        <title>{label}</title>
        <defs>
          <linearGradient id={`grad-${label}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.3" />
            <stop offset="100%" stopColor={color} stopOpacity="0.02" />
          </linearGradient>
        </defs>
        <path d={areaPath} fill={`url(#grad-${label})`} />
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="1.5"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {data.length > 0 && (
          <circle
            cx={W - pad}
            cy={H - pad - ((data[data.length - 1] / max) * (H - pad * 2))}
            r="2.5"
            fill={color}
          />
        )}
      </svg>
      <div
        className="meta-text"
        style={{ display: "flex", justifyContent: "space-between" }}
      >
        <span>48h ago</span>
        <span>now</span>
      </div>
    </div>
  );
}

export default function TrendGraph({
  mentionsOverTime,
  engagementOverTime,
  direction,
}: TrendGraphProps) {
  // Use fixed hex equivalents so SVG gradients work cross-browser without CSS vars
  const mentionColor =
    direction === "rising" ? "#2C6E49" : direction === "declining" ? "#C0392B" : "#1A1A1A";

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, paddingTop: 8 }}>
      <Sparkline data={mentionsOverTime} color={mentionColor} label="Mentions" />
      <Sparkline data={engagementOverTime} color="#a78bfa" label="Engagement" />
    </div>
  );
}
