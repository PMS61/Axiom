/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Bandwidth Curve
   Matches landing.html stepped area chart: gray fill for
   available energy, red fill for overload zones, raw data
   line on top with miter joins.
   ═══════════════════════════════════════════════════════════ */

"use client";

import { useApp } from "@/lib/store";
import { slotToTime } from "@/lib/engine";

export default function BandwidthCurve() {
  const { state } = useApp();
  const curve = state.adjustedBandwidthCurve;
  const WIDTH = 1000;
  const HEIGHT = 140;
  const START = 24; // 06:00
  const END = 92;   // 23:00
  const slots = END - START;
  const maxBW = Math.max(...curve.slice(START, END), 1);

  // Compute Etask demand per slot from section schedule (primary) with slot fallback.
  const clPerSlot = new Array(96).fill(0);
  const today = state.scheduledDays.find((d) => d.dayOffset === 0);

  if (today) {
    const sectionFallbackStart = (section: string): number => {
      if (section === "morning") return 24;
      if (section === "afternoon") return 48;
      return 72;
    };

    for (const section of today.sections) {
      for (const item of section.tasks) {
        if (item.isRecreational) continue;

        const task = state.tasks.find((t) => t.id === item.taskId);
        if (!task || task.state === "completed" || task.state === "skipped") continue;

        const start = item.startSlot ?? sectionFallbackStart(section.section);
        const end = start + Math.ceil(item.completionTime / 15);
        const clPerUnit = Math.abs(task.etask) / Math.max(end - start, 1);

        for (let s = start; s < end; s++) {
          clPerSlot[s] += task.etask > 0 ? clPerUnit : 0;
        }
      }
    }
  } else {
    for (const task of state.tasks) {
      if (task.scheduledSlot && task.state !== "completed" && task.state !== "skipped") {
        const clPerUnit = Math.abs(task.etask) / Math.max(task.scheduledSlot.endSlot - task.scheduledSlot.startSlot, 1);
        for (let s = task.scheduledSlot.startSlot; s < task.scheduledSlot.endSlot; s++) {
          clPerSlot[s] += task.etask > 0 ? clPerUnit : 0;
        }
      }
    }
  }

  function x(i: number) { return (i / (slots - 1)) * WIDTH; }
  function y(val: number) { return HEIGHT - 20 - (val / maxBW) * (HEIGHT - 30); }

  // Build area fill path (closed polygon)
  let fillPath = `M${x(0)},${HEIGHT - 20}`;
  for (let i = 0; i < slots; i++) {
    fillPath += ` L${x(i)},${y(curve[START + i])}`;
  }
  fillPath += ` L${x(slots - 1)},${HEIGHT - 20} Z`;

  // Build line path
  let linePath = `M${x(0)},${y(curve[START])}`;
  for (let i = 1; i < slots; i++) {
    linePath += ` L${x(i)},${y(curve[START + i])}`;
  }

  // Find overload zones (CL > bandwidth)
  const overloadPaths: string[] = [];
  let inOverload = false;
  let overloadPath = "";
  for (let i = 0; i < slots; i++) {
    const slot = START + i;
    if (clPerSlot[slot] > curve[slot]) {
      if (!inOverload) {
        inOverload = true;
        overloadPath = `M${x(i)},${y(curve[slot])}`;
      }
      overloadPath += ` L${x(i)},${y(curve[slot] + (clPerSlot[slot] - curve[slot]))}`;
    } else if (inOverload) {
      inOverload = false;
      overloadPath += ` L${x(i - 1)},${y(curve[START + i - 1])} Z`;
      overloadPaths.push(overloadPath);
    }
  }

  // Time labels
  const labels: { x: number; text: string }[] = [];
  for (let slot = START; slot <= END; slot += 8) {
    labels.push({ x: x(slot - START), text: slotToTime(slot) });
  }

  // Threshold line at CL 7
  const thresholdY = y(7);

  return (
    <div>
      <div className="chart-wrapper" style={{ height: HEIGHT + 20 }}>
        <svg
          className="energy-chart"
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          preserveAspectRatio="none"
          style={{ width: "100%", height: "100%", overflow: "visible" }}
        >
          {/* Grid lines at 1/3 and 2/3 */}
          <line x1={WIDTH / 3} y1={0} x2={WIDTH / 3} y2={HEIGHT - 20} className="chart-grid" />
          <line x1={(WIDTH * 2) / 3} y1={0} x2={(WIDTH * 2) / 3} y2={HEIGHT - 20} className="chart-grid" />

          {/* Burnout / threshold line */}
          <line
            x1={0} y1={thresholdY} x2={WIDTH} y2={thresholdY}
            stroke="var(--vermillion)" strokeWidth={1} strokeDasharray="6 4"
          />

          {/* Area fill */}
          <path d={fillPath} className="chart-fill" />

          {/* Overload zones */}
          {overloadPaths.map((p, i) => (
            <path key={i} d={p} className="chart-fill-danger" opacity={0.5} />
          ))}

          {/* Main data line */}
          <path d={linePath} className="chart-line" />
        </svg>

        {/* Threshold label overlaid via HTML */}
        <div style={{
          position: "absolute",
          top: thresholdY - 14,
          right: 4,
          fontFamily: "var(--mono)",
          fontSize: 9,
          fill: "var(--vermillion)",
          color: "var(--vermillion)",
          fontWeight: 700
        }}>
          [!] CL THRESHOLD
        </div>
      </div>

      <div className="chart-legend">
        {labels.map((l, i) => (
          <span key={i}>{l.text}</span>
        ))}
      </div>
    </div>
  );
}
