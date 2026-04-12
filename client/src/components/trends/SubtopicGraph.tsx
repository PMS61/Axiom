/* ═══════════════════════════════════════════════════════════
   Subtopic Dependency Graph — SVG DAG renderer
   Closed graph: prerequisites only reference the 10 subtopics.
   Layout: topological levels → columns, nodes → rows.
   ═══════════════════════════════════════════════════════════ */

"use client";

import { useMemo } from "react";
import type { ExploreSubtopic, ExploreSubtopicGraph } from "@/lib/trend-engine/types";

const NODE_W = 152;
const NODE_H = 52;
const COL_STEP = 220;   // horizontal distance between column centres
const ROW_STEP = 76;    // vertical distance between node centres
const PAD = 32;

interface SubtopicGraphProps {
  subtopics: ExploreSubtopic[];
  graph?: ExploreSubtopicGraph;
  addedIds: Set<string>;
  onNodeClick: (id: string) => void;
}

interface NodeLayout {
  id: string;
  name: string;
  difficulty: number;
  hours: number;
  cx: number;
  cy: number;
}

function computeLayout(
  subtopics: ExploreSubtopic[],
  graph?: ExploreSubtopicGraph,
): {
  nodes: NodeLayout[];
  edges: { fromX: number; fromY: number; toX: number; toY: number }[];
  svgW: number;
  svgH: number;
} {
  if (subtopics.length === 0) return { nodes: [], edges: [], svgW: 0, svgH: 0 };

  // 1. Assign topological levels
  const levels = new Map<string, number>();
  subtopics.forEach((s) => {
    if (s.prerequisites.length === 0) levels.set(s.id, 0);
  });

  let changed = true;
  while (changed) {
    changed = false;
    for (const s of subtopics) {
      const prereqLevels = s.prerequisites
        .map((p) => levels.get(p) ?? 0)
        .filter((_, i) => s.prerequisites[i] !== undefined);
      const newLevel =
        prereqLevels.length > 0 ? Math.max(...prereqLevels) + 1 : 0;
      if ((levels.get(s.id) ?? -1) < newLevel) {
        levels.set(s.id, newLevel);
        changed = true;
      }
    }
  }

  // Assign level 0 to any nodes that still have no level
  subtopics.forEach((s) => {
    if (!levels.has(s.id)) levels.set(s.id, 0);
  });

  // 2. Group by level
  const byLevel = new Map<number, ExploreSubtopic[]>();
  for (const s of subtopics) {
    const lv = levels.get(s.id) ?? 0;
    if (!byLevel.has(lv)) byLevel.set(lv, []);
    byLevel.get(lv)!.push(s);
  }

  const maxLevel = Math.max(...Array.from(levels.values()));
  const maxPerLevel = Math.max(...Array.from(byLevel.values()).map((v) => v.length));

  // 3. Assign pixel coordinates (centre of node)
  const pos = new Map<string, { cx: number; cy: number }>();
  for (const [lv, nodes] of byLevel) {
    const cx = PAD + NODE_W / 2 + lv * COL_STEP;
    const totalH = nodes.length * NODE_H + (nodes.length - 1) * (ROW_STEP - NODE_H);
    const fullH = maxPerLevel * NODE_H + (maxPerLevel - 1) * (ROW_STEP - NODE_H);
    const startY = PAD + (fullH - totalH) / 2 + NODE_H / 2;
    nodes.forEach((s, i) => {
      pos.set(s.id, { cx, cy: startY + i * ROW_STEP });
    });
  }

  // 4. Build node layout list
  const nodes: NodeLayout[] = subtopics.map((s) => {
    const { cx, cy } = pos.get(s.id) ?? { cx: PAD, cy: PAD };
    return { id: s.id, name: s.name, difficulty: s.difficulty, hours: s.hours, cx, cy };
  });

  // 5. Build edges
  const edges: { fromX: number; fromY: number; toX: number; toY: number }[] = [];
  const graphEdges = graph?.edges?.length
    ? graph.edges
    : subtopics.flatMap((s) => s.prerequisites.map((from) => ({ from, to: s.id })));

  for (const edge of graphEdges) {
    const from = pos.get(edge.from);
    const to = pos.get(edge.to);
    if (!from || !to) continue;
    edges.push({
      fromX: from.cx + NODE_W / 2,
      fromY: from.cy,
      toX: to.cx - NODE_W / 2,
      toY: to.cy,
    });
  }

  const svgW = PAD * 2 + NODE_W + maxLevel * COL_STEP;
  const svgH = PAD * 2 + maxPerLevel * NODE_H + (maxPerLevel - 1) * (ROW_STEP - NODE_H);

  return { nodes, edges, svgW, svgH };
}

function truncate(str: string, max: number): string {
  return str.length > max ? str.slice(0, max - 1) + "…" : str;
}

function difficultyColor(d: number): string {
  if (d <= 3) return "var(--safe)";
  if (d <= 6) return "var(--watch)";
  return "var(--vermillion)";
}

export default function SubtopicGraph({ subtopics, graph, addedIds, onNodeClick }: SubtopicGraphProps) {
  const { nodes, edges, svgW, svgH } = useMemo(
    () => computeLayout(subtopics, graph),
    [subtopics, graph],
  );

  if (nodes.length === 0) return null;

  return (
    <div
      style={{
        overflowX: "auto",
        overflowY: "visible",
        border: "0.5px solid var(--rule)",
        background: "var(--bg)",
      }}
    >
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        style={{ display: "block", minWidth: svgW }}
      >
        {/* Arrow marker */}
        <defs>
          <marker
            id="arrowhead"
            markerWidth="7"
            markerHeight="7"
            refX="6"
            refY="3.5"
            orient="auto"
          >
            <polygon
              points="0 0, 7 3.5, 0 7"
              fill="var(--muted)"
              opacity="0.6"
            />
          </marker>
        </defs>

        {/* Edges */}
        {edges.map((e, i) => {
          const dx = e.toX - e.fromX;
          const cp1x = e.fromX + dx * 0.45;
          const cp2x = e.toX - dx * 0.45;
          return (
            <path
              key={i}
              d={`M ${e.fromX} ${e.fromY} C ${cp1x} ${e.fromY}, ${cp2x} ${e.toY}, ${e.toX} ${e.toY}`}
              fill="none"
              stroke="var(--muted)"
              strokeWidth="1"
              strokeOpacity="0.5"
              markerEnd="url(#arrowhead)"
            />
          );
        })}

        {/* Nodes */}
        {nodes.map((n) => {
          const x = n.cx - NODE_W / 2;
          const y = n.cy - NODE_H / 2;
          const added = addedIds.has(n.id);
          return (
            <g
              key={n.id}
              onClick={() => onNodeClick(n.id)}
              style={{ cursor: "pointer" }}
              role="button"
              aria-label={n.name}
            >
              {/* Node rect */}
              <rect
                x={x}
                y={y}
                width={NODE_W}
                height={NODE_H}
                fill={added ? "var(--ink)" : "var(--card-bg)"}
                stroke={added ? "var(--ink)" : "var(--rule)"}
                strokeWidth="0.5"
                rx="1"
              />
              {/* Difficulty bar — left edge */}
              <rect
                x={x}
                y={y}
                width={3}
                height={NODE_H}
                fill={difficultyColor(n.difficulty)}
                rx="1"
              />

              {/* Topic name */}
              <text
                x={x + 12}
                y={y + 20}
                fontSize="11"
                fontWeight="600"
                fontFamily="inherit"
                fill={added ? "var(--bg)" : "var(--ink)"}
              >
                {truncate(n.name, 18)}
              </text>

              {/* Meta: hours + difficulty */}
              <text
                x={x + 12}
                y={y + 36}
                fontSize="9"
                fontFamily="var(--mono)"
                fill={added ? "var(--bg)" : "var(--muted)"}
                opacity="0.8"
              >
                {n.hours}h · D{n.difficulty}
              </text>

              {/* Added badge */}
              {added && (
                <text
                  x={x + NODE_W - 8}
                  y={y + 21}
                  fontSize="10"
                  textAnchor="end"
                  fill="var(--bg)"
                  opacity="0.8"
                >
                  ✓
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
