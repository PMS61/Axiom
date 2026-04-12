/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Roadmap DAG View
   Renders the adaptive learning roadmap as an interactive
   Directed Acyclic Graph. Nodes update their state in
   real time as mastery scores change.
   ═══════════════════════════════════════════════════════════ */

"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { useApp } from "@/lib/store";
import { getStoredContentTypesAction, type StoredContentKind } from "@/app/actions/courses";
import type { RoadmapDAGNode, RoadmapNodeState } from "@/lib/types";

// ── Constants ─────────────────────────────────────────────

const NODE_W = 320;
const NODE_H = 140;
const H_GAP = 40;
const V_GAP = 80;

const STATE_COLORS: Record<RoadmapNodeState, string> = {
  locked: "var(--rule)",
  available: "var(--ink)",
  in_progress: "var(--watch)",
  mastered: "var(--safe)",
};

const STATE_LABELS: Record<RoadmapNodeState, string> = {
  locked: "LOCKED",
  available: "AVAILABLE",
  in_progress: "IN PROGRESS",
  mastered: "MASTERED",
};

const STATE_ICONS: Record<RoadmapNodeState, string> = {
  locked: "⊠",
  available: "⊡",
  in_progress: "⊞",
  mastered: "⊛",
};

// ── Layout Engine: Topological sort → column assignment ───

function computeLayout(
  nodes: RoadmapDAGNode[],
): Map<string, { col: number; row: number }> {
  const positions = new Map<string, { col: number; row: number }>();
  const colCounts = new Map<number, number>();

  // BFS-based topological layering
  const inDegree = new Map<string, number>();
  const adj = new Map<string, string[]>(); // id → dependents

  for (const n of nodes) {
    if (!inDegree.has(n.id)) inDegree.set(n.id, 0);
    if (!adj.has(n.id)) adj.set(n.id, []);
    for (const p of n.prerequisites) {
      inDegree.set(n.id, (inDegree.get(n.id) ?? 0) + 1);
      if (!adj.has(p)) adj.set(p, []);
      adj.get(p)!.push(n.id);
    }
  }

  // Assign columns via longest path from root
  const colOf = new Map<string, number>();
  const queue: string[] = [];
  const ready = new Set<string>();

  for (const n of nodes) {
    if ((inDegree.get(n.id) ?? 0) === 0) {
      queue.push(n.id);
      colOf.set(n.id, 0);
      ready.add(n.id);
    }
  }

  const processed = new Set<string>();
  while (queue.length > 0) {
    const id = queue.shift()!;
    if (processed.has(id)) continue;
    processed.add(id);

    const col = colOf.get(id) ?? 0;
    const count = colCounts.get(col) ?? 0;
    colCounts.set(col, count + 1);
    positions.set(id, { col, row: count });

    for (const dep of adj.get(id) ?? []) {
      colOf.set(dep, Math.max(colOf.get(dep) ?? 0, col + 1));
      // Check if all predecessors are processed
      const node = nodes.find((n) => n.id === dep);
      if (node && node.prerequisites.every((p) => processed.has(p))) {
        if (!ready.has(dep)) {
          ready.add(dep);
          queue.push(dep);
        }
      }
    }
  }

  // Second pass: fix row counts per column
  const perCol = new Map<number, string[]>();
  for (const [id, pos] of positions.entries()) {
    if (!perCol.has(pos.col)) perCol.set(pos.col, []);
    perCol.get(pos.col)!.push(id);
  }
  for (const [col, ids] of perCol.entries()) {
    ids.forEach((id, idx) => {
      positions.set(id, { col, row: idx });
    });
  }

  return positions;
}

// ── Edge connector (SVG path) ──────────────────────────────

function EdgePath({
  fromX,
  fromY,
  toX,
  toY,
  color,
}: {
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  color: string;
}) {
  const midY = (fromY + toY) / 2;
  const d = `M ${fromX} ${fromY} C ${fromX} ${midY} ${toX} ${midY} ${toX} ${toY}`;
  return (
    <path
      d={d}
      fill="none"
      stroke={color}
      strokeWidth={1.5}
      strokeDasharray={color === "var(--rule)" ? "4 4" : undefined}
      opacity={0.6}
    />
  );
}

// ── Node Card ─────────────────────────────────────────────

function NodeCard({
  node,
  x,
  y,
  isSelected,
  onClick,
}: {
  node: RoadmapDAGNode;
  x: number;
  y: number;
  isSelected: boolean;
  onClick: () => void;
}) {
  const stateColor = STATE_COLORS[node.nodeState];
  const isLocked = node.nodeState === "locked";
  const progress = Math.min(100, Math.round(node.masteryScore * 100));

  return (
    <g
      transform={`translate(${x}, ${y})`}
      onClick={isLocked ? undefined : onClick}
      style={{ cursor: isLocked ? "not-allowed" : "pointer" }}
    >
      {/* Shadow */}
      <rect
        x={3}
        y={3}
        width={NODE_W}
        height={NODE_H}
        fill="rgba(0,0,0,0.06)"
        rx={0}
      />

      {/* Background */}
      <rect
        x={0}
        y={0}
        width={NODE_W}
        height={NODE_H}
        fill={isSelected ? "var(--ink)" : "var(--card-bg)"}
        stroke={isSelected ? "var(--ink)" : stateColor}
        strokeWidth={isSelected ? 1.5 : 0.5}
        rx={0}
        opacity={isLocked ? 0.55 : 1}
      />

      {/* Top accent bar */}
      <rect
        x={0}
        y={0}
        width={NODE_W}
        height={3}
        fill={stateColor}
        opacity={isLocked ? 0.4 : 1}
      />

      {/* Mastery progress bar */}
      {progress > 0 && (
        <>
          <rect
            x={0}
            y={NODE_H - 3}
            width={NODE_W}
            height={3}
            fill="var(--rule)"
          />
          <rect
            x={0}
            y={NODE_H - 3}
            width={(NODE_W * progress) / 100}
            height={3}
            fill={stateColor}
          />
        </>
      )}

      {/* Type badge */}
      <text
        x={12}
        y={22}
        fontFamily="var(--mono)"
        fontSize={8}
        letterSpacing="0.15em"
        fill={isSelected ? "var(--bg)" : "var(--muted)"}
        textAnchor="start"
      >
        {node.type === "assessment" ? "▲ ASSESSMENT" : "● COURSE"}
      </text>

      {/* State icon + label */}
      <text
        x={NODE_W - 12}
        y={22}
        fontFamily="var(--mono)"
        fontSize={8}
        letterSpacing="0.1em"
        fill={isSelected ? "var(--bg)" : stateColor}
        textAnchor="end"
      >
        {STATE_ICONS[node.nodeState]} {STATE_LABELS[node.nodeState]}
      </text>

      {/* Title & Description */}
      <foreignObject x={16} y={36} width={NODE_W - 32} height={NODE_H - 60}>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <div
            style={{
              fontFamily: "var(--mono)",
              fontSize: 14,
              fontWeight: 700,
              color: isSelected ? "var(--bg)" : "var(--ink)",
              lineHeight: 1.35,
              overflow: "hidden",
              display: "-webkit-box",
              WebkitLineClamp: 2,
              WebkitBoxOrient: "vertical",
              opacity: isLocked ? 0.6 : 1,
            }}
          >
            {node.title}
          </div>
          <div
            style={{
              fontFamily: "var(--sans, var(--mono))",
              fontSize: 11,
              color: isSelected ? "var(--bg)" : "var(--muted)",
              lineHeight: 1.4,
              overflow: "hidden",
              display: "-webkit-box",
              WebkitLineClamp: 3,
              WebkitBoxOrient: "vertical",
              opacity: isLocked ? 0.5 : 0.8,
            }}
          >
            {node.description}
          </div>
        </div>
      </foreignObject>

      {/* Meta row */}
      <text
        x={12}
        y={NODE_H - 10}
        fontFamily="var(--mono)"
        fontSize={9}
        fill={isSelected ? "var(--bg)" : "var(--muted)"}
        opacity={0.8}
      >
        {node.estimatedMinutes}m · {node.subject} · diff {node.difficulty}/10
      </text>
    </g>
  );
}

// ── Main DAG View ─────────────────────────────────────────

interface RoadmapDAGViewProps {
  nodes: RoadmapDAGNode[];
  onNodeSelect?: (node: RoadmapDAGNode) => void;
  onMasteryUpdate?: (nodeId: string, score: number) => void;
  onConvertToTask?: (node: RoadmapDAGNode) => void;
}

type ContentExperience =
  | "ppt"
  | "flashcards"
  | "shortbits"
  | "one-shot"
  | "storytelling";

const STORED_CONTENT_OPTIONS: Array<{
  value: StoredContentKind;
  label: string;
}> = [
  { value: "course", label: "Slides (PPT)" },
  { value: "flashcards", label: "Flashcards" },
  { value: "short_bits", label: "Short Bits" },
  { value: "cheatsheet", label: "Cheat Sheet" },
  { value: "story", label: "Story" },
];

export default function RoadmapDAGView({
  nodes,
  onNodeSelect,
  onMasteryUpdate,
  onConvertToTask,
}: RoadmapDAGViewProps) {
  const router = useRouter();
  const { state } = useApp();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 40 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [selectedExperience, setSelectedExperience] =
    useState<ContentExperience>("ppt");
  const [storedContentTypes, setStoredContentTypes] = useState<StoredContentKind[]>([]);
  const [selectedStoredType, setSelectedStoredType] = useState<StoredContentKind>("course");

  const positions = useMemo(() => computeLayout(nodes), [nodes]);

  // Compute SVG viewBox dimensions based on centered layers
  const colSizes = useMemo(() => {
    const counts = new Map<number, number>();
    for (const [id, pos] of positions.entries()) {
      counts.set(pos.col, Math.max(counts.get(pos.col) ?? 0, pos.row + 1));
    }
    return counts;
  }, [positions]);

  const maxCol = Math.max(
    ...Array.from(positions.values()).map((p) => p.col),
    0,
  );
  const maxRowElements = Math.max(...Array.from(colSizes.values()), 0);
  const maxLayerWidth = maxRowElements * (NODE_W + H_GAP) - H_GAP;

  const svgWidth = Math.max(maxLayerWidth + 120, 800); // 60px padding on each side
  const svgHeight = (maxCol + 1) * (NODE_H + V_GAP) + 80;

  const nodeX = (col: number, row: number) => {
    const numElements = colSizes.get(col) ?? 1;
    const layerWidth = numElements * (NODE_W + H_GAP) - H_GAP;
    const offset = (svgWidth - layerWidth) / 2;
    return offset + row * (NODE_W + H_GAP);
  };
  const nodeY = (col: number) => col * (NODE_H + V_GAP) + 40;

  // Edges
  const edges = useMemo(() => {
    const result: { from: RoadmapDAGNode; to: RoadmapDAGNode }[] = [];
    for (const node of nodes) {
      for (const prereqId of node.prerequisites) {
        const from = nodes.find((n) => n.id === prereqId);
        if (from) result.push({ from, to: node });
      }
    }
    return result;
  }, [nodes]);

  const handleMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    if ((e.target as Element).closest("g[data-node]")) return;
    setIsPanning(true);
    setPanStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!isPanning) return;
    setPanOffset({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
  };

  const handleMouseUp = () => setIsPanning(false);

  const selectedNode = selectedId
    ? nodes.find((n) => n.id === selectedId)
    : null;
  const selectedTaskId = selectedNode?.taskId ?? null;

  const linkedTask = useMemo(() => {
    if (!selectedTaskId) return null;
    return state.tasks.find((t) => t.id === selectedTaskId) ?? null;
  }, [selectedTaskId, state.tasks]);

  useEffect(() => {
    if (!linkedTask) {
      setSelectedExperience("ppt");
      return;
    }
    setSelectedExperience(
      (linkedTask.contentType ?? "ppt") as ContentExperience,
    );
  }, [linkedTask]);

  useEffect(() => {
    let cancelled = false;

    async function loadStoredTypes() {
      if (!selectedTaskId) {
        setStoredContentTypes([]);
        return;
      }

      const res = await getStoredContentTypesAction(selectedTaskId);
      if (cancelled) return;

      if (res.types) {
        setStoredContentTypes(res.types);
      } else {
        setStoredContentTypes([]);
      }
    }

    loadStoredTypes();

    return () => {
      cancelled = true;
    };
  }, [selectedTaskId]);

  useEffect(() => {
    if (storedContentTypes.length === 0) {
      setSelectedStoredType("course");
      return;
    }
    if (!storedContentTypes.includes(selectedStoredType)) {
      setSelectedStoredType(storedContentTypes[0]);
    }
  }, [storedContentTypes, selectedStoredType]);

  const getStoredContentRoute = useCallback(
    (taskId: string, kind: StoredContentKind) => {
      switch (kind) {
        case "flashcards":
          return `/dashboard/flashcard/${taskId}`;
        case "short_bits":
          return `/dashboard/short-bit/${taskId}`;
        case "story":
          return `/dashboard/story/${taskId}`;
        case "cheatsheet":
          return `/dashboard/cheatsheet/${taskId}`;
        default:
          return `/dashboard/course/${taskId}`;
      }
    },
    [],
  );

  const getPrimaryContentRoute = useCallback(
    (taskId: string, experience: ContentExperience) => {
      switch (experience) {
        case "flashcards":
          return `/dashboard/flashcard/${taskId}`;
        case "shortbits":
          return `/dashboard/short-bit/${taskId}`;
        case "storytelling":
          return `/dashboard/story/${taskId}`;
        case "one-shot":
          return `/dashboard/cheatsheet/${taskId}`;
        default:
          return `/dashboard/course/${taskId}`;
      }
    },
    [],
  );

  const getPrimaryContentLabel = useCallback(
    (experience: ContentExperience) => {
      switch (experience) {
        case "flashcards":
          return "FLASHCARDS";
        case "shortbits":
          return "SHORTBITS";
        case "storytelling":
          return "STORY";
        case "one-shot":
          return "CHEAT SHEET";
        default:
          return "PPT";
      }
    },
    [],
  );

  return (
    <div style={{ display: "flex", height: "100%", gap: 0 }}>
      {/* ── DAG Canvas ── */}
      <div
        style={{
          flex: 1,
          overflow: "hidden",
          position: "relative",
          background: "var(--bg)",
          cursor: isPanning ? "grabbing" : "grab",
          display: "flex",
          justifyContent: "center",
        }}
      >
        {/* Grid background */}
        <svg
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
          }}
          preserveAspectRatio="none"
        >
          <defs>
            <pattern
              id="dag-grid"
              width="40"
              height="40"
              patternUnits="userSpaceOnUse"
            >
              <path
                d="M 40 0 L 0 0 0 40"
                fill="none"
                stroke="var(--rule)"
                strokeWidth="0.3"
              />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#dag-grid)" />
        </svg>

        <svg
          width={svgWidth}
          height={svgHeight}
          style={{
            display: "block",
            overflow: "visible",
            transform: `translate(${panOffset.x}px, ${panOffset.y}px)`,
            transition: isPanning ? "none" : "transform 0.1s",
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          {/* Edges */}
          <g>
            {edges.map(({ from, to }) => {
              const fp = positions.get(from.id);
              const tp = positions.get(to.id);
              if (!fp || !tp) return null;
              const fromX = nodeX(fp.col, fp.row) + NODE_W / 2;
              const fromY = nodeY(fp.col) + NODE_H;
              const toX = nodeX(tp.col, tp.row) + NODE_W / 2;
              const toY = nodeY(tp.col);
              const color =
                to.nodeState === "locked"
                  ? "var(--rule)"
                  : to.nodeState === "mastered"
                    ? "var(--safe)"
                    : to.nodeState === "in_progress"
                      ? "var(--watch)"
                      : "var(--ink)";
              return (
                <EdgePath
                  key={`${from.id}->${to.id}`}
                  fromX={fromX}
                  fromY={fromY}
                  toX={toX}
                  toY={toY}
                  color={color}
                />
              );
            })}
          </g>

          {/* Nodes */}
          <g>
            {nodes.map((node) => {
              const pos = positions.get(node.id);
              if (!pos) return null;
              return (
                <g key={node.id} data-node={node.id}>
                  <NodeCard
                    node={node}
                    x={nodeX(pos.col, pos.row)}
                    y={nodeY(pos.col)}
                    isSelected={selectedId === node.id}
                    onClick={() => {
                      setSelectedId((prev) =>
                        prev === node.id ? null : node.id,
                      );
                      onNodeSelect?.(node);
                    }}
                  />
                </g>
              );
            })}
          </g>
        </svg>

        {/* Legend */}
        <div
          style={{
            position: "absolute",
            bottom: 24,
            left: 24,
            display: "flex",
            gap: 16,
            background: "var(--card-bg)",
            border: "0.5px solid var(--rule)",
            padding: "10px 16px",
          }}
        >
          {(
            [
              "locked",
              "available",
              "in_progress",
              "mastered",
            ] as RoadmapNodeState[]
          ).map((state) => (
            <div
              key={state}
              style={{ display: "flex", alignItems: "center", gap: 6 }}
            >
              <div
                style={{
                  width: 8,
                  height: 8,
                  background: STATE_COLORS[state],
                  flexShrink: 0,
                }}
              />
              <span
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 9,
                  letterSpacing: "0.12em",
                  color: "var(--muted)",
                }}
              >
                {STATE_LABELS[state]}
              </span>
            </div>
          ))}
        </div>

        {/* Node count */}
        <div
          style={{
            position: "absolute",
            top: 16,
            right: 16,
            fontFamily: "var(--mono)",
            fontSize: 9,
            letterSpacing: "0.15em",
            color: "var(--muted)",
          }}
        >
          {nodes.filter((n) => n.nodeState === "mastered").length}/
          {nodes.length} MASTERED ·{" "}
          {nodes.filter((n) => n.nodeState === "in_progress").length} IN
          PROGRESS
        </div>
      </div>

      {/* ── Detail Panel ── */}
      {selectedNode && (
        <div
          style={{
            width: 320,
            borderLeft: "0.5px solid var(--rule)",
            background: "var(--card-bg)",
            padding: "32px 24px",
            overflowY: "auto",
            flexShrink: 0,
          }}
        >
          {/* State badge */}
          <div
            style={{
              display: "inline-block",
              padding: "4px 10px",
              border: `0.5px solid ${STATE_COLORS[selectedNode.nodeState]}`,
              color: STATE_COLORS[selectedNode.nodeState],
              fontFamily: "var(--mono)",
              fontSize: 9,
              letterSpacing: "0.12em",
              marginBottom: 16,
            }}
          >
            {STATE_ICONS[selectedNode.nodeState]}{" "}
            {STATE_LABELS[selectedNode.nodeState]}
          </div>

          <h3
            style={{
              fontSize: 16,
              fontWeight: 700,
              marginBottom: 8,
              lineHeight: 1.4,
            }}
          >
            {selectedNode.title}
          </h3>

          <div
            className="meta-text"
            style={{ marginBottom: 16, color: "var(--muted)" }}
          >
            {selectedNode.subject} · {selectedNode.type.toUpperCase()}
          </div>

          <p
            style={{
              fontSize: 12,
              color: "var(--muted)",
              lineHeight: 1.7,
              marginBottom: 24,
            }}
          >
            {selectedNode.description}
          </p>

          {/* Stats */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 12,
              marginBottom: 24,
              padding: 16,
              border: "0.5px solid var(--rule)",
              background: "var(--bg)",
            }}
          >
            <div>
              <div
                style={{
                  fontSize: 9,
                  fontFamily: "var(--mono)",
                  color: "var(--muted)",
                  letterSpacing: "0.12em",
                  marginBottom: 4,
                }}
              >
                DIFFICULTY
              </div>
              <div
                style={{
                  fontWeight: 700,
                  fontSize: 22,
                  fontFamily: "var(--mono)",
                }}
              >
                {selectedNode.difficulty}/10
              </div>
            </div>
            <div>
              <div
                style={{
                  fontSize: 9,
                  fontFamily: "var(--mono)",
                  color: "var(--muted)",
                  letterSpacing: "0.12em",
                  marginBottom: 4,
                }}
              >
                EST. TIME
              </div>
              <div
                style={{
                  fontWeight: 700,
                  fontSize: 22,
                  fontFamily: "var(--mono)",
                }}
              >
                {selectedNode.estimatedMinutes}m
              </div>
            </div>
            <div>
              <div
                style={{
                  fontSize: 9,
                  fontFamily: "var(--mono)",
                  color: "var(--muted)",
                  letterSpacing: "0.12em",
                  marginBottom: 4,
                }}
              >
                MASTERY
              </div>
              <div
                style={{
                  fontWeight: 700,
                  fontSize: 22,
                  fontFamily: "var(--mono)",
                  color:
                    selectedNode.masteryScore >= 0.8
                      ? "var(--safe)"
                      : selectedNode.masteryScore > 0
                        ? "var(--watch)"
                        : "var(--muted)",
                }}
              >
                {Math.round(selectedNode.masteryScore * 100)}%
              </div>
            </div>
            <div>
              <div
                style={{
                  fontSize: 9,
                  fontFamily: "var(--mono)",
                  color: "var(--muted)",
                  letterSpacing: "0.12em",
                  marginBottom: 4,
                }}
              >
                PREREQS
              </div>
              <div
                style={{
                  fontWeight: 700,
                  fontSize: 22,
                  fontFamily: "var(--mono)",
                }}
              >
                {selectedNode.prerequisites.length}
              </div>
            </div>
          </div>



          {/* Convert to Task (Now Automated, maybe Fallback) */}
          {onConvertToTask &&
            selectedNode.nodeState !== "locked" &&
            !selectedNode.taskId && (
              <button
                type="button"
                className="btn btn-primary"
                style={{
                  width: "100%",
                  marginBottom: 8,
                  justifyContent: "center",
                }}
                onClick={() => onConvertToTask(selectedNode)}
              >
                + Create CogFlow Task
              </button>
            )}

          {selectedTaskId &&
            (linkedTask && linkedTask.state === "scheduled" ? (
              <>
                {storedContentTypes.length > 0 && (
                  <>
                    <div style={{ marginBottom: 8 }}>
                      <label
                        htmlFor="stored-content-selector"
                        className="meta-text"
                        style={{ display: "block", marginBottom: 6 }}
                      >
                        STORED CONTENT
                      </label>
                      <select
                        id="stored-content-selector"
                        value={selectedStoredType}
                        onChange={(e) =>
                          setSelectedStoredType(e.target.value as StoredContentKind)
                        }
                        style={{ width: "100%" }}
                      >
                        {STORED_CONTENT_OPTIONS.filter((opt) =>
                          storedContentTypes.includes(opt.value),
                        ).map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    <button
                      type="button"
                      className="btn btn-primary"
                      style={{
                        width: "100%",
                        marginBottom: 12,
                        justifyContent: "center",
                      }}
                      onClick={() =>
                        router.push(
                          getStoredContentRoute(selectedTaskId, selectedStoredType),
                        )
                      }
                    >
                      ↳ Open Stored Content
                    </button>
                  </>
                )}

                <div style={{ marginBottom: 8 }}>
                  <label
                    htmlFor="content-experience-selector"
                    className="meta-text"
                    style={{ display: "block", marginBottom: 6 }}
                  >
                    SELECT CONTENT MODE {storedContentTypes.length > 0 ? "(GENERATE / REGENERATE)" : ""}
                  </label>
                  <select
                    id="content-experience-selector"
                    value={selectedExperience}
                    onChange={(e) =>
                      setSelectedExperience(e.target.value as ContentExperience)
                    }
                    style={{ width: "100%" }}
                  >
                    <option value="ppt">Slides (PPT)</option>
                    <option value="flashcards">Flashcards</option>
                    <option value="shortbits">Short Bits</option>
                    <option value="one-shot">Cheat Sheet</option>
                    <option value="storytelling">Story</option>
                  </select>
                </div>

                <button
                  type="button"
                  className="btn btn-primary"
                  style={{
                    width: "100%",
                    marginBottom: 8,
                    justifyContent: "center",
                  }}
                  onClick={() =>
                    router.push(
                      getPrimaryContentRoute(
                        selectedTaskId,
                        selectedExperience,
                      ),
                    )
                  }
                >
                  ✦ {storedContentTypes.length > 0 ? "Generate / Regenerate" : "Generate"} {getPrimaryContentLabel(selectedExperience)}
                </button>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 8,
                    marginBottom: 8,
                  }}
                >
                  <button
                    type="button"
                    className="btn"
                    onClick={() =>
                      router.push(`/dashboard/course/${selectedTaskId}`)
                    }
                    style={{ justifyContent: "center" }}
                  >
                    Slides
                  </button>
                  <button
                    type="button"
                    className="btn"
                    onClick={() =>
                      router.push(`/dashboard/flashcard/${selectedTaskId}`)
                    }
                    style={{ justifyContent: "center" }}
                  >
                    Flashcards
                  </button>
                  <button
                    type="button"
                    className="btn"
                    onClick={() =>
                      router.push(`/dashboard/short-bit/${selectedTaskId}`)
                    }
                    style={{ justifyContent: "center" }}
                  >
                    Short Bits
                  </button>
                  <button
                    type="button"
                    className="btn"
                    onClick={() =>
                      router.push(`/dashboard/cheatsheet/${selectedTaskId}`)
                    }
                    style={{ justifyContent: "center" }}
                  >
                    Cheat Sheet
                  </button>
                  <button
                    type="button"
                    className="btn"
                    onClick={() =>
                      router.push(`/dashboard/story/${selectedTaskId}`)
                    }
                    style={{ justifyContent: "center" }}
                  >
                    Story
                  </button>
                </div>
              </>
            ) : (
              <div
                style={{
                  padding: "8px 12px",
                  border: "0.5px solid var(--watch)",
                  color: "var(--watch)",
                  fontFamily: "var(--mono)",
                  fontSize: 10,
                  letterSpacing: "0.1em",
                  marginBottom: 8,
                }}
              >
                ⊛ SCHEDULE TASK TO UNLOCK CONTENT
              </div>
            ))}

          {/* Prerequisites list */}
          {selectedNode.prerequisites.length > 0 && (
            <div>
              <div className="meta-text" style={{ marginBottom: 8 }}>
                Prerequisites
              </div>
              {selectedNode.prerequisites.map((prereqId) => {
                const prereq = nodes.find((n) => n.id === prereqId);
                if (!prereq) return null;
                return (
                  <div
                    key={prereqId}
                    style={{
                      padding: "8px 0",
                      borderBottom: "0.5px solid var(--rule)",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    <span style={{ fontSize: 11, fontFamily: "var(--mono)" }}>
                      {prereq.title}
                    </span>
                    <span
                      style={{
                        fontSize: 9,
                        fontFamily: "var(--mono)",
                        color: STATE_COLORS[prereq.nodeState],
                      }}
                    >
                      {STATE_ICONS[prereq.nodeState]}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
