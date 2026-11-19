"use client";

import { useState } from "react";
import { AGENTS, type RoadmapNode, roadmapFor, TOPICS } from "@/lib/demo/story";
import styles from "./demo.module.css";
import { useDrawn } from "./useDrawn";

interface Props {
  topicId: string;
}

const NODE_W = 180;
const NODE_H = 56;
const COL_X = [0, 240, 480, 720];
const ROW_Y = [40, 160, 280];
const CANVAS_W = 900;
const CANVAS_H = 350;

function nodeXY(n: RoadmapNode) {
  return { x: COL_X[n.col], y: ROW_Y[n.row] };
}

export default function StepRoadmap({ topicId }: Props) {
  const drawn = useDrawn();
  const roadmap = roadmapFor(topicId);
  const topic = TOPICS.find((t) => t.id === topicId);
  const [picked, setPicked] = useState<string | null>(null);
  const selected =
    roadmap.nodes.find((n) => n.id === picked) ??
    roadmap.nodes[1] ??
    roadmap.nodes[0];
  const byId = new Map(roadmap.nodes.map((n) => [n.id, n]));

  return (
    <section
      className={`${styles.step} ${drawn ? styles.drawn : ""}`}
      aria-labelledby="step-roadmap-title"
    >
      <header>
        <p className={styles.kicker}>Step 2 · Roadmap</p>
        <h1 id="step-roadmap-title" className={styles.headline}>
          How to learn {topic?.name ?? "this topic"}
        </h1>
        <p className={styles.caption}>
          Six nodes from prerequisites on the left to advanced on the right.
          Click a node to see what each agent produced.
        </p>
      </header>

      <div className={styles.roadmap}>
        <svg
          className={styles.dag}
          viewBox={`0 0 ${CANVAS_W} ${CANVAS_H}`}
          role="img"
          aria-label={`Roadmap DAG for ${topic?.name ?? topicId}`}
        >
          <defs>
            <marker
              id="roadmap-arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M0,0 L10,5 L0,10 z" style={{ fill: "var(--ink)" }} />
            </marker>
          </defs>

          <text x={0} y={20} className={styles.dagLabel}>
            Prerequisites
          </text>
          <text x={COL_X[1]} y={20} className={styles.dagLabel}>
            Core
          </text>
          <text x={COL_X[3]} y={20} className={styles.dagLabel}>
            Advanced
          </text>

          {roadmap.edges.map((e) => {
            const a = byId.get(e.from);
            const b = byId.get(e.to);
            if (!a || !b) return null;
            const p = nodeXY(a);
            const q = nodeXY(b);
            const x1 = p.x + NODE_W;
            const y1 = p.y + NODE_H / 2;
            const x2 = q.x;
            const y2 = q.y + NODE_H / 2;
            const mx = (x1 + x2) / 2;
            const hot = selected.id === e.from || selected.id === e.to;
            return (
              <path
                key={`${e.from}-${e.to}`}
                pathLength={1}
                d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`}
                markerEnd="url(#roadmap-arrow)"
                className={`${styles.edge} ${styles.draw} ${drawn ? styles.on : ""} ${hot ? styles.edgeOn : ""}`}
              />
            );
          })}

          {roadmap.nodes.map((n) => {
            const { x, y } = nodeXY(n);
            const on = n.id === selected.id;
            return (
              // biome-ignore lint/a11y/useSemanticElements: an SVG node cannot be a <button>
              <g
                key={n.id}
                role="button"
                tabIndex={0}
                aria-pressed={on}
                className={`${styles.node} ${on ? styles.nodeOn : ""}`}
                onClick={() => setPicked(n.id)}
                onKeyDown={(ev) => {
                  if (ev.key === "Enter" || ev.key === " ") {
                    ev.preventDefault();
                    setPicked(n.id);
                  }
                }}
              >
                <rect
                  x={x}
                  y={y}
                  width={NODE_W}
                  height={NODE_H}
                  rx={4}
                  className={styles.nodeBox}
                />
                <text
                  x={x + 14}
                  y={y + NODE_H / 2 + 5}
                  className={styles.nodeText}
                >
                  {n.label}
                </text>
              </g>
            );
          })}
        </svg>

        <aside className={styles.panel} aria-live="polite">
          <p className={styles.panelKicker}>Node</p>
          <h2 className={styles.panelTitle}>{selected.label}</h2>
          <ul className={styles.agents}>
            {AGENTS.map((a) => (
              <li key={a.id} className={styles.agent}>
                <span className={styles.agentName}>{a.label}</span>
                <span>{selected.outputs[a.id]}</span>
              </li>
            ))}
          </ul>
        </aside>
      </div>
    </section>
  );
}
