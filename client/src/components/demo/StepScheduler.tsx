"use client";

import { useState } from "react";
import {
  BANDWIDTH,
  type Block,
  COMMITMENTS,
  DAY_END,
  DAY_START,
  EXCLUSIONS,
  formatTime,
  LOW_WINDOWS,
  PEAK_WINDOWS,
  TASKS,
} from "@/lib/demo/story";
import styles from "./demo.module.css";
import { useDrawn } from "./useDrawn";

const W = 1200;
const H = 250;
const PLOT_X = 120;
const PLOT_W = W - PLOT_X;
const CURVE_BASE = 100;
const CURVE_TOP = 10;
const ROW_COMMIT = 120;
const ROW_EXCL = 150;
const ROW_TASK = 180;
const ROW_H = 22;
const TASK_H = 30;
const AXIS_Y = 222;

function x(minute: number): number {
  return PLOT_X + ((minute - DAY_START) / (DAY_END - DAY_START)) * PLOT_W;
}

function y(value: number): number {
  return CURVE_BASE - value * (CURVE_BASE - CURVE_TOP);
}

const curvePath = BANDWIDTH.map(
  ([m, v], i) => `${i === 0 ? "M" : "L"}${x(m).toFixed(1)},${y(v).toFixed(1)}`,
).join(" ");
const areaPath = `${curvePath} L${x(BANDWIDTH[BANDWIDTH.length - 1][0]).toFixed(1)},${CURVE_BASE} L${x(BANDWIDTH[0][0]).toFixed(1)},${CURVE_BASE} Z`;

const HOURS = Array.from({ length: 16 }, (_, i) => 8 + i); // 08:00 .. 23:00

function BlockRect({
  block,
  row,
  className,
  label,
}: {
  block: Block;
  row: number;
  className: string;
  label?: string;
}) {
  const x0 = x(block.start);
  const w = x(block.end) - x0;
  return (
    <g>
      <rect x={x0} y={row} width={w} height={ROW_H} className={className} />
      {label ? (
        <text x={x0 + w + 6} y={row + 15} className={styles.tlText}>
          {label}
        </text>
      ) : null}
    </g>
  );
}

export default function StepScheduler() {
  const drawn = useDrawn();
  const [picked, setPicked] = useState(
    Math.max(
      0,
      TASKS.findIndex((t) => t.id === "course"),
    ),
  );
  const task = TASKS[picked] ?? TASKS[0];

  return (
    <section
      className={`${styles.step} ${drawn ? styles.drawn : ""}`}
      aria-labelledby="step-scheduler-title"
    >
      <header>
        <p className={styles.kicker}>Step 3 · Scheduler</p>
        <h1 id="step-scheduler-title" className={styles.headline}>
          When to fit it in
        </h1>
        <p className={styles.caption}>
          The scheduler places study in Riya's peak hours and keeps her
          commitments clear. Open any task to see why it sits where it does.
        </p>
      </header>

      <svg
        className={styles.timeline}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Today's timeline from 07:30 to 23:30"
      >
        {PEAK_WINDOWS.map((w) => (
          <rect
            key={w.id}
            x={x(w.start)}
            y={0}
            width={x(w.end) - x(w.start)}
            height={CURVE_BASE + 8}
            className={styles.tlPeak}
          />
        ))}
        {LOW_WINDOWS.map((w) => (
          <rect
            key={w.id}
            x={x(w.start)}
            y={0}
            width={x(w.end) - x(w.start)}
            height={CURVE_BASE + 8}
            className={styles.tlLow}
          />
        ))}

        <text x={0} y={56} className={styles.tlLabel}>
          Bandwidth
        </text>
        <path
          d={areaPath}
          className={`${styles.tlArea} ${styles.fade} ${drawn ? styles.on : ""}`}
        />
        <path
          d={curvePath}
          pathLength={1}
          className={`${styles.tlCurve} ${styles.draw} ${drawn ? styles.on : ""}`}
        />

        <text x={0} y={ROW_COMMIT + 15} className={styles.tlLabel}>
          Commitments
        </text>
        {COMMITMENTS.map((c) => (
          <BlockRect
            key={c.id}
            block={c}
            row={ROW_COMMIT}
            className={styles.tlCommit}
            label={`${c.label} ${formatTime(c.start)}`}
          />
        ))}

        <text x={0} y={ROW_EXCL + 15} className={styles.tlLabel}>
          Exclusions
        </text>
        {EXCLUSIONS.map((c) => (
          <BlockRect
            key={c.id}
            block={c}
            row={ROW_EXCL}
            className={styles.tlExcl}
            label={c.label}
          />
        ))}

        <text x={0} y={ROW_TASK + 19} className={styles.tlLabel}>
          Tasks
        </text>
        {TASKS.map((t, i) => {
          const x0 = x(t.start);
          const w = x(t.end) - x0;
          const on = i === picked;
          return (
            // biome-ignore lint/a11y/useSemanticElements: an SVG node cannot be a <button>
            <g
              key={t.id}
              role="button"
              tabIndex={0}
              aria-label={`${t.label}, ${formatTime(t.start)} to ${formatTime(t.end)}`}
              aria-pressed={on}
              onClick={() => setPicked(i)}
              onKeyDown={(ev) => {
                if (ev.key === "Enter" || ev.key === " ") {
                  ev.preventDefault();
                  setPicked(i);
                }
              }}
              style={{ cursor: "pointer" }}
            >
              <rect
                x={x0}
                y={ROW_TASK}
                width={w}
                height={TASK_H}
                className={t.fromMemory ? styles.tlTaskMemory : styles.tlTask}
              />
              {on ? (
                <rect
                  x={x0 - 3}
                  y={ROW_TASK - 3}
                  width={w + 6}
                  height={TASK_H + 6}
                  className={styles.tlSelected}
                />
              ) : null}
              <text
                x={x0 + w / 2}
                y={ROW_TASK + 20}
                textAnchor="middle"
                className={styles.tlTaskNum}
              >
                {i + 1}
              </text>
            </g>
          );
        })}

        <line
          x1={PLOT_X}
          x2={W}
          y1={AXIS_Y}
          y2={AXIS_Y}
          style={{ stroke: "var(--rule)" }}
        />
        {HOURS.map((h) => (
          <text
            key={h}
            x={x(h * 60)}
            y={AXIS_Y + 18}
            textAnchor="middle"
            className={styles.tlTick}
          >
            {formatTime(h * 60)}
          </text>
        ))}
      </svg>

      <ul className={styles.legend}>
        <li>
          <span
            className={styles.swatch}
            style={{
              background: "color-mix(in srgb, var(--ink) 9%, transparent)",
            }}
          />
          Peak window
        </li>
        <li>
          <span
            className={styles.swatch}
            style={{ background: "var(--rule)" }}
          />
          Low window
        </li>
        <li>
          <span
            className={styles.swatch}
            style={{ background: "var(--ink)" }}
          />
          Commitment
        </li>
        <li>
          <span
            className={styles.swatch}
            style={{ background: "var(--rule)" }}
          />
          Exclusion
        </li>
        <li>
          <span
            className={styles.swatch}
            style={{ background: "var(--ink)" }}
          />
          Study task
        </li>
        <li>
          <span
            className={styles.swatch}
            style={{
              background: "var(--vermillion)",
              borderColor: "var(--vermillion)",
            }}
          />
          Task from memory engine
        </li>
        <li>Bandwidth curve is an illustrative profile</li>
      </ul>

      <div className={styles.lower}>
        <ul className={styles.taskList}>
          {TASKS.map((t, i) => (
            <li key={t.id}>
              <button
                type="button"
                className={`${styles.taskBtn} ${i === picked ? styles.taskBtnOn : ""}`}
                aria-pressed={i === picked}
                onClick={() => setPicked(i)}
              >
                <span className={styles.taskTime}>
                  {formatTime(t.start)}–{formatTime(t.end)}
                </span>
                <span>
                  {i + 1}. {t.label}
                </span>
                {t.fromMemory ? (
                  <span className={styles.badge}>from memory engine →</span>
                ) : (
                  <span />
                )}
              </button>
            </li>
          ))}
        </ul>

        <div className={styles.reason} aria-live="polite">
          <p className={styles.panelKicker}>
            Reasoning · {formatTime(task.start)}–{formatTime(task.end)}
          </p>
          <h2 className={styles.panelTitle} style={{ fontSize: 22 }}>
            {task.label}
          </h2>
          <ol className={styles.reasonSteps}>
            {task.reasoning.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  );
}
