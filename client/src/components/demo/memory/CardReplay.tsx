"use client";

import { useEffect, useState } from "react";
import seed from "@/lib/demo/seed/online.json";
import styles from "./memory.module.css";

type ReplayRow = {
  review: number;
  day: number;
  rating: number;
  y: number;
  scored: boolean;
  pOnline?: number;
  pGbdt?: number;
  pFsrsDefault?: number;
};

type ReplayCard = {
  label: string;
  pickedFor: string;
  nScored: number;
  reviews: ReplayRow[];
};

type OnlineSeed = { cards: ReplayCard[]; fsrsNote: string };

const data = seed as unknown as OnlineSeed;

const CLIP = 1e-6;
const STEP_MS = 250;
const W = 640;
const H = 272;
const PL = 52;
const PR = 36;
const PT = 24; // y of probability 100%
const PB = 196; // y of probability 0%
const TOP_MARK_Y = 12;
const BOTTOM_MARK_Y = 214;
const STRIP_Y = 228;
const TICK_Y = 258;

const FSRS_COLOR = "var(--muted)";
const OURS_COLOR = "var(--vermillion)";

function logLoss(y: number, p: number): number {
  const q = Math.min(Math.max(p, CLIP), 1 - CLIP);
  return -(y * Math.log(q) + (1 - y) * Math.log(1 - q));
}

function need(v: number | undefined): number {
  if (v === undefined) throw new Error("scored review without a prediction");
  return v;
}

function mean(xs: number[]): number {
  return xs.reduce((s, x) => s + x, 0) / xs.length;
}

export default function CardReplay() {
  const [index, setIndex] = useState(0);
  const card = data.cards[index];
  const scored = card.reviews.filter((r) => r.scored);
  const history = card.reviews.filter((r) => !r.scored);
  const total = scored.length;

  // Points revealed so far (replay animates this from 0 up to total).
  const [shown, setShown] = useState(total);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    if (!running) return;
    if (shown >= total) {
      setRunning(false);
      return;
    }
    const timer = setTimeout(() => setShown((s) => s + 1), STEP_MS);
    return () => clearTimeout(timer);
  }, [running, shown, total]);

  function selectCard(i: number) {
    setIndex(i);
    setShown(0);
    setRunning(true);
  }

  function replay() {
    setShown(0);
    setRunning(true);
  }

  const fsrsLL = mean(scored.map((r) => logLoss(r.y, need(r.pFsrsDefault))));
  const oursLL = mean(scored.map((r) => logLoss(r.y, need(r.pOnline))));
  const gap = fsrsLL - oursLL;

  const days = card.reviews.map((r) => r.day);
  const d0 = Math.min(...days);
  const d1 = Math.max(...days);
  const xOf = (day: number) =>
    PL + ((day - d0) / (d1 - d0 || 1)) * (W - PL - PR);
  const yOf = (p: number) => PB - p * (PB - PT);

  const visible = scored.slice(0, shown);
  const fsrsPts = visible.map(
    (r) => `${xOf(r.day)},${yOf(need(r.pFsrsDefault))}`,
  );
  const oursPts = visible.map((r) => `${xOf(r.day)},${yOf(need(r.pOnline))}`);
  const ticks = [0, 1, 2, 3, 4].map((k) =>
    Math.round(d0 + (k * (d1 - d0)) / 4),
  );

  return (
    <section className={styles.panel} aria-label="Real card replay">
      <h3 className={styles.panelTitle}>A real card, replayed</h3>
      <p className={styles.small}>
        Each dot is one scored review from an unseen srs-benchmark user, at the
        day it happened.
      </p>

      <div className={styles.tabs} role="tablist" aria-label="Choose a card">
        {data.cards.map((c, i) => (
          <button
            key={c.label}
            type="button"
            role="tab"
            className={styles.tab}
            aria-selected={i === index}
            onClick={() => selectCard(i)}
          >
            {c.label}
            {c.pickedFor === "fsrs_wins" ? " · where FSRS wins" : ""}
          </button>
        ))}
      </div>

      <svg
        className={styles.svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${card.label}: predicted recall over time against the actual outcome`}
      >
        {[0, 0.5, 1].map((p) => (
          <g key={p}>
            <line
              x1={PL}
              x2={W - PR}
              y1={yOf(p)}
              y2={yOf(p)}
              style={{ stroke: "var(--rule)" }}
              strokeWidth={1}
            />
            <text
              className={styles.svgText}
              x={PL - 6}
              y={yOf(p) + 4}
              textAnchor="end"
            >
              {Math.round(p * 100)}%
            </text>
          </g>
        ))}
        <line
          x1={PL}
          x2={W - PR}
          y1={PB}
          y2={PB}
          style={{ stroke: "var(--ink)" }}
          strokeWidth={1}
        />

        {/* earlier, unscored history as a faint strip */}
        {history.map((r) => (
          <rect
            key={`h-${r.review}`}
            x={xOf(r.day) - 1}
            y={STRIP_Y}
            width={2}
            height={10}
            style={{ fill: "var(--ink)" }}
            opacity={0.22}
          />
        ))}

        {fsrsPts.length > 1 && (
          <polyline
            points={fsrsPts.join(" ")}
            fill="none"
            style={{ stroke: FSRS_COLOR }}
            strokeWidth={1.5}
          />
        )}
        {oursPts.length > 1 && (
          <polyline
            points={oursPts.join(" ")}
            fill="none"
            style={{ stroke: OURS_COLOR }}
            strokeWidth={2}
          />
        )}

        {visible.map((r) => (
          <g key={`s-${r.review}`}>
            <circle
              cx={xOf(r.day)}
              cy={yOf(need(r.pFsrsDefault))}
              r={4}
              style={{ fill: FSRS_COLOR }}
            />
            <circle
              cx={xOf(r.day)}
              cy={yOf(need(r.pOnline))}
              r={4}
              style={{ fill: OURS_COLOR }}
            />
            <text
              x={xOf(r.day)}
              y={r.y === 1 ? TOP_MARK_Y : BOTTOM_MARK_Y}
              textAnchor="middle"
              fontSize={15}
              style={{ fill: "var(--ink)" }}
            >
              {r.y === 1 ? "✓" : "✗"}
            </text>
          </g>
        ))}

        {ticks.map((t) => (
          <text
            key={`t-${t}`}
            className={styles.svgText}
            x={xOf(t)}
            y={TICK_Y}
            textAnchor="middle"
          >
            day {t}
          </text>
        ))}
      </svg>

      <div className={styles.legend}>
        <span>
          <span className={styles.swatch} style={{ background: FSRS_COLOR }} />
          FSRS-6 (default weights)
        </span>
        <span>
          <span className={styles.swatch} style={{ background: OURS_COLOR }} />
          Ours (GBDT + online)
        </span>
        <span>✓ recalled · ✗ forgot (actual)</span>
        <span>faint ticks: earlier reviews, not scored</span>
      </div>

      <div className={styles.tally}>
        <span>FSRS-6 log loss, {total} scored reviews</span>
        <span className={styles.tallyValue}>{fsrsLL.toFixed(3)}</span>
        <span>Ours log loss, same reviews</span>
        <span className={styles.tallyValue}>{oursLL.toFixed(3)}</span>
        <span className={styles.verdict}>
          {Math.abs(gap) < 0.0005
            ? "Tie on this card."
            : gap > 0
              ? `Ours lower by ${gap.toFixed(3)} on this card.`
              : `FSRS lower by ${(-gap).toFixed(3)} on this card.`}
        </span>
      </div>

      <div className={styles.replayRow}>
        <button
          type="button"
          className={styles.button}
          onClick={replay}
          disabled={running}
        >
          {running ? "Replaying…" : "Replay"}
        </button>
        <span className={styles.small}>
          {shown} of {total} scored reviews shown
        </span>
      </div>

      <p className={styles.small}>{data.fsrsNote}</p>
    </section>
  );
}
