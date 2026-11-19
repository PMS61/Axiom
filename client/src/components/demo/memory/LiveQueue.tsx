"use client";

import { useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  answer,
  ETA,
  FEATURE_NAMES,
  type Flashcard,
  predict,
  RIYA_CARDS,
  riskQueue,
  W0,
} from "@/lib/demo/online";
import styles from "./memory.module.css";

const WEIGHT_MIN = -1;
const WEIGHT_MAX = 2;
const ANIMATION_MS = 400;

type Flash = {
  n: number;
  remembered: boolean;
  title: string;
  deltas: Record<string, number>;
};

function weightPos(v: number): number {
  return ((v - WEIGHT_MIN) / (WEIGHT_MAX - WEIGHT_MIN)) * 100;
}

function formatPts(delta: number): string {
  const pts = Math.round(delta * 100);
  if (pts === 0) return "0 pts";
  return `${pts > 0 ? "+" : "−"}${Math.abs(pts)} pts`;
}

function statusText(f: Flash): string {
  const n = RIYA_CARDS.length;
  const changes = RIYA_CARDS.map((c) => f.deltas[c.id]);
  if (f.remembered) {
    const up = changes.filter((d) => d > 0).length;
    return up === n
      ? `You remembered '${f.title}'. Every card for Riya rose a little: the online layer learns she is on a good day.`
      : `You remembered '${f.title}'. ${up} of ${n} cards rose a little.`;
  }
  const down = changes.filter((d) => d < 0).length;
  return down === n
    ? `You forgot '${f.title}'. Every card for Riya dropped a little: the online layer learns she is having an off day.`
    : `You forgot '${f.title}'. ${down} of ${n} cards dropped a little.`;
}

export default function LiveQueue() {
  const [w, setW] = useState<number[]>([...W0]);
  const [flash, setFlash] = useState<Flash | null>(null);
  const queue = useMemo(() => riskQueue(RIYA_CARDS, w), [w]);

  const rowRefs = useRef<Map<string, HTMLLIElement>>(new Map());
  const pendingRects = useRef<Map<string, DOMRect> | null>(null);

  // FLIP: remember row positions before the state change, then animate from the old offset.
  // Runs after every render; it does nothing unless an answer was just made.
  useLayoutEffect(() => {
    const before = pendingRects.current;
    if (!before) return;
    pendingRects.current = null;
    rowRefs.current.forEach((el, id) => {
      const from = before.get(id);
      if (!from) return;
      const dy = from.top - el.getBoundingClientRect().top;
      if (Math.abs(dy) < 1) return;
      el.animate(
        [{ transform: `translateY(${dy}px)` }, { transform: "translateY(0)" }],
        { duration: ANIMATION_MS, easing: "ease" },
      );
    });
  });

  function onAnswer(card: Flashcard, remembered: boolean) {
    const before = new Map<string, DOMRect>();
    for (const [id, el] of rowRefs.current) {
      before.set(id, el.getBoundingClientRect());
    }
    pendingRects.current = before;

    const next = answer(w, card, remembered);
    const deltas: Record<string, number> = {};
    for (const c of RIYA_CARDS)
      deltas[c.id] = predict(next, c.x) - predict(w, c.x);
    setW(next);
    setFlash({
      n: (flash?.n ?? 0) + 1,
      remembered,
      title: card.front.replace(/[?.]$/, ""),
      deltas,
    });
  }

  function reset() {
    pendingRects.current = null;
    setW([...W0]);
    setFlash(null);
  }

  return (
    <section className={styles.panel} aria-label="Live queue with online layer">
      <h3 className={styles.panelTitle}>Riya's queue, riskiest first</h3>
      <p className={styles.small}>
        Transformers &amp; attention · P(recall) with 90% band (illustrative)
      </p>

      <ol className={styles.queue}>
        {queue.map(({ card, p }, i) => {
          const d = flash ? flash.deltas[card.id] : 0;
          const lo = Math.min(1, Math.max(0, card.bayesLow + (p - card.baseP)));
          const hi = Math.min(
            1,
            Math.max(0, card.bayesHigh + (p - card.baseP)),
          );
          return (
            <li
              key={card.id}
              ref={(el) => {
                if (el) rowRefs.current.set(card.id, el);
                else rowRefs.current.delete(card.id);
              }}
              className={styles.row}
            >
              <span className={styles.rank}>{i + 1}</span>
              <p className={styles.front}>{card.front}</p>
              <div className={styles.barWrap}>
                <div className={styles.barTrack}>
                  <div
                    className={styles.barFill}
                    style={{ width: `${p * 100}%` }}
                  />
                  <div
                    className={styles.band}
                    style={{
                      left: `${lo * 100}%`,
                      width: `${Math.max(0, hi - lo) * 100}%`,
                    }}
                    title={`illustrative 90% band ${Math.round(lo * 100)}–${Math.round(hi * 100)}%`}
                  />
                </div>
                <span className={styles.pct}>{Math.round(p * 100)}%</span>
                {flash && (
                  <span
                    key={flash.n}
                    className={`${styles.delta} ${d < 0 ? styles.deltaDown : styles.deltaUp}`}
                  >
                    {formatPts(d)}
                  </span>
                )}
              </div>
              <div className={styles.buttons}>
                <button
                  type="button"
                  className={styles.answer}
                  onClick={() => onAnswer(card, true)}
                  aria-label={`Remembered: ${card.front}`}
                >
                  ✓ Knew
                </button>
                <button
                  type="button"
                  className={styles.answer}
                  onClick={() => onAnswer(card, false)}
                  aria-label={`Forgot: ${card.front}`}
                >
                  ✗ Forgot
                </button>
              </div>
            </li>
          );
        })}
      </ol>

      <div className={styles.weights}>
        {FEATURE_NAMES.map((name, j) => {
          const w0 = W0[j];
          const lo = Math.min(w[j], w0);
          const hi = Math.max(w[j], w0);
          return (
            <div key={name} className={styles.weightRow}>
              <span>{name}</span>
              <div className={styles.weightTrack}>
                <div
                  className={styles.weightBar}
                  style={{
                    left: `${weightPos(lo)}%`,
                    width: `${weightPos(hi) - weightPos(lo)}%`,
                  }}
                />
                <div
                  className={styles.weightTick}
                  style={{ left: `${weightPos(w0)}%` }}
                />
              </div>
              <span className={styles.weightValue}>{w[j].toFixed(2)}</span>
            </div>
          );
        })}
      </div>
      <p className={styles.small}>
        Bars show each weight; the red tick marks its starting value w0.
      </p>

      <p className={styles.status} aria-live="polite">
        {flash
          ? statusText(flash)
          : "Mark a card as remembered or forgotten to see the online layer move every prediction."}
      </p>

      <div className={styles.footer}>
        <p className={styles.small}>
          Online layer: logit(p′) = logit(p) + w·x, one SGD step per answer (η ={" "}
          {ETA}, sped up for the demo).
        </p>
        <button type="button" className={styles.button} onClick={reset}>
          Reset
        </button>
      </div>
    </section>
  );
}
