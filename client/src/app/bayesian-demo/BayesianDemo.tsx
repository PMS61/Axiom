"use client";

import { useEffect, useState } from "react";
import {
  type Fit,
  initialHistories,
  makeQueue,
  topics,
} from "@/lib/bayesian/demo";
import styles from "./demo.module.css";

const percent = (n: number) => `${Math.round(n * 100)}%`;
const clock = (minutes: number) =>
  `${String(9 + Math.floor(minutes / 60)).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}`;

async function fitHistory(
  histories: number[][],
  signal?: AbortSignal,
): Promise<Fit> {
  const response = await fetch("/api/bayesian-demo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(histories),
    signal,
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Could not fit model.");
  return result;
}

export default function BayesianDemo() {
  const [histories, setHistories] = useState(initialHistories);
  const [fit, setFit] = useState<Fit | null>(null);
  const [budget, setBudget] = useState(60);
  const [selected, setSelected] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState(
    "Start with the topic at greatest predicted recall risk.",
  );

  useEffect(() => {
    const controller = new AbortController();
    fitHistory(initialHistories, controller.signal)
      .then(setFit)
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setBusy(false);
      });
    return () => controller.abort();
  }, []);

  async function update(next: number[][], feedback?: number) {
    setBusy(true);
    setError("");
    const before = fit?.items[selected].recall;
    try {
      const result = await fitHistory(next);
      setHistories(next);
      setFit(result);
      setRevealed(false);
      setNotice(
        feedback === undefined
          ? "Demo reset to the original synthetic histories."
          : `${topics[selected].title}: ${feedback ? "remembered" : "forgot"}. Next-review prediction ${percent(before ?? 0)} → ${percent(result.items[selected].recall)}. Queue recomputed.`,
      );
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Could not fit model. Try again.",
      );
    } finally {
      setBusy(false);
    }
  }

  const queue = fit ? makeQueue(fit.items, budget) : [];
  const baseline = fit ? makeQueue(fit.items, budget, false) : [];
  const used = queue.reduce((sum, item) => sum + topics[item.id].minutes, 0);
  const prediction = fit?.items[selected];
  let elapsed = 0;

  return (
    <main className={styles.workbench}>
      <header className={styles.header}>
        <a href="/" className={styles.brand}>
          Axiom<span> / research lab</span>
        </a>
        <span className={styles.badge}>Bayesian scheduler · prototype 01</span>
      </header>
      <section className={styles.hero}>
        <div>
          <p className={styles.eyebrow}>A study plan that listens.</p>
          <h1>
            Make room for
            <br />
            <em>what you might forget.</em>
          </h1>
          <p className={styles.intro}>
            Review history becomes a belief about recall. That belief becomes a
            study queue. Your next answer changes the plan.
          </p>
        </div>
        <aside className={styles.heroNote}>
          <span className={styles.bigNumber}>06</span>
          <p>
            learning topics
            <br />
            one evolving plan
          </p>
          <span className={styles.dot} /> Live Python model · synthetic
          histories
        </aside>
      </section>
      <div className={styles.toolbar}>
        <label htmlFor="budget">
          Study window <strong>{budget} min</strong>
        </label>
        <input
          id="budget"
          type="range"
          min="30"
          max="120"
          step="15"
          value={budget}
          onChange={(e) => setBudget(Number(e.target.value))}
        />
        <span>Starts 09:00 · continuous availability</span>
        <button
          type="button"
          disabled={busy}
          onClick={() => update(initialHistories)}
        >
          Reset demo
        </button>
      </div>
      {error && (
        <div className={styles.error} role="alert">
          {error}{" "}
          <button
            type="button"
            disabled={busy}
            onClick={() => update(histories)}
          >
            Retry fit
          </button>
        </div>
      )}
      <output className={styles.status}>
        {busy
          ? "Fitting the Bayesian models… controls return when the posterior is ready."
          : notice}
      </output>
      <div className={styles.mainGrid}>
        <section className={styles.beliefs} aria-labelledby="beliefs-title">
          <div className={styles.sectionHeading}>
            <span>01 / Belief</span>
            <h2 id="beliefs-title">What needs attention?</h2>
          </div>
          <p className={styles.muted}>
            Next-review recall probability. Select a topic to test your memory.
          </p>
          <div className={styles.legend}>
            <span>● Dynamic mean</span>
            <span>│ 90% predictive interval</span>
            <span>◇ Static model</span>
          </div>
          <div className={styles.axis}>
            <span>Less likely to recall</span>
            <span>More likely →</span>
          </div>
          {topics.map((topic, i) => {
            const item = fit?.items[i];
            return (
              <button
                key={topic.title}
                type="button"
                className={`${styles.topic} ${selected === i ? styles.selected : ""}`}
                onClick={() => {
                  setSelected(i);
                  setRevealed(false);
                }}
                disabled={busy}
                aria-pressed={selected === i}
              >
                <div className={styles.topicTitle}>
                  <strong>{topic.title}</strong>
                  <span>{item ? percent(item.recall) : "…"}</span>
                </div>
                <p>{topic.detail}</p>
                <div className={styles.plot} aria-hidden="true">
                  {item && (
                    <>
                      <span
                        className={styles.interval}
                        style={{
                          left: percent(item.low),
                          width: percent(item.high - item.low),
                        }}
                      />
                      <span
                        className={styles.staticMark}
                        style={{ left: percent(item.static) }}
                      >
                        ◇
                      </span>
                      <span
                        className={styles.mean}
                        style={{ left: percent(item.recall) }}
                      />
                    </>
                  )}
                </div>
                <span className={styles.history}>
                  {histories[i].map((r, index) => (
                    <span
                      key={`${index}-${r}`}
                      className={r ? styles.remembered : styles.forgot}
                      title={`Review ${index + 1}: ${r ? "remembered" : "forgot"}`}
                    >
                      {r ? "·" : "×"}
                    </span>
                  ))}
                  <span>{histories[i].length} reviews</span>
                </span>
                {item && (
                  <span className={styles.srOnly}>
                    90% predictive interval {percent(item.low)} to{" "}
                    {percent(item.high)}. Static reference{" "}
                    {percent(item.static)}.
                  </span>
                )}
              </button>
            );
          })}
        </section>
        <div className={styles.rightColumn}>
          <section className={styles.plan} aria-labelledby="plan-title">
            <div className={styles.sectionHeading}>
              <span>02 / Plan</span>
              <h2 id="plan-title">Your next {budget} minutes.</h2>
            </div>
            <div className={styles.planSummary}>
              <strong>
                {queue.length} {queue.length === 1 ? "topic" : "topics"}
              </strong>
              <span>
                {used} min planned · {budget - used} min free
              </span>
            </div>
            {queue.map((item, index) => {
              const start = elapsed;
              elapsed += topics[item.id].minutes;
              return (
                <button
                  type="button"
                  className={styles.queueItem}
                  key={item.id}
                  disabled={busy}
                  onClick={() => {
                    setSelected(item.id);
                    setRevealed(false);
                  }}
                >
                  <span className={styles.queueNumber}>0{index + 1}</span>
                  <div>
                    <strong>{topics[item.id].title}</strong>
                    <p>
                      {clock(start)}–{clock(elapsed)} ·{" "}
                      {topics[item.id].minutes} min
                    </p>
                    <small>
                      {percent(1 - item.recall)} predicted recall risk
                    </small>
                  </div>
                  <span>↗</span>
                </button>
              );
            })}
            <details className={styles.baseline}>
              <summary>Compare with fixed-order scheduling</summary>
              <p>
                Same topics, durations and window. Original topic order; no
                learned recall signal. Both use whole sessions and skip a topic
                if it does not fit.
              </p>
              <ol>
                {baseline.map((item) => (
                  <li key={item.id}>
                    {topics[item.id].title} · {topics[item.id].minutes} min
                  </li>
                ))}
              </ol>
              <p>
                This is a scheduling illustration, not a measured effectiveness
                comparison.
              </p>
            </details>
          </section>
          <section className={styles.review} aria-labelledby="review-title">
            <div className={styles.sectionHeading}>
              <span>03 / Adapt</span>
              <h2 id="review-title">Try a recall check.</h2>
            </div>
            <p className={styles.eyebrow}>{topics[selected].title}</p>
            <h3>{topics[selected].question}</h3>
            {revealed ? (
              <p className={styles.answer}>{topics[selected].answer}</p>
            ) : (
              <button
                type="button"
                className={styles.reveal}
                onClick={() => setRevealed(true)}
                disabled={busy}
              >
                Reveal answer ↗
              </button>
            )}
            <div className={styles.feedback}>
              <button
                type="button"
                disabled={busy || !fit || histories[selected].length >= 40}
                onClick={() =>
                  update(
                    histories.map((h, i) => (i === selected ? [...h, 0] : h)),
                    0,
                  )
                }
              >
                I forgot
              </button>
              <button
                type="button"
                disabled={busy || !fit || histories[selected].length >= 40}
                onClick={() =>
                  update(
                    histories.map((h, i) => (i === selected ? [...h, 1] : h)),
                    1,
                  )
                }
              >
                I remembered
              </button>
            </div>
            <p className={styles.muted}>
              {histories[selected].length >= 40
                ? "40-review demo limit reached. Reset to start again."
                : "Self-reported feedback refits the model and rebuilds the queue. Changes last for this page session."}
            </p>
            {prediction && (
              <p className={styles.muted}>
                Current prediction {percent(prediction.recall)} · static
                reference {percent(prediction.static)}
              </p>
            )}
          </section>
        </div>
      </div>
      <footer className={styles.method}>
        <div>
          <p className={styles.eyebrow}>Under the surface</p>
          <h2>History → belief → plan → feedback.</h2>
          <p>
            Uses this branch’s hierarchical dynamic retention model: Pólya–Gamma
            Gibbs sampling and forward-filtering backward-sampling. The chart
            forecasts one review event ahead, integrating state, parameter and
            process uncertainty.
          </p>
        </div>
        <div>
          <p>
            {fit
              ? `${fit.observations} observations · ${fit.draws} posterior draws · seed ${fit.seed} · mean persistence ${fit.phi.toFixed(2)}`
              : "Preparing posterior draws…"}
          </p>
          <p>
            Prototype limits: synthetic data, one learner, a short single chain
            and approximate Pólya–Gamma sampling. The model does not use elapsed
            time. Risk-first placement is a heuristic, not an optimal scheduler.
            Wide intervals reflect uncertainty; no learning benefit is claimed.
          </p>
        </div>
      </footer>
    </main>
  );
}
