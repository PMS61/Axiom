"use client";

import { useEffect, useState } from "react";
import Header from "@/components/Header";
import s from "./learn.module.css";
import {
  initialStudy,
  nextReviewDay,
  recordAnswer,
  replay,
  restoreStudy,
  reviewQueue,
  saveStudy,
} from "./scheduled-reviews";

const storageKey = "axiom-spanish-study";
const firstPractice = Math.max(
  ...replay.cards.flatMap((card) =>
    card.practice.map((entry) => entry.daysAgo),
  ),
);

function relativeDate(offset: number) {
  if (offset === 0) return "Today";
  if (offset === 1) return "Tomorrow";
  if (offset === -1) return "Yesterday";
  return offset < 0 ? `${-offset} days ago` : `In ${offset} days`;
}

export default function ScheduledLearning() {
  const [state, setState] = useState(initialStudy);
  const [loaded, setLoaded] = useState(false);
  const [revealed, setRevealed] = useState(false);
  const [historyWord, setHistoryWord] = useState<string>();
  const [notice, setNotice] = useState("");
  const [storageUnavailable, setStorageUnavailable] = useState(false);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(storageKey);
      if (saved) setState(restoreStudy(saved));
    } catch {
      setNotice("Started from the original study history.");
    }
    setLoaded(true);
  }, []);

  useEffect(() => {
    if (!loaded) return;
    try {
      localStorage.setItem(storageKey, saveStudy(state));
    } catch {
      setStorageUnavailable(true);
    }
  }, [state, loaded]);

  const queue = reviewQueue(state);
  const due = queue.filter((item) => item.canReview && item.due === state.day);
  const card = due[0];
  const countToday = state.answers.filter(
    ({ day }) => day === state.day,
  ).length;
  const next = nextReviewDay(state);
  const lastPractice = state.answers.at(-1)?.day ?? 0;
  const historyCard =
    queue.find((item) => item.word === historyWord) ?? card ?? queue[0];
  const historyEntries = [
    ...historyCard.practice.map((entry) => ({
      day: -entry.daysAgo,
      remembered: entry.recalled,
    })),
    ...historyCard.answers.map((entry) => ({
      day: entry.day,
      remembered: entry.answer === "remembered",
    })),
  ].reverse();

  function answer(result: "remembered" | "missed") {
    if (!card || !loaded || !revealed) return;
    const branch = card.plan[result];
    if (!branch) return;

    setState((previous) => recordAnswer(previous, card.word, result));
    setNotice(
      `${card.word} · Next review ${relativeDate(branch.interval).toLowerCase()}.`,
    );
    setHistoryWord(card.word);
    setRevealed(false);
  }

  function advance() {
    if (next === undefined || next <= state.day || !loaded) return;
    setState((previous) => ({ ...previous, day: next }));
    setNotice("");
    setHistoryWord(undefined);
  }

  return (
    <>
      <Header study />
      <main className={`${s.app} ${s.scheduledApp}`} id="study">
        <section className={s.intro} aria-labelledby="study-title">
          <div>
            <p className="meta-text">Your study desk</p>
            <h1 id="study-title">Spanish vocabulary</h1>
            <p>
              Last practice{" "}
              {relativeDate(lastPractice - state.day).toLowerCase()}.
            </p>
          </div>
          <div>
            <p className="meta-text">Study calendar</p>
            <strong className={s.studyDay}>
              Day {firstPractice + 1 + state.day}
            </strong>
          </div>
        </section>

        <section className={s.stats} aria-label="Study progress">
          <div>
            <strong>{due.length}</strong>
            <span>due today</span>
          </div>
          <div>
            <strong>{countToday}</strong>
            <span>reviewed today</span>
          </div>
          <div>
            <strong>{queue.length}</strong>
            <span>words in this deck</span>
          </div>
          <div>
            <strong>
              {queue.reduce(
                (total, item) =>
                  total + item.practice.length + item.answers.length,
                0,
              )}
            </strong>
            <span>practice records</span>
          </div>
        </section>

        <div className={s.layout}>
          <aside
            className={s.deck}
            id="schedule"
            aria-labelledby="schedule-title"
          >
            <h2 id="schedule-title">Review schedule</h2>
            <ul className={s.scheduleList}>
              {queue.map((item) => (
                <li key={item.word}>
                  <button
                    type="button"
                    aria-pressed={historyCard.word === item.word}
                    aria-controls="word-history"
                    onClick={() => setHistoryWord(item.word)}
                  >
                    <span lang="es">{item.word}</span>
                    <small>
                      {item.due <= state.day && !item.canReview
                        ? "Replay complete"
                        : relativeDate(item.due - state.day)}
                    </small>
                  </button>
                </li>
              ))}
            </ul>
          </aside>

          <section
            className={s.session}
            aria-label="Flashcard session"
            aria-busy={!loaded}
          >
            <div className={s.sessionTop}>
              <span>DAILY REVIEW</span>
              <span>
                {due.length ? `${due.length} remaining` : "All caught up"}
              </span>
            </div>
            <div className={s.progressTrack} aria-hidden="true">
              <div
                style={{
                  width: `${(countToday / Math.max(1, countToday + due.length)) * 100}%`,
                }}
              />
            </div>
            {card ? (
              <div className={s.flashcard} key={card.word}>
                <span className={s.questionLabel}>
                  What does this word mean?
                </span>
                <h2 lang="es">{card.word}</h2>
                {!revealed ? (
                  <button
                    type="button"
                    className={s.primary}
                    disabled={!loaded}
                    onClick={() => setRevealed(true)}
                  >
                    Reveal answer
                  </button>
                ) : (
                  <>
                    <div className={s.answer}>
                      <p>{card.meaning}</p>
                    </div>
                    <p className={s.ratingPrompt}>Did you remember it?</p>
                    <div className={s.answerChoices}>
                      <button
                        type="button"
                        className="btn"
                        onClick={() => answer("missed")}
                      >
                        I forgot
                      </button>
                      <button
                        type="button"
                        className={`btn btn-primary ${s.rememberButton}`}
                        onClick={() => answer("remembered")}
                      >
                        I remembered
                      </button>
                    </div>
                  </>
                )}
              </div>
            ) : (
              <div className={s.complete}>
                <h2>
                  {next === undefined
                    ? "Study replay complete."
                    : "You’re all caught up."}
                </h2>
                {next !== undefined ? (
                  <>
                    <p>
                      Your next review is{" "}
                      {relativeDate(next - state.day).toLowerCase()}.
                    </p>
                    <button
                      type="button"
                      className={s.primary}
                      disabled={!loaded}
                      onClick={advance}
                    >
                      Go to next review day →
                    </button>
                  </>
                ) : (
                  <p>You’ve completed the interactive reviews for this deck.</p>
                )}
              </div>
            )}
            {notice && <output className={s.notice}>{notice}</output>}
          </section>

          <aside
            className={s.insights}
            id="history"
            aria-labelledby="history-title"
          >
            <h2 id="history-title">Practice history</h2>
            <div id="word-history" aria-live="polite">
              <h3 lang="es">{historyCard.word}</h3>
              <ul className={s.practiceList}>
                {historyEntries.slice(0, 4).map((entry, index) => (
                  <li key={`${entry.day}-${index}`}>
                    <span>{relativeDate(entry.day - state.day)}</span>
                    <span data-recalled={entry.remembered}>
                      {entry.remembered ? "Remembered" : "Missed"}
                    </span>
                  </li>
                ))}
              </ul>
              <details className={s.method} key={historyCard.word}>
                <summary>Why this review date?</summary>
                <p>
                  At the scheduled review, Axiom estimates a{" "}
                  {Math.round(historyCard.plan.recall * 100)}% chance of
                  remembering this word. It aims for at least{" "}
                  {replay.target * 100}% recall, schedules the next day if the
                  estimate is already below target, and looks at most{" "}
                  {historyCard.forecast.length} days ahead.
                </p>
                {historyCard.answers.length === 0 && (
                  <p>
                    A fixed weekly plan would review this word seven days after
                    the recorded practice ended, with estimated recall of{" "}
                    {(historyCard.forecast[6] * 100).toFixed(1)}%.
                  </p>
                )}
              </details>
            </div>
          </aside>
        </div>

        <footer className={s.footer}>
          <details className={s.method}>
            <summary>About this study deck</summary>
            <p>
              Practice history comes from an anonymized Duolingo learner. The
              study calendar replays that history, with up to{" "}
              {replay.reviewLimit} interactive reviews per word.
            </p>
            <p>
              The demo uses precomputed branches from the report’s GBDT
              settings, refit on a sample of{" "}
              {replay.fitLearners.toLocaleString("en-US")} other learners. Your
              answers update the word you reviewed. Each word’s forecast starts
              from the same recorded learner history.
            </p>
            <p>
              Duolingo held-out recall prediction log loss was 0.259 for GBDT
              and 0.300 for optimized FSRS-6. The report evaluated recall
              predictions. Learning outcomes for these schedules remain
              untested.
            </p>
            <a href="https://github.com/PMS61/Axiom/blob/BayesianScheduler/blackbook/main.pdf">
              Project report ↗
            </a>
            {" · "}
            <a href="https://doi.org/10.7910/DVN/N8XJME">Practice data ↗</a>
          </details>
          {storageUnavailable && (
            <span>Progress is saved for this visit only.</span>
          )}
          <button
            type="button"
            disabled={!loaded}
            onClick={() => {
              setState(initialStudy());
              setRevealed(false);
              setHistoryWord(undefined);
              setNotice("");
            }}
          >
            Reset replay
          </button>
        </footer>
      </main>
    </>
  );
}
