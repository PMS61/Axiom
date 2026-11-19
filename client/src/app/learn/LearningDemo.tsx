"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  initialLearningState,
  karlDataset,
  type LearningState,
  type Rating,
  rateCard,
  recallEstimate,
  restoreLearningState,
  reviewInterval,
} from "@/lib/bayesian/learning";
import s from "./learn.module.css";

const storageKey = "axiom-learning-karl-v1";
const ratings: { value: Rating; label: string; detail: string }[] = [
  { value: "again", label: "Again", detail: "I forgot" },
  { value: "hard", label: "Hard", detail: "With effort" },
  { value: "good", label: "Good", detail: "I remembered" },
  { value: "easy", label: "Easy", detail: "Instant recall" },
];

export default function LearningDemo() {
  const [state, setState] = useState<LearningState>(initialLearningState);
  const [ready, setReady] = useState(false);
  const [revealed, setRevealed] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [notice, setNotice] = useState(
    "Choose a card, think of the answer, then reveal it.",
  );
  const [storageNotice, setStorageNotice] = useState("");

  useEffect(() => {
    try {
      const saved = localStorage.getItem(storageKey);
      if (saved) setState(restoreLearningState(saved));
    } catch {
      setStorageNotice(
        "Saved progress could not be restored. A fresh demo is ready.",
      );
    }
    setReady(true);
  }, []);

  useEffect(() => {
    if (!ready) return;
    try {
      localStorage.setItem(storageKey, JSON.stringify(state));
    } catch {
      setStorageNotice(
        "Browser storage is unavailable. Progress lasts for this visit.",
      );
    }
  }, [state, ready]);

  const due = state.cards
    .filter((card) => card.due <= state.day)
    .sort(
      (a, b) =>
        recallEstimate(a, state.day) - recallEstimate(b, state.day) ||
        a.id.localeCompare(b.id),
    );
  const card = due.find((item) => item.id === selected) || due[0];
  const deckCards = [
    ...due,
    ...state.cards
      .filter((item) => item.due > state.day)
      .sort((a, b) => a.due - b.due),
  ];
  const sourceCard = karlDataset.cards.find(
    (item) => item.cardId === card?.sourceId,
  );
  const sourceReviews = karlDataset.cards.reduce(
    (sum, item) => sum + item.events.length,
    0,
  );
  const today = state.history.filter((event) => event.day === state.day);
  const remembered = today.filter((event) => event.rating !== "again").length;
  const nextDue = Math.min(...state.cards.map((item) => item.due));
  const forecast = Array.from({ length: 7 }, (_, offset) => ({
    day: state.day + offset + 1,
    count: state.cards.filter((item) => item.due === state.day + offset + 1)
      .length,
  }));
  const forecastMax = Math.max(1, ...forecast.map((item) => item.count));

  function rate(rating: Rating) {
    if (!card || !revealed) return;
    const interval = reviewInterval(card, rating);
    setState((previous) => rateCard(previous, card.id, rating));
    setNotice(
      `${card.title} · ${ratings.find((item) => item.value === rating)?.label}. Due in ${interval} ${interval === 1 ? "day" : "days"}. ${rating === "again" ? "A shorter interval brings it back sooner." : "Your rating sets the next review interval."}`,
    );
    setSelected(null);
    setRevealed(false);
  }

  function advance(day: number) {
    setState((previous) => ({ ...previous, day }));
    setSelected(null);
    setRevealed(false);
    setNotice(
      `Moved to demo day ${day + 1}. Cards due by this day return to the queue.`,
    );
  }

  return (
    <main className={s.app}>
      <header className={s.header}>
        <Link href="/" className={s.brand}>
          axiom<span>learn</span>
        </Link>
        <nav aria-label="Learning navigation">
          <Link href="/bayesian-demo/benchmarks">Latest benchmarks ↗</Link>
          <Link href="/bayesian-demo">Research workspace ↗</Link>
          <span className={s.demoTag}>KAR³L DATA DEMO</span>
        </nav>
      </header>
      <section className={s.intro}>
        <div>
          <p className={s.eyebrow}>A LITTLE EVERY DAY.</p>
          <h1>Keep what you learn.</h1>
          <p>A question. A moment to remember. A plan for next time.</p>
        </div>
        <div className={s.dayControls}>
          <span>SIMULATED TIME</span>
          <strong>Day {state.day + 1}</strong>
          <button
            type="button"
            disabled={!ready}
            onClick={() => advance(state.day + 1)}
          >
            Next day →
          </button>
        </div>
      </section>
      <section className={s.sourceBanner} aria-label="Dataset source">
        <div>
          <strong>Real questions. Real review histories.</strong>
          <p>
            {karlDataset.cards.length} matched trivia cards · {sourceReviews}{" "}
            recorded reviews · anonymized learner {karlDataset.learnerId}. Your
            answers start a separate demo session.
          </p>
        </div>
        <a href={karlDataset.datasetUrl} target="_blank" rel="noreferrer">
          Explore KAR³L ↗
        </a>
      </section>
      <section className={s.stats} aria-label="Study progress">
        <div>
          <strong>{due.length.toString().padStart(2, "0")}</strong>
          <span>cards due</span>
        </div>
        <div>
          <strong>{today.length.toString().padStart(2, "0")}</strong>
          <span>reviewed today</span>
        </div>
        <div>
          <strong>
            {today.length
              ? `${Math.round((remembered / today.length) * 100)}%`
              : "—"}
          </strong>
          <span>self-reported recall today</span>
        </div>
        <div>
          <strong>{state.cards.length.toString().padStart(2, "0")}</strong>
          <span>cards in your deck</span>
        </div>
      </section>
      <div className={s.layout}>
        <aside className={s.deck}>
          <div className={s.sectionHead}>
            <div>
              <p className={s.eyebrow}>YOUR DECK</p>
              <h2>KAR³L trivia</h2>
            </div>
            <span>{state.cards.length}</span>
          </div>
          <p className={s.small}>
            Due cards are ordered by estimated recall, lowest first.
          </p>
          <div className={s.cardList}>
            {deckCards.map((item) => {
              const isDue = item.due <= state.day;
              return (
                <button
                  type="button"
                  key={item.id}
                  disabled={!isDue || !ready}
                  aria-pressed={card?.id === item.id}
                  className={card?.id === item.id ? s.activeCard : ""}
                  onClick={() => {
                    setSelected(item.id);
                    setRevealed(false);
                  }}
                >
                  <span className={s.cardDot} />
                  <span>
                    <strong>{item.title}</strong>
                    <small>
                      {isDue ? "Ready to review" : `Due day ${item.due + 1}`}
                    </small>
                  </span>
                  <span>
                    {item.reviews || item.sourceId
                      ? `${Math.round(recallEstimate(item, state.day) * 100)}%`
                      : "New"}
                  </span>
                </button>
              );
            })}
          </div>
          <button
            type="button"
            className={s.addButton}
            onClick={() => setAdding(!adding)}
            disabled={!ready}
          >
            {adding ? "Close card editor" : "+ Add your own card"}
          </button>
          {adding && (
            <form
              className={s.form}
              onSubmit={(event) => {
                event.preventDefault();
                const form = event.currentTarget;
                const values = new FormData(form);
                const title = String(values.get("title") || "").trim();
                const question = String(values.get("question") || "").trim();
                const answer = String(values.get("answer") || "").trim();
                if (!title || !question || !answer) return;
                const id = crypto.randomUUID();
                setState((previous) => ({
                  ...previous,
                  cards: [
                    ...previous.cards,
                    {
                      id,
                      title,
                      question,
                      answer,
                      stability: 1,
                      lastReview: previous.day,
                      due: previous.day,
                      reviews: 0,
                    },
                  ],
                }));
                setSelected(id);
                setRevealed(false);
                setAdding(false);
                setNotice(
                  "Your new card is ready. Reveal the answer and rate your recall.",
                );
              }}
            >
              <label>
                Topic
                <input name="title" maxLength={80} required />
              </label>
              <label>
                Question
                <textarea name="question" maxLength={500} required />
              </label>
              <label>
                Answer
                <textarea name="answer" maxLength={1500} required />
              </label>
              <button className={s.primary} type="submit">
                Add to deck
              </button>
            </form>
          )}
        </aside>
        <section
          className={s.session}
          aria-label="Flashcard session"
          aria-busy={!ready}
        >
          <div className={s.sessionTop}>
            <span>FOCUS SESSION</span>
            <span>
              {due.length ? `${due.length} remaining` : "All caught up"}
            </span>
          </div>
          <div className={s.progressTrack}>
            <div
              style={{
                width: `${(today.length / Math.max(1, today.length + due.length)) * 100}%`,
              }}
            />
          </div>
          {card ? (
            <div className={s.flashcard} key={card.id}>
              <p className={s.eyebrow}>{card.title}</p>
              <span className={s.questionLabel}>Think before you reveal.</span>
              <h2>{card.question}</h2>
              <div className={s.recallNote}>
                <span className={s.recallRing}>
                  {card.reviews || card.sourceId ? (
                    <>
                      {Math.round(recallEstimate(card, state.day) * 100)}
                      <small>%</small>
                    </>
                  ) : (
                    "—"
                  )}
                </span>
                <p>
                  Illustrative recall estimate
                  <span>
                    {sourceCard
                      ? `${sourceCard.events.length} recorded reviews · ${state.history.filter((event) => event.cardId === card.id).length} in this demo`
                      : card.reviews
                        ? `${card.reviews} demo reviews recorded`
                        : "New card · no review history yet"}
                  </span>
                </p>
              </div>
              {sourceCard && (
                <details className={s.history} key={sourceCard.cardId}>
                  <summary>
                    Recorded history ·{" "}
                    {sourceCard.events.filter((event) => event.recalled).length}{" "}
                    / {sourceCard.events.length} recalled
                  </summary>
                  <p>
                    Original card #{sourceCard.cardId} · learner{" "}
                    {karlDataset.learnerId}. Latest eight released reviews, in
                    UTC.
                  </p>
                  <ol>
                    {sourceCard.events.slice(-8).map((event) => (
                      <li key={event.at}>
                        <time dateTime={event.at}>{event.at.slice(0, 10)}</time>
                        <span data-recalled={event.recalled}>
                          {event.recalled ? "Recalled" : "Forgotten"}
                        </span>
                      </li>
                    ))}
                  </ol>
                  <p>
                    Question and answer joined by card ID, with matching
                    question text. Wording is retained from the source.
                  </p>
                </details>
              )}
              {!revealed ? (
                <button
                  type="button"
                  className={s.primary}
                  onClick={() => setRevealed(true)}
                  disabled={!ready}
                >
                  Reveal answer <span>↓</span>
                </button>
              ) : (
                <>
                  <div className={s.answer}>
                    <p className={s.eyebrow}>THE ANSWER</p>
                    <p>{card.answer}</p>
                  </div>
                  <p className={s.ratingPrompt}>How well did you remember?</p>
                  <div className={s.ratings}>
                    {ratings.map((rating) => (
                      <button
                        type="button"
                        key={rating.value}
                        onClick={() => rate(rating.value)}
                        data-rating={rating.value}
                      >
                        <strong>{rating.label}</strong>
                        <span>{rating.detail}</span>
                        <small>{reviewInterval(card, rating.value)}d</small>
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          ) : (
            <div className={s.complete}>
              <span className={s.completeMark}>✓</span>
              <p className={s.eyebrow}>ROOM TO LET IT SINK IN.</p>
              <h2>You’re all caught up.</h2>
              <p>
                {today.length} cards reviewed today. Your next review is on demo
                day {nextDue + 1}.
              </p>
              <button
                type="button"
                className={s.primary}
                disabled={!ready}
                onClick={() => advance(Math.max(state.day + 1, nextDue))}
              >
                Jump to next review →
              </button>
            </div>
          )}
          <output className={s.notice} aria-live="polite">
            {notice}
          </output>
        </section>
        <aside className={s.insights}>
          <p className={s.eyebrow}>THE DAYS AHEAD</p>
          <h2>A little, spaced out.</h2>
          <p className={s.small}>
            Your upcoming reviews change with every answer.
          </p>
          <section
            className={s.forecast}
            aria-label="Review forecast for the next seven days"
          >
            {forecast.map((item) => (
              <div key={item.day}>
                <span>{item.count || "·"}</span>
                <div className={s.barTrack}>
                  <div
                    style={{ height: `${(item.count / forecastMax) * 100}%` }}
                  />
                </div>
                <small>D{item.day + 1}</small>
              </div>
            ))}
          </section>
          <div className={s.explainer}>
            <span>01</span>
            <h3>Try to remember</h3>
            <p>Recall the answer before revealing it.</p>
            <span>02</span>
            <h3>Give honest feedback</h3>
            <p>
              Forgotten cards return sooner. Easier cards get longer intervals.
            </p>
            <span>03</span>
            <h3>Come back tomorrow</h3>
            <p>Use “Next day” to show the review cycle in seconds.</p>
          </div>
          <details className={s.method}>
            <summary>About this demo</summary>
            <p>
              <a href={karlDataset.paperUrl} target="_blank" rel="noreferrer">
                KAR³L · Shu, Balepur, Feng &amp; Boyd-Graber · EMNLP 2024 ↗
              </a>
            </p>
            <p>
              The eight starter cards and their historical responses come from
              KAR³L. Day 1 starts one day after the latest selected recorded
              review ({karlDataset.snapshotAt.slice(0, 10)}). All selected cards
              are offered for practice; this is not the original learner’s
              scheduled queue.
            </p>
            <p>
              The illustrative forgetting curve starts from each card’s median
              recorded review gap, with a minimum of one day. New intervals use
              demo rules, not a fitted Bayesian model or FSRS. Historical recall
              counts are observed; recall estimates and future dates are
              simulated.
            </p>
            <p>
              <a href={karlDataset.factsUrl} target="_blank" rel="noreferrer">
                Original questions and answers ↗
              </a>
            </p>
            <Link href="/bayesian-demo/audit">See the research evidence ↗</Link>
          </details>
        </aside>
      </div>
      <footer className={s.footer}>
        <span>
          {storageNotice ||
            (ready
              ? "Progress saved in this browser · No account needed"
              : "Loading your study desk…")}
        </span>
        <button
          type="button"
          disabled={!ready}
          onClick={() => {
            setState(initialLearningState());
            setSelected(null);
            setRevealed(false);
            setAdding(false);
            setNotice("Demo reset. Eight KAR³L cards are ready to review.");
          }}
        >
          Reset demo
        </button>
      </footer>
    </main>
  );
}
