"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { type Answer, initialStudy } from "@/app/learn/scheduled-reviews";
import Dashboard from "@/components/Dashboard";
import {
  describeGeneratedReviews,
  queuedLessons,
  updateLibrary,
  useLibrary,
} from "@/lib/learning-library";
import { nextReviewInterval } from "@/lib/review-scheduler";
import s from "./demo.module.css";
import {
  addDays,
  buildCalendar,
  type CalendarBlock,
  formatStudyDate,
  learnerName,
  roadmapTitle,
  scheduleDemoAnswer,
  study,
  upcomingReviewId,
} from "./demo-data";
import { epochDay, progressKey, readProgress } from "./progress";
import StudyHistory from "./StudyHistory";

function normalizeAnswer(answer: string) {
  return answer
    .normalize("NFKC")
    .replace(/[‘’]/g, "'")
    .trim()
    .replace(/[.!?]+$/, "")
    .trimEnd()
    .replace(/\s+/g, " ")
    .toLowerCase();
}

function ReviewDialog({
  card,
  onAnswer,
  onClose,
}: {
  card: {
    word: string;
    question: string;
    meaning: string;
    acceptedAnswers?: string[];
  };
  onAnswer: (answer: Answer, expected: string) => void;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const field = useRef<HTMLInputElement>(null);
  const continueButton = useRef<HTMLButtonElement>(null);
  const [response, setResponse] = useState("");
  const [result, setResult] = useState<Answer>();
  const normalized = normalizeAnswer(response);

  useEffect(() => {
    const element = dialog.current;
    if (!element?.open) element?.showModal();
    field.current?.focus();
  }, []);

  useEffect(() => {
    if (result) continueButton.current?.focus();
  }, [result]);

  function finish() {
    if (result) {
      onAnswer(result, card.meaning);
    } else {
      onClose();
    }
  }

  return (
    <dialog
      ref={dialog}
      className={s.dialog}
      aria-labelledby="review-title"
      onClose={finish}
    >
      <div className={s.dialogHead}>
        <p className="meta-text">Flashcard review · {card.word}</p>
        <button
          className="btn btn-sm"
          type="button"
          onClick={() => dialog.current?.close()}
        >
          Close
        </button>
      </div>
      {result ? (
        <div className={s.answerBack}>
          <h2 id="review-title">Answer</h2>
          <p className={s.expectedAnswer}>{card.meaning}</p>
          <output>
            {result === "remembered" ? "Correct." : "Incorrect."} Your answer:{" "}
            {response.trim()}
          </output>
          <button
            ref={continueButton}
            className="btn btn-primary"
            type="button"
            onClick={finish}
          >
            Continue
          </button>
        </div>
      ) : (
        <>
          <h2 id="review-title">{card.question}</h2>
          <form
            className={s.answerForm}
            onSubmit={(event) => {
              event.preventDefault();
              if (!normalized) return;
              setResult(
                [card.meaning, ...(card.acceptedAnswers ?? [])].some(
                  (answer) => normalized === normalizeAnswer(answer),
                )
                  ? "remembered"
                  : "missed",
              );
            }}
          >
            <label htmlFor="review-response">Your answer</label>
            <input
              id="review-response"
              ref={field}
              value={response}
              onChange={(event) => setResponse(event.target.value)}
              autoComplete="off"
              spellCheck={false}
              required
            />
            <button
              className="btn btn-primary"
              type="submit"
              disabled={!normalized}
            >
              Check answer
            </button>
          </form>
        </>
      )}
    </dialog>
  );
}

export default function GuestStudyDashboard({ today }: { today: string }) {
  const library = useLibrary();
  const roadmap = library.roadmaps.find(
    ({ id }) => id === library.activeRoadmapId,
  );
  const studied = new Set(
    library.lessons.filter(({ finished }) => finished).map(({ id }) => id),
  );
  const completed = roadmap?.nodes.filter(({ title }) =>
    studied.has(title.toLowerCase()),
  ).length;
  const [state, setState] = useState(initialStudy);
  const [startDate, setStartDate] = useState(today);
  const [ready, setReady] = useState(false);
  const [notice, setNotice] = useState("");
  const [active, setActive] = useState<string>();

  useEffect(() => {
    try {
      const previous = readProgress();
      setState(previous.state);
      const currentAnchor = addDays(today, 1 - previous.state.day);
      setStartDate(
        previous.start && previous.start > currentAnchor
          ? previous.start
          : currentAnchor,
      );
    } catch {
      // An invalid or unavailable browser save starts a fresh local replay.
    }
    setReady(true);
  }, [today]);

  useEffect(() => {
    if (!ready) return;
    try {
      localStorage.setItem(
        progressKey,
        JSON.stringify({
          start: startDate,
          progress: study.saveStudy(state),
        }),
      );
    } catch {
      // Guest progress remains available for the current page.
    }
  }, [state, startDate, ready]);

  const queue = study.reviewQueue(state);
  const current = queue.find(
    (card) => card.canReview && card.due === state.day,
  );
  const lastAnswer = current?.answers.at(-1);
  const lastRecorded = current?.practice.at(-1);
  const lastResult =
    lastAnswer?.answer ?? (lastRecorded?.recalled ? "remembered" : "missed");
  const daysSincePractice =
    state.day - (lastAnswer?.day ?? -(lastRecorded?.daysAgo ?? 0));
  const activeSeed = queue.find(
    (card) => card.canReview && card.due === state.day && card.word === active,
  );
  const currentDate = addDays(startDate, state.day - 1);
  const todayNumber = epochDay(currentDate);
  const generated = queuedLessons(library);
  const currentGenerated = generated.find((card) => card.due <= todayNumber);
  const activeGenerated = generated.find(
    (card) => card.id === active && card.due <= todayNumber,
  );
  const nextSeed = study.nextReviewDay(state);
  const nextGenerated = generated.find((card) => card.due > todayNumber)?.due;
  const futureReview =
    nextGenerated === undefined
      ? Infinity
      : state.day + nextGenerated - todayNumber;
  const nextDay = Math.min(nextSeed ?? Infinity, futureReview);
  const canAdvance =
    !currentGenerated && Number.isFinite(nextDay) && nextDay > state.day;

  const deckBlocks = new Map<
    string,
    { block: CalendarBlock; count: number; title: string }
  >();
  for (const card of generated) {
    const studyDay = state.day + Math.max(0, card.due - todayNumber);
    const key = `${card.lessonId}:${studyDay}`;
    const group = deckBlocks.get(key) ?? {
      block: {
        id: card.id,
        name: "",
        studyDay,
        slot: 46,
        minutes: 15,
        completed: false,
      },
      count: 0,
      title: card.title,
    };
    group.count++;
    deckBlocks.set(key, group);
  }
  const generatedBlocks = [...deckBlocks.values()].map(
    ({ block, count, title }) => ({
      ...block,
      name: `Review · ${title} (${count} ${count === 1 ? "card" : "cards"})`,
    }),
  );
  const calendar = buildCalendar(startDate, state, generatedBlocks);
  const activeCard =
    activeSeed ??
    (activeGenerated
      ? { ...activeGenerated, word: activeGenerated.title }
      : undefined);

  function answer(result: Answer, expected: string) {
    let title: string;
    let interval: number;
    if (activeSeed) {
      const nextState = study.recordAnswer(state, activeSeed.word, result);
      if (nextState === state) return;
      const nextPlan = scheduleDemoAnswer(
        activeSeed,
        nextState.answers,
        state.day,
      );
      title = activeSeed.word;
      interval = nextPlan.interval;
      setState(nextState);
    } else if (activeGenerated) {
      const card = activeGenerated;
      const event = {
        cardId: card.id,
        day: todayNumber,
        recalled: result === "remembered",
      };
      title = card.title;
      interval = 0;
      updateLibrary((previous) => {
        const lesson = previous.lessons.find(
          (item) => item.id === card.lessonId,
        );
        if (
          lesson?.due?.[card.cardId] === undefined ||
          lesson.due[card.cardId] > todayNumber
        ) {
          return previous;
        }
        const events = [...previous.reviews, event];
        interval = nextReviewInterval(card.id, events, todayNumber);
        return {
          ...previous,
          activeTopic: title,
          reviews: events,
          lessons: previous.lessons.map((item) =>
            item.id === card.lessonId
              ? {
                  ...item,
                  due: { ...item.due, [card.cardId]: todayNumber + interval },
                }
              : item,
          ),
        };
      });
      if (!interval) {
        setActive(undefined);
        return;
      }
    } else {
      return;
    }

    const date = formatStudyDate(addDays(currentDate, interval));
    const when =
      interval === 1 ? `tomorrow, ${date}` : `${date}, in ${interval} days`;
    const feedback =
      result === "remembered" ? "Correct." : `Expected answer: ${expected}.`;
    setNotice(`${feedback} Next review ${when}.`);
    setActive(undefined);
  }

  function reset() {
    setState(initialStudy());
    setStartDate(today);
    setNotice("");
    setActive(undefined);
  }

  return (
    <>
      <Dashboard
        key={state.day}
        guest={{
          tasks: calendar.tasks,
          days: calendar.days,
          currentDate: new Date(`${calendar.today}T12:00:00Z`),
          summary: (
            <div className={s.summary}>
              <strong>Welcome back, {learnerName}.</strong>
              <Link href="/dashboard/demo/roadmap">
                {roadmap
                  ? `${roadmap.goal} · ${completed} of ${roadmap.nodes.length} topics completed`
                  : `${roadmapTitle} · 2 of 4 topics completed`}
              </Link>
              {library.activeTopic !== roadmapTitle && (
                <Link
                  href={`/dashboard/demo/study?topic=${encodeURIComponent(library.activeTopic)}`}
                >
                  Continue studying {library.activeTopic}
                </Link>
              )}
            </div>
          ),
          controls: (
            <section
              className={`container section-rule ${s.controls}`}
              aria-label="Schedule controls"
            >
              <h2>My schedule</h2>
              {canAdvance && (
                <button
                  className="btn btn-sm"
                  type="button"
                  disabled={!ready}
                  onClick={() => {
                    setState((previous) => ({ ...previous, day: nextDay }));
                    setNotice("");
                  }}
                >
                  Go to next review day →
                </button>
              )}
            </section>
          ),
          focus: (
            <div className={s.focus}>
              <div>
                <p className="meta-text">
                  {current || currentGenerated ? "Due today" : "Review queue"}
                </p>
                <h3>
                  {current?.word ??
                    currentGenerated?.title ??
                    "You’re caught up on reviews."}
                </h3>
                <p>
                  {current
                    ? `Last review ${lastResult} · ${daysSincePractice === 1 ? "yesterday" : `${daysSincePractice} days ago`}.`
                    : currentGenerated
                      ? "Flashcards from your lesson are ready to review."
                      : !Number.isFinite(nextDay)
                        ? "No reviews are scheduled."
                        : `Next review on ${formatStudyDate(addDays(startDate, nextDay - 1))}.`}
                </p>
                <output aria-live="polite">{notice}</output>
              </div>
              {(current || currentGenerated) && (
                <button
                  className="btn btn-primary"
                  type="button"
                  disabled={!ready}
                  onClick={() =>
                    setActive(current?.word ?? currentGenerated?.id)
                  }
                >
                  Review now
                </button>
              )}
            </div>
          ),
          renderTaskAction: (id) => {
            const card = queue.find(
              (item) => upcomingReviewId(item.word) === id,
            );
            if (card) {
              return (
                <div className={s.taskAction}>
                  <span className="meta-text">GBDT + online review</span>
                  {card.canReview && card.due === state.day && (
                    <button
                      className="btn btn-sm"
                      type="button"
                      disabled={!ready}
                      onClick={() => setActive(card.word)}
                    >
                      Start review
                    </button>
                  )}
                </div>
              );
            }
            const generatedCard = generated.find((item) => item.id === id);
            if (generatedCard) {
              return (
                <div className={s.taskAction}>
                  <span className="meta-text">
                    GBDT + online flashcard review
                  </span>
                  {generatedCard.due <= todayNumber && (
                    <button
                      className="btn btn-sm"
                      type="button"
                      disabled={!ready}
                      onClick={() => setActive(id)}
                    >
                      Start review
                    </button>
                  )}
                </div>
              );
            }
            const task = calendar.tasks.find((item) => item.id === id);
            return task?.state === "completed" ? (
              <span className={`meta-text ${s.completed}`}>Completed</span>
            ) : null;
          },
          activity: (
            <StudyHistory
              startDate={startDate}
              answers={state.answers}
              generated={describeGeneratedReviews(library).map((event) => ({
                ...event,
                day: event.day - epochDay(startDate) + 1,
              }))}
              ready={ready}
              onReset={reset}
            />
          ),
        }}
      />
      {activeCard && (
        <ReviewDialog
          key={activeCard.word}
          card={activeCard}
          onAnswer={answer}
          onClose={() => setActive(undefined)}
        />
      )}
    </>
  );
}
