"use client";

import Link from "next/link";
import { useState } from "react";
import type { StudyState } from "@/app/learn/scheduled-reviews";
import s from "./demo.module.css";
import { addDays, formatStudyDate, preset, roadmapTitle } from "./demo-data";

function describeOutcomes(remembered: number, missed: number) {
  return [
    remembered > 0 && `${remembered.toLocaleString("en-IN")} remembered`,
    missed > 0 && `${missed.toLocaleString("en-IN")} missed`,
  ]
    .filter(Boolean)
    .join(" · ");
}

type GeneratedReview = { title: string; day: number; recalled: boolean };

function buildHistory(
  answers: StudyState["answers"],
  generated: GeneratedReview[],
) {
  type Counts = { remembered: number; missed: number };
  const totals = new Map<number, Counts & { cards: Map<string, Counts> }>();
  const entries = [
    ...preset.cards.flatMap((card) =>
      card.practice.map(({ daysAgo, recalled }) => ({
        day: -daysAgo,
        title: card.word,
        recalled,
      })),
    ),
    ...answers.map(({ word, day, answer }) => ({
      day,
      title: word,
      recalled: answer === "remembered",
    })),
    ...generated,
  ];

  for (const entry of entries) {
    const day = totals.get(entry.day) ?? {
      remembered: 0,
      missed: 0,
      cards: new Map<string, { remembered: number; missed: number }>(),
    };
    const card = day.cards.get(entry.title) ?? { remembered: 0, missed: 0 };
    const outcome = entry.recalled ? "remembered" : "missed";
    card[outcome]++;
    day[outcome]++;
    day.cards.set(entry.title, card);
    totals.set(entry.day, day);
  }

  return [...totals.entries()].sort(([a], [b]) => b - a);
}

export default function StudyHistory({
  startDate,
  answers,
  generated,
  ready,
  onReset,
}: {
  startDate: string;
  answers: StudyState["answers"];
  generated: GeneratedReview[];
  ready: boolean;
  onReset: () => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const history = buildHistory(answers, generated);
  const reviews = history.reduce(
    (sum, [, day]) => sum + day.remembered + day.missed,
    0,
  );
  const remembered = history.reduce((sum, [, day]) => sum + day.remembered, 0);
  const shown = showAll ? history : history.slice(0, 7);

  return (
    <section
      className={`container ${s.activity}`}
      id="study-history"
      aria-labelledby="activity-title"
    >
      <div className={s.activityHead}>
        <div>
          <h2 id="activity-title">Study history</h2>
          <p>Past practice across all decks, including {roadmapTitle}.</p>
        </div>
        <Link href="/dashboard/demo/roadmap" className="btn btn-sm">
          View roadmap
        </Link>
      </div>
      <dl className={s.historyStats}>
        <div>
          <dt>Reviews recorded</dt>
          <dd>{reviews.toLocaleString("en-IN")}</dd>
        </div>
        <div>
          <dt>Days studied</dt>
          <dd>{history.length}</dd>
        </div>
        <div>
          <dt>Remembered</dt>
          <dd>{Math.round((remembered / reviews) * 100)}%</dd>
        </div>
      </dl>
      <ol
        className={s.history}
        id="daily-activity"
        aria-label="Daily practice history"
      >
        {shown.map(([day, activity]) => {
          const date = addDays(startDate, day - 1);
          const counts = describeOutcomes(activity.remembered, activity.missed);
          const count = activity.remembered + activity.missed;
          return (
            <li key={day}>
              <div className={s.historyDay}>
                <time dateTime={date}>{formatStudyDate(date)}</time>
                <strong>
                  {count.toLocaleString("en-IN")}{" "}
                  {count === 1 ? "review" : "reviews"}
                </strong>
                <span>
                  {counts}
                  {answers.some((answer) => answer.day === day) &&
                    " · this demo"}
                </span>
              </div>
              {activity.cards.size > 0 && (
                <ul
                  className={s.historyCards}
                  aria-label={`Flashcard reviews on ${formatStudyDate(date)}`}
                >
                  {[...activity.cards].map(([title, outcomes]) => (
                    <li key={title}>
                      <span>{title}</span>
                      <span>
                        {describeOutcomes(outcomes.remembered, outcomes.missed)}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </li>
          );
        })}
      </ol>
      {history.length > 7 && (
        <button
          type="button"
          className={`btn btn-sm ${s.historyToggle}`}
          aria-expanded={showAll}
          aria-controls="daily-activity"
          onClick={() => setShowAll(!showAll)}
        >
          {showAll
            ? "Show recent days"
            : `Show all ${history.length} study days`}
        </button>
      )}
      <div className={s.activityFoot}>
        <details>
          <summary>About this learner and the review dates</summary>
          <p>
            Alex’s roadmap and earlier practice are illustrative. The flashcards
            cover concepts from the Axiom project.
          </p>
          <p>
            Initial review dates are seeded. After each answer, the demo
            recomputes the next date using its GBDT and the six-weight online
            update from the benchmark. Live scheduling targets 80% recall.
          </p>
          <a href="https://github.com/PMS61/Axiom/blob/BayesianScheduler/blackbook/main.pdf">
            Project report ↗
          </a>
        </details>
        <button
          type="button"
          className="btn btn-sm"
          disabled={!ready}
          onClick={onReset}
        >
          Reset demo
        </button>
      </div>
    </section>
  );
}
