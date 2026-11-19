"use client";

import Link from "next/link";
import { useState } from "react";
import SlideViewer from "@/components/SlideViewer";
import { deckSchema, slidesSchema } from "@/lib/learning-content";
import {
  type Lesson,
  updateLesson,
  updateLibrary,
  useLibrary,
} from "@/lib/learning-library";
import { addDays } from "../demo-data";
import s from "../learning.module.css";
import { epochDay, localToday, readProgress } from "../progress";

async function generate(input: object) {
  const response = await fetch("/api/learning/content", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error ?? "Generation failed.");
  return result;
}

function studyDay() {
  const today = localToday();
  try {
    const { state, start } = readProgress();
    const simulatedDate = start ? addDays(start, state.day - 1) : today;
    return epochDay(simulatedDate > today ? simulatedDate : today);
  } catch {
    return epochDay(today);
  }
}

export default function StudyWorkspace({
  startingTopic,
  context,
}: {
  startingTopic: string;
  context?: string;
}) {
  const library = useLibrary();
  const [topic, setTopic] = useState(startingTopic);
  const [lessonId, setLessonId] = useState(startingTopic.toLowerCase().trim());
  const [busy, setBusy] = useState<"slides" | "cards">();
  const [error, setError] = useState("");
  const lesson = library.lessons.find((item) => item.id === lessonId);

  async function createLesson(upgrade?: Lesson) {
    const normalized = upgrade?.topic ?? topic.trim();
    const id = normalized.toLowerCase();
    setError("");
    const existing = library.lessons.find((item) => item.id === id);
    if (existing && !upgrade) {
      setLessonId(id);
      updateLibrary((previous) => ({
        ...previous,
        activeTopic: existing.topic,
      }));
      return;
    }
    if (!upgrade && library.lessons.length >= 40) {
      setError("Your library already has 40 lessons.");
      return;
    }
    setBusy("slides");
    try {
      const result = await generate({
        kind: "lesson",
        topic: normalized,
        context: normalized === startingTopic ? context : undefined,
      });
      const slides = slidesSchema.parse(result.slides);
      updateLibrary((previous) => ({
        ...previous,
        activeTopic: normalized,
        lessons: upgrade
          ? previous.lessons.map((item) =>
              item.id === id ? { ...item, slides, slideIndex: 0 } : item,
            )
          : previous.lessons.some((lesson) => lesson.id === id)
            ? previous.lessons
            : [
                ...previous.lessons,
                { id, topic: normalized, slides, slideIndex: 0 },
              ],
      }));
      setLessonId(id);
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Lessons are temporarily unavailable.",
      );
    } finally {
      setBusy(undefined);
    }
  }

  async function createCards(active: Lesson) {
    setError("");
    setBusy("cards");
    try {
      const result = await generate({
        kind: "cards",
        topic: active.topic,
        slides: active.slides,
      });
      const cards = deckSchema.parse(result.cards);
      updateLesson(active.id, (item) =>
        item.cards ? item : { ...item, cards },
      );
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Flashcards are temporarily unavailable.",
      );
    } finally {
      setBusy(undefined);
    }
  }

  function addToSchedule(active: Lesson) {
    if (!active.cards) return;
    const due = studyDay() + 1;
    updateLesson(active.id, (item) =>
      item.due
        ? item
        : {
            ...item,
            due: Object.fromEntries(
              active.cards?.map((card) => [card.id, due]) ?? [],
            ),
          },
    );
  }

  const slide = lesson?.slides[lesson.slideIndex];
  return (
    <main className={`container ${s.learningPage}`}>
      <header className={s.pageHead}>
        <h1>Study a topic</h1>
        <p>
          Learn it with slides, turn the key ideas into flashcards, then put the
          cards on your review schedule.
        </p>
      </header>
      <form
        className={s.generationForm}
        onSubmit={(event) => {
          event.preventDefault();
          if (!busy) void createLesson();
        }}
      >
        <div>
          <label htmlFor="learning-topic">Topic</label>
          <input
            id="learning-topic"
            required
            minLength={2}
            maxLength={100}
            value={topic}
            onChange={(event) => setTopic(event.target.value)}
          />
        </div>
        <button
          type="submit"
          className="btn btn-primary"
          disabled={Boolean(busy) || topic.trim().length < 2}
        >
          {busy === "slides" ? "Generating slides…" : "Open or generate lesson"}
        </button>
      </form>
      {library.lessons.length > 0 && (
        <section className={s.library} aria-label="Saved lessons">
          <h2>My lessons</h2>
          <div>
            {library.lessons.map((item) => (
              <button
                key={item.id}
                className="btn btn-sm"
                type="button"
                aria-pressed={item.id === lesson?.id}
                disabled={Boolean(busy)}
                onClick={() => {
                  setLessonId(item.id);
                  setTopic(item.topic);
                  setError("");
                  updateLibrary((previous) => ({
                    ...previous,
                    activeTopic: item.topic,
                  }));
                }}
              >
                {item.topic}
              </button>
            ))}
          </div>
        </section>
      )}
      {error && (
        <p role="alert" className={s.error}>
          {error}
        </p>
      )}
      {lesson && (
        <article className={s.lesson}>
          <header>
            <h2>{lesson.topic}</h2>
            <p className="meta-text">
              Slide {lesson.slideIndex + 1} of {lesson.slides.length}
            </p>
          </header>
          {slide && (
            <>
              <div className={s.slide} data-slide-type={slide.type}>
                <SlideViewer slide={slide} />
              </div>
              {slide.script && (
                <details className={s.speakerNotes}>
                  <summary>Tutor notes</summary>
                  <p>{slide.script}</p>
                </details>
              )}
            </>
          )}
          <div className={s.lessonControls}>
            <button
              type="button"
              className="btn"
              disabled={lesson.slideIndex === 0}
              onClick={() =>
                updateLesson(lesson.id, (item) => ({
                  ...item,
                  slideIndex: Math.max(0, item.slideIndex - 1),
                }))
              }
            >
              Previous slide
            </button>
            {lesson.slideIndex < lesson.slides.length - 1 ? (
              <button
                type="button"
                className="btn btn-primary"
                onClick={() =>
                  updateLesson(lesson.id, (item) => ({
                    ...item,
                    slideIndex: Math.min(
                      item.slides.length - 1,
                      item.slideIndex + 1,
                    ),
                  }))
                }
              >
                Next slide
              </button>
            ) : (
              !lesson.cards && (
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={Boolean(busy)}
                  onClick={() => void createCards(lesson)}
                >
                  {busy === "cards"
                    ? "Generating flashcards…"
                    : "Generate flashcards"}
                </button>
              )
            )}
          </div>
          {lesson.slides.every((item) => item.type === "content") && (
            <button
              type="button"
              className={`btn btn-sm ${s.upgrade}`}
              disabled={Boolean(busy)}
              onClick={() => void createLesson(lesson)}
            >
              Upgrade slides
            </button>
          )}
          {lesson.slideIndex === lesson.slides.length - 1 &&
            !lesson.finished && (
              <button
                type="button"
                className={`btn btn-sm ${s.upgrade}`}
                onClick={() =>
                  updateLesson(lesson.id, (item) => ({
                    ...item,
                    finished: true,
                  }))
                }
              >
                Mark lesson studied
              </button>
            )}
          {lesson.cards && (
            <section className={s.studyCards} aria-labelledby="deck-heading">
              <h3 id="deck-heading">{lesson.cards.length} flashcards</h3>
              <ul>
                {lesson.cards.map((card) => (
                  <li key={card.id}>
                    <details>
                      <summary>{card.front}</summary>
                      <p>{card.back}</p>
                    </details>
                  </li>
                ))}
              </ul>
              {lesson.due ? (
                <Link className="btn btn-primary" href="/dashboard/demo">
                  Open my review schedule
                </Link>
              ) : (
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => addToSchedule(lesson)}
                >
                  Schedule first review for tomorrow
                </button>
              )}
            </section>
          )}
        </article>
      )}
    </main>
  );
}
