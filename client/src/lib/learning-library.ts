"use client";

import { useSyncExternalStore } from "react";
import { z } from "zod";
import preset from "@/app/dashboard/demo/preset.json";
import {
  deckSchema,
  roadmapSchema,
  slidesSchema,
  topicSchema,
} from "./learning-content";

const libraryKey = "axiom-learning-library-v1";
const changed = "axiom-learning-library-change";

const lessonSchema = z
  .object({
    id: z.string().max(100),
    topic: topicSchema,
    slides: slidesSchema,
    slideIndex: z.number().int().min(0).max(9),
    cards: deckSchema.optional(),
    finished: z.boolean().optional(),
    due: z.record(z.string(), z.number().int().positive()).optional(),
  })
  .refine((lesson) => lesson.slideIndex < lesson.slides.length);

const librarySchema = z.object({
  interests: z.array(topicSchema).max(12),
  lessons: z.array(lessonSchema).max(40),
  activeTopic: topicSchema,
  roadmaps: z.array(roadmapSchema).max(20).default([]),
  activeRoadmapId: z.string().max(64).optional(),
  reviews: z.array(
    z.object({
      cardId: z.string().min(1).max(190),
      day: z.number().int().positive(),
      recalled: z.boolean(),
    }),
  ),
});
export type Library = z.infer<typeof librarySchema>;
export type Lesson = z.infer<typeof lessonSchema>;

const empty: Library = {
  interests: preset.interests,
  lessons: [],
  roadmaps: [],
  activeTopic: preset.title,
  reviews: [],
};

let snapshot = empty;
let previousRaw: string | null | undefined;

function readLibrary() {
  try {
    const raw = localStorage.getItem(libraryKey);
    if (raw !== previousRaw) {
      previousRaw = raw;
      const parsed = raw ? librarySchema.safeParse(JSON.parse(raw)) : null;
      snapshot = parsed?.success ? parsed.data : empty;
    }
  } catch {
    // Browser storage may be unavailable. Keep the current in-memory value.
  }
  return snapshot;
}

function subscribe(notify: () => void) {
  window.addEventListener(changed, notify);
  window.addEventListener("storage", notify);
  return () => {
    window.removeEventListener(changed, notify);
    window.removeEventListener("storage", notify);
  };
}

export function useLibrary() {
  return useSyncExternalStore(subscribe, readLibrary, () => empty);
}

export function updateLibrary(change: (previous: Library) => Library) {
  const next = change(readLibrary());
  snapshot = next;
  try {
    const raw = JSON.stringify(next);
    localStorage.setItem(libraryKey, raw);
    previousRaw = raw;
  } catch {
    // Keep changes in memory for the current browser tab.
  }
  window.dispatchEvent(new Event(changed));
}

export function updateLesson(id: string, update: (lesson: Lesson) => Lesson) {
  updateLibrary((library) => ({
    ...library,
    lessons: library.lessons.map((lesson) =>
      lesson.id === id ? update(lesson) : lesson,
    ),
  }));
}

export function learningCardId(lessonId: string, cardId: string) {
  return `generated:${lessonId}:${cardId}`;
}

export function queuedLessons(library: Library) {
  return library.lessons
    .flatMap((lesson) =>
      (lesson.cards ?? []).flatMap((card) => {
        const due = lesson.due?.[card.id];
        return due === undefined
          ? []
          : [
              {
                id: learningCardId(lesson.id, card.id),
                title: lesson.topic,
                question: card.front,
                meaning: card.back,
                lessonId: lesson.id,
                cardId: card.id,
                due,
              },
            ];
      }),
    )
    .sort((a, b) => a.due - b.due);
}

export function describeGeneratedReviews(library: Library) {
  const titles = new Map(
    queuedLessons(library).map((card) => [card.id, card.title]),
  );
  return library.reviews.flatMap((entry) => {
    const title = titles.get(entry.cardId);
    return title ? [{ title, day: entry.day, recalled: entry.recalled }] : [];
  });
}
