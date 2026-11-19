import snapshot from "./karl-sample/karl-demo.json";

export const karlDataset = snapshot;

export type Rating = "again" | "hard" | "good" | "easy";
export type StudyCard = {
  id: string;
  title: string;
  question: string;
  answer: string;
  stability: number;
  lastReview: number;
  due: number;
  reviews: number;
  sourceId?: string;
};
export type LearningState = {
  version: 2;
  day: number;
  cards: StudyCard[];
  history: { cardId: string; day: number; rating: Rating }[];
};

export function initialLearningState(): LearningState {
  return {
    version: 2,
    day: 0,
    cards: snapshot.cards.map((card, i) => ({
      id: `karl-${card.cardId}`,
      sourceId: card.cardId,
      title: `Trivia · card ${String(i + 1).padStart(2, "0")}`,
      question: card.question,
      answer: card.answer,
      stability: card.initialStabilityDays,
      lastReview: card.lastReviewDay,
      due: 0,
      reviews: card.events.length,
    })),
    history: [],
  };
}

// Illustrative forgetting curve and interval rules, not FSRS or a fitted model.
export function recallEstimate(card: StudyCard, day: number) {
  return 0.9 ** (Math.max(0, day - card.lastReview) / card.stability);
}

export function reviewInterval(card: StudyCard, rating: Rating) {
  switch (rating) {
    case "again":
      return 1;
    case "hard":
      return Math.max(1, Math.round(card.stability * 0.7));
    case "good":
      return Math.max(2, Math.round(card.stability * 1.8));
    case "easy":
      return Math.max(3, Math.round(card.stability * 2.6));
  }
}

export function rateCard(
  state: LearningState,
  id: string,
  rating: Rating,
): LearningState {
  const card = state.cards.find((item) => item.id === id);
  if (!card || card.due > state.day) return state;
  const interval = reviewInterval(card, rating);
  return {
    ...state,
    cards: state.cards.map((item) =>
      item.id === id
        ? {
            ...item,
            stability:
              rating === "again" ? Math.max(1, item.stability * 0.5) : interval,
            due: state.day + interval,
            lastReview: state.day,
            reviews: item.reviews + 1,
          }
        : item,
    ),
    history: [...state.history, { cardId: id, day: state.day, rating }],
  };
}

export function restoreLearningState(raw: string): LearningState {
  const value = JSON.parse(raw);
  const finite = (v: unknown) => typeof v === "number" && Number.isFinite(v);
  if (
    value?.version !== 2 ||
    !Number.isInteger(value.day) ||
    value.day < 0 ||
    !Array.isArray(value.cards) ||
    !Array.isArray(value.history) ||
    !value.cards.every(
      (card: StudyCard) =>
        card &&
        [card.id, card.title, card.question, card.answer].every(
          (v) => typeof v === "string" && v.length > 0,
        ) &&
        finite(card.stability) &&
        card.stability > 0 &&
        finite(card.lastReview) &&
        Number.isInteger(card.due) &&
        card.due >= 0 &&
        Number.isInteger(card.reviews) &&
        card.reviews >= 0 &&
        (card.sourceId === undefined ||
          (typeof card.sourceId === "string" &&
            card.id === `karl-${card.sourceId}` &&
            snapshot.cards.some((source) => source.cardId === card.sourceId))),
    ) ||
    new Set(value.cards.map((card: StudyCard) => card.id)).size !==
      value.cards.length ||
    !value.history.every(
      (event: LearningState["history"][number]) =>
        event &&
        typeof event.cardId === "string" &&
        Number.isInteger(event.day) &&
        event.day >= 0 &&
        ["again", "hard", "good", "easy"].includes(event.rating),
    )
  ) {
    throw new Error("Saved demo is invalid.");
  }
  return value;
}
