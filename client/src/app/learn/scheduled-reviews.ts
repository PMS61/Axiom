import snapshot from "./replay.json";

export type Answer = "remembered" | "missed";

type ReviewPlan = {
  interval: number;
  recall: number;
  remembered?: ReviewPlan;
  missed?: ReviewPlan;
};

export type StudyCard = {
  word: string;
  meaning: string;
  practice: { daysAgo: number; recalled: boolean }[];
  forecast: number[];
  plan: ReviewPlan;
};
export type StudyState = {
  day: number;
  answers: { word: string; day: number; answer: Answer }[];
};

export const replay = snapshot;
export function createStudyReplay<Card extends Omit<StudyCard, "forecast">>({
  cards,
  reviewLimit,
  storageVersion,
  continueAfterReplay = false,
  scheduleAnswer,
}: {
  cards: Card[];
  reviewLimit: number;
  storageVersion: string;
  continueAfterReplay?: boolean;
  scheduleAnswer?: (
    card: Card,
    answers: StudyState["answers"],
    day: number,
  ) => ReviewPlan;
}) {
  function reviewQueue(state: StudyState) {
    if (scheduleAnswer) {
      return cards
        .map((card) => {
          const answers = state.answers.filter(
            (event) => event.word === card.word,
          );
          const last = answers.at(-1);
          const throughLast = last
            ? state.answers.slice(0, state.answers.lastIndexOf(last) + 1)
            : [];
          const plan = last
            ? scheduleAnswer(card, throughLast, last.day)
            : card.plan;
          return {
            ...card,
            due: (last?.day ?? 0) + plan.interval,
            plan,
            answers,
            canReview: answers.length < reviewLimit,
          };
        })
        .sort((a, b) => a.due - b.due || a.plan.recall - b.plan.recall);
    }
    return cards
      .map(({ plan: initialPlan, ...card }) => {
        let plan: ReviewPlan = initialPlan;
        let due = plan.interval;
        const answers = state.answers.filter(
          (event) => event.word === card.word,
        );
        for (const event of answers) {
          const next = plan[event.answer];
          if (!next) break;
          plan = next;
          due = event.day + next.interval;
        }
        return {
          ...card,
          due,
          plan,
          answers,
          canReview: Boolean(plan.remembered),
        };
      })
      .sort((a, b) => a.due - b.due || a.plan.recall - b.plan.recall);
  }

  function recordAnswer(state: StudyState, word: string, answer: Answer) {
    const card = reviewQueue(state).find((item) => item.word === word);
    if (!card?.canReview || card.due !== state.day) return state;

    return {
      ...state,
      answers: [...state.answers, { word, day: state.day, answer }],
    };
  }

  function nextReviewDay(state: StudyState) {
    return reviewQueue(state).find((card) => card.canReview)?.due;
  }

  function saveStudy(state: StudyState) {
    return JSON.stringify({ version: storageVersion, ...state });
  }

  function restoreStudy(raw: string): StudyState {
    const value = JSON.parse(raw);
    if (
      value?.version !== storageVersion ||
      !Number.isInteger(value.day) ||
      value.day < 1 ||
      !Array.isArray(value.answers) ||
      value.answers.length > reviewLimit * cards.length
    ) {
      return initialStudy();
    }

    let state = initialStudy();
    for (const event of value.answers) {
      if (
        !event ||
        typeof event.word !== "string" ||
        (event.answer !== "remembered" && event.answer !== "missed") ||
        event.day !== nextReviewDay(state) ||
        event.day < state.day ||
        event.day > value.day
      ) {
        return initialStudy();
      }
      state = { ...state, day: event.day };
      const updated = recordAnswer(state, event.word, event.answer);
      if (updated === state) return initialStudy();
      state = updated;
    }

    const next = nextReviewDay(state);
    if (
      value.day >
      (next ?? (continueAfterReplay ? state.day + 36500 : state.day))
    ) {
      return initialStudy();
    }
    return { ...state, day: value.day };
  }

  return { reviewQueue, recordAnswer, nextReviewDay, saveStudy, restoreStudy };
}

export function initialStudy(): StudyState {
  return { day: 1, answers: [] };
}

export const {
  reviewQueue,
  recordAnswer,
  nextReviewDay,
  saveStudy,
  restoreStudy,
} = createStudyReplay({ ...snapshot, storageVersion: "duolingo-1" });
