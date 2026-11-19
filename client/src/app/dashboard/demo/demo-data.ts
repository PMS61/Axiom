import {
  createStudyReplay,
  type StudyState,
} from "@/app/learn/scheduled-reviews";
import { computeDiversityEntropy } from "@/lib/constraints";
import { BASE_AXIOMS, buildSections } from "@/lib/energyModel";
import { computeCL } from "@/lib/engine";
import { forecastRecall, reviewTarget } from "@/lib/review-scheduler";
import type { DaySchedule, RoadmapDAGNode, Task } from "@/lib/types";
import preset from "./preset.json";

export { preset };

export const roadmapTitle = preset.title;
export const learnerName = "Alex";

export const milestones: RoadmapDAGNode[] = [
  {
    id: "spaced-repetition",
    title: "Spaced repetition",
    description:
      "Explain why review timing matters and how recall changes with time.",
    type: "course",
    difficulty: 2,
    estimatedMinutes: 20,
    priority: "normal",
    sequence: 0,
    prerequisites: [],
    masteryScore: 1,
    subject: "Axiom",
    nodeState: "mastered",
  },
  {
    id: "recall-prediction",
    title: "Recall prediction and FSRS",
    description:
      "Connect review history, forgetting curves, and the FSRS baseline.",
    type: "course",
    difficulty: 3,
    estimatedMinutes: 20,
    priority: "normal",
    sequence: 1,
    prerequisites: ["spaced-repetition"],
    masteryScore: 1,
    subject: "Axiom",
    nodeState: "mastered",
  },
  {
    id: "adaptive-scheduling",
    title: "GBDT and online adaptation",
    description:
      "Trace how the GBDT predicts recall and how each answer updates the learner’s weights.",
    type: "course",
    difficulty: 3,
    estimatedMinutes: 25,
    priority: "normal",
    sequence: 2,
    prerequisites: ["recall-prediction"],
    masteryScore: 0.5,
    subject: "Axiom",
    nodeState: "in_progress",
  },
  {
    id: "axiom-quiz",
    title: "Axiom concepts quiz",
    description:
      "Recall the model, baseline, online update, and how predictions are evaluated.",
    type: "assessment",
    difficulty: 4,
    estimatedMinutes: 15,
    priority: "normal",
    sequence: 3,
    prerequisites: ["adaptive-scheduling"],
    masteryScore: 0,
    subject: "Axiom",
    nodeState: "locked",
  },
];

export function scheduleDemoAnswer(
  card: { word: string },
  answers: StudyState["answers"],
  day: number,
) {
  const events = [
    ...preset.cards.flatMap(({ word, practice }) =>
      practice.map(({ daysAgo, recalled }) => ({
        cardId: word,
        day: -daysAgo,
        recalled,
      })),
    ),
    ...answers.map((event) => ({
      cardId: event.word,
      day: event.day,
      recalled: event.answer === "remembered",
    })),
  ];
  const forecast = forecastRecall(card.word, events, day);
  if (!forecast) return { interval: 1, recall: 0 };
  const firstBelow = forecast.findIndex((score) => score < reviewTarget);
  const interval = firstBelow < 0 ? forecast.length : Math.max(1, firstBelow);
  return { interval, recall: forecast[interval - 1] };
}

export const study = createStudyReplay({
  ...preset,
  storageVersion: "axiom-concepts-online-1",
  scheduleAnswer: scheduleDemoAnswer,
  continueAfterReplay: true,
});

export function addDays(start: string, offset: number) {
  const date = new Date(`${start}T12:00:00Z`);
  date.setUTCDate(date.getUTCDate() + offset);
  return date.toISOString().slice(0, 10);
}

const baselineTasks = [
  {
    id: "reading",
    name: "Read · Why spaced repetition works",
    studyDay: 1,
    slot: 37,
    minutes: 15,
    completed: true,
  },
  {
    id: "model-study",
    name: "Study · How the GBDT predicts recall",
    studyDay: 1,
    slot: 64,
    minutes: 25,
    completed: false,
  },
  {
    id: "notes",
    name: "Write · Compare FSRS and GBDT recall forecasts",
    studyDay: 2,
    slot: 62,
    minutes: 20,
    completed: false,
  },
] as const;

export function upcomingReviewId(title: string) {
  return `review:${title}`;
}

export type CalendarBlock = {
  id: string;
  name: string;
  studyDay: number;
  slot: number;
  minutes: number;
  completed: boolean;
};

export function buildCalendar(
  startDate: string,
  state: StudyState,
  generated: CalendarBlock[] = [],
) {
  const queue = study.reviewQueue(state);
  const today = addDays(startDate, state.day - 1);
  const blocks = [
    ...baselineTasks.filter((task) => task.studyDay >= state.day),
    ...generated,
    ...queue.map((card) => ({
      id: upcomingReviewId(card.word),
      name: `Review · ${card.word}`,
      studyDay: card.due,
      slot: 42 + preset.cards.findIndex((item) => item.word === card.word),
      minutes: 15,
      completed: false,
    })),
    ...state.answers
      .filter((event) => event.day === state.day)
      .map((event) => ({
        id: `reviewed:${event.word}`,
        name: `Review · ${event.word}`,
        studyDay: event.day,
        slot: 42 + preset.cards.findIndex((item) => item.word === event.word),
        minutes: 15,
        completed: true,
      })),
  ].filter((block) => block.studyDay >= state.day);

  const tasks: Task[] = blocks.map((block, sequence) => {
    const type = /^(review|generated):?/.test(block.id)
      ? "revision"
      : "learning";
    const difficulty = type === "revision" ? 2 : 3;
    const etask = computeCL(difficulty, block.minutes, null, type, "normal");
    return {
      id: block.id,
      name: block.name,
      type,
      difficulty,
      completionTime: block.minutes,
      priority: "normal",
      state: block.completed ? "completed" : "scheduled",
      subject: roadmapTitle,
      etask: etask.total,
      etaskBreakdown: etask,
      sequence,
      scheduledSlot: {
        day: block.studyDay - state.day,
        startSlot: block.slot,
        endSlot: block.slot + Math.ceil(block.minutes / 15),
        fitnessScore: 0,
        reasoningSteps: [],
      },
      createdAt: `${startDate}T00:00:00Z`,
    };
  });

  const sections = buildSections();
  const scheduledOffsets = tasks.flatMap(
    (task) => task.scheduledSlot?.day ?? [],
  );
  const offsets = [...new Set([0, 1, ...scheduledOffsets])].sort(
    (a, b) => a - b,
  );
  const days: DaySchedule[] = offsets.map((dayOffset) => {
    const todayTasks = tasks.filter(
      (task) => task.scheduledSlot?.day === dayOffset,
    );
    const totalAxiomsUsed = todayTasks.reduce(
      (sum, task) => sum + task.etask,
      0,
    );
    return {
      dayOffset,
      date: addDays(today, dayOffset),
      totalAxiomsUsed,
      totalAxiomsRemaining: BASE_AXIOMS - totalAxiomsUsed,
      diversityEntropy: computeDiversityEntropy(todayTasks),
      loadAcceptable: totalAxiomsUsed <= BASE_AXIOMS,
      sections: sections.map((section) => {
        const entries = todayTasks
          .filter((task) => {
            const hour = (task.scheduledSlot?.startSlot ?? 0) / 4;
            return hour >= section.startHour && hour < section.endHour;
          })
          .sort(
            (a, b) =>
              (a.scheduledSlot?.startSlot ?? 0) -
              (b.scheduledSlot?.startSlot ?? 0),
          );
        const axiomUsed = entries.reduce((sum, task) => sum + task.etask, 0);
        return {
          section: section.name,
          axiomBudget: section.axiomBudget,
          axiomUsed,
          axiomRemaining: section.axiomBudget - axiomUsed,
          tasks: entries.map((task) => ({
            taskId: task.id,
            taskName: task.name,
            completionTime: task.completionTime,
            axiomCost: task.etask,
            axiomGain: 0,
            isRecreational: false,
            startSlot: task.scheduledSlot?.startSlot,
          })),
        };
      }),
    };
  });

  return { tasks, days, today };
}

export function formatStudyDate(value: string) {
  return new Intl.DateTimeFormat("en-IN", {
    day: "numeric",
    month: "short",
    timeZone: "UTC",
  }).format(new Date(`${value}T12:00:00Z`));
}
