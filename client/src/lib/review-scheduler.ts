import artifact from "./review-booster.json";

export type ReviewEvent = { cardId: string; day: number; recalled: boolean };
type TreeNode =
  | number
  | [
      feature: number,
      threshold: number,
      missingLeft: boolean,
      left: number,
      right: number,
    ];

const model = artifact as {
  names: string[];
  trees: TreeNode[][];
  fsrs: number[];
  span: number;
  prior: number;
};

export const reviewTarget = 0.8;
const horizon = 21;
const w = model.fsrs;
const decay = w[20];
const factor = 0.9 ** (-1 / decay) - 1;
const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));
const retrievability = (gap: number, stability: number) =>
  (1 + (factor * gap) / stability) ** -decay;

function cardFeatures(history: ReviewEvent[]) {
  let stability = 0;
  let difficulty = 0;
  let remembered = 0;
  const lags: { gap: number; y: number; rating: number }[] = [];
  for (const [index, event] of history.entries()) {
    const y = Number(event.recalled);
    const rating = y ? 3 : 1;
    const gap = index ? event.day - history[index - 1].day : -1;
    remembered += y;
    lags.unshift({ gap, y, rating });
    if (index === 0) {
      stability = clamp(w[rating - 1], 0.001, 36500);
      difficulty = clamp(w[4] - Math.exp(w[5] * (rating - 1)) + 1, 1, 10);
      continue;
    }

    const recall = clamp(retrievability(gap, stability), 1e-4, 1);
    const success =
      stability *
      (1 +
        Math.exp(w[8]) *
          (11 - difficulty) *
          stability ** -w[9] *
          (Math.exp(w[10] * (1 - recall)) - 1));
    const failure = Math.min(
      w[11] *
        difficulty ** -w[12] *
        ((stability + 1) ** w[13] - 1) *
        Math.exp((1 - recall) * w[14]),
      stability / Math.exp(w[17] * w[18]),
    );
    const multiplier =
      Math.exp(w[17] * (rating - 3 + w[18])) * stability ** -w[19];
    const sameDay = stability * (y ? Math.max(1, multiplier) : multiplier);
    stability = clamp(
      gap === 0 ? sameDay : y ? success : failure,
      0.001,
      36500,
    );
    const easyDifficulty = w[4] - Math.exp(w[5] * 3) + 1;
    const delta = (-w[6] * (rating - 3) * (10 - difficulty)) / 9;
    difficulty = clamp(
      w[7] * easyDifficulty + (1 - w[7]) * (difficulty + delta),
      1,
      10,
    );
  }

  return {
    card_n: history.length,
    card_lapses: history.length - remembered,
    card_rate: remembered / history.length,
    g1: lags[0]?.rating,
    g2: lags[1]?.rating,
    g3: lags[2]?.rating,
    y1: lags[0]?.y,
    y2: lags[1]?.y,
    y3: lags[2]?.y,
    gap1: lags[0]?.gap,
    gap2: lags[1]?.gap,
    gap3: lags[2]?.gap,
    f_S: stability,
    f_D: difficulty,
  };
}

function userFeatures(events: ReviewEvent[]) {
  const alpha = 2 / (model.span + 1);
  let successes = 0;
  let informative = 0;
  let informativeSuccesses = 0;
  let smoothed = 0;
  let weight = 0;
  const lastByCard = new Map<string, number>();
  for (const { day, cardId, recalled } of events) {
    const previous = lastByCard.get(cardId);
    const result = Number(recalled);
    successes += result;
    if (previous !== undefined && day > previous) {
      informative++;
      informativeSuccesses += result;
    }
    smoothed = (1 - alpha) * smoothed + alpha * result;
    weight = (1 - alpha) * weight + alpha;
    lastByCard.set(cardId, day);
  }
  return {
    u_n: events.length,
    u_rate: successes / events.length,
    u_rate_inf: informativeSuccesses / informative,
    u_recent: smoothed / weight,
    u_day_n: 0,
    lex_rate: model.prior,
    lex_n: 0, // Newly generated cards do not have a benchmark item ID.
  };
}

function predictMargin(features: Record<string, number | undefined>) {
  const values = model.names.map((name) => features[name]);
  let raw = 0;
  for (const tree of model.trees) {
    let node = tree[0];
    while (typeof node !== "number") {
      const [index, threshold, missingLeft, left, right] = node;
      const value = values[index];
      const useLeft =
        value === undefined || Number.isNaN(value)
          ? missingLeft
          : value <= threshold;
      node = tree[useLeft ? left : right];
    }
    raw += node;
  }
  return raw;
}

const initialWeights = [1, 0, 0, 0, 0, 0];
const learningRate = 0.003;
const regularization = 0.001;
const sigmoid = (margin: number) => 1 / (1 + Math.exp(-margin));

function reviewFeatures(
  history: ReviewEvent[],
  past: ReviewEvent[],
  day: number,
) {
  const card = cardFeatures(history);
  const gap = day - history[history.length - 1].day;
  const recall = retrievability(gap, card.f_S);
  const clipped = clamp(recall, 1e-6, 1 - 1e-6);
  return {
    ...userFeatures(past),
    ...card,
    u_since_last: day - past[past.length - 1].day,
    el: gap,
    dt_days: gap,
    log_el: Math.log1p(gap),
    same_day: Number(gap === 0),
    since_first: day - history[0].day,
    f_R: recall,
    f_logitR: Math.log(clipped / (1 - clipped)),
    f_el_over_S: gap / card.f_S,
  };
}

function onlineInputs(
  features: ReturnType<typeof reviewFeatures>,
  recentRecall: number,
  sameDayRecall: number,
) {
  return [
    predictMargin(features),
    1,
    Math.log1p(features.el) / 3,
    (recentRecall - 0.5) * 4,
    features.f_logitR / 3,
    (sameDayRecall - 0.5) * 4,
  ];
}

function adjustedRecall(inputs: number[], weights: number[]) {
  return sigmoid(
    inputs.reduce((margin, value, i) => margin + value * weights[i], 0),
  );
}

export function forecastRecall(
  cardId: string,
  events: ReviewEvent[],
  day: number,
) {
  const past = events
    .filter((event) => event.day <= day)
    .sort((a, b) => a.day - b.day);
  const histories = new Map<string, ReviewEvent[]>();
  const weights = [...initialWeights];
  const prefix: ReviewEvent[] = [];
  const decay50 = 1 - 2 / 51;
  let recentSum = 0;
  let recentWeight = 0;
  let currentDay: number | undefined;
  let todaySum = 0;
  let todayCount = 0;
  const recentRecall = () => (recentWeight ? recentSum / recentWeight : 0.5);

  // Replay the single learner's reviews in order. Predict before applying
  // each answer and skip first reviews, matching benchmark Variant B.
  for (const event of past) {
    if (currentDay !== event.day) {
      currentDay = event.day;
      todaySum = 0;
      todayCount = 0;
    }
    const history = histories.get(event.cardId) ?? [];
    if (history.length) {
      const inputs = onlineInputs(
        reviewFeatures(history, prefix, event.day),
        recentRecall(),
        todayCount ? todaySum / todayCount : 0.5,
      );
      const answer = Number(event.recalled);
      const error = adjustedRecall(inputs, weights) - answer;
      for (const [i, feature] of inputs.entries()) {
        weights[i] -=
          learningRate *
          (error * feature + regularization * (weights[i] - initialWeights[i]));
      }
      recentSum = decay50 * recentSum + answer;
      recentWeight = decay50 * recentWeight + 1;
      todaySum += answer;
      todayCount++;
    }
    history.push(event);
    histories.set(event.cardId, history);
    prefix.push(event);
  }

  const history = histories.get(cardId);
  if (!history) return undefined;
  // Each candidate date shares the same weights and has no earlier reviews that day.
  return Array.from({ length: horizon }, (_, index) =>
    clamp(
      adjustedRecall(
        onlineInputs(
          reviewFeatures(history, past, day + index + 1),
          recentRecall(),
          0.5,
        ),
        weights,
      ),
      1e-6,
      1 - 1e-6,
    ),
  );
}

export function nextReviewInterval(
  cardId: string,
  events: ReviewEvent[],
  day: number,
) {
  const forecast = forecastRecall(cardId, events, day);
  if (!forecast) return 1;
  const firstBelow = forecast.findIndex((score) => score < reviewTarget);
  return firstBelow < 0 ? horizon : Math.max(1, firstBelow);
}
