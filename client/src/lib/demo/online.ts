// Per-user online layer for the /demo Memory-engine step (pure logic, no dependencies).
//
// Mirrors the frozen final model in fsrs_benchmark/srsb/online_adapt.py:
//   z = w . x,  p = sigmoid(z),  x[0] = GBDT raw margin (logit of the GBDT probability),
//   w0 = [1, 0, 0, 0, 0, 0]  so that with lr = 0 the output equals the GBDT,
//   update: w <- w + eta * (y - p) * x - eta * lam * (w - w0)   (L2 pull toward w0).
// The demo uses eta = 0.35 (sped up for the demo; the real run used 0.003).
// x[1..5] are scaled to about [-1, 1] by fixed constants, never by data statistics.

export const FEATURE_NAMES = [
  "GBDT margin",
  "bias",
  "recall rate today",
  "reviews today",
  "recency",
  "lapses",
] as const;

export const ETA = 0.35;
export const LAMBDA = 1e-3;
export const W0: readonly number[] = [1, 0, 0, 0, 0, 0];

// z is clamped so p stays strictly inside (0, 1) even after many updates.
const Z_LIMIT = 20;

export type Flashcard = {
  id: string;
  front: string;
  back: string;
  minutes: number;
  /** GBDT probability of recall before any answer today. */
  baseP: number;
  /** Features: [GBDT margin, bias, recall rate today, reviews today, recency, lapses]. */
  x: readonly number[];
  /** Illustrative 90% Bayesian band around baseP (not a fitted interval). */
  bayesLow: number;
  bayesHigh: number;
};

// Riya's 6 flashcards for the "Transformers & attention" topic.
// x[0] is logit(baseP) (checked by the tests); the other five are fixed, illustrative values.
export const RIYA_CARDS: readonly Flashcard[] = [
  {
    id: "attn-formula",
    front: "Write the scaled dot-product attention formula.",
    back: "softmax(QKᵀ / √d_k) V",
    minutes: 4,
    baseP: 0.62,
    x: [0.4895, 1, 0.1, 0.3, 0.2, 0.0],
    bayesLow: 0.5,
    bayesHigh: 0.74,
  },
  {
    id: "sqrt-dk",
    front: "Why divide the scores by √d_k?",
    back: "Large dot products push softmax into saturated regions, so gradients vanish.",
    minutes: 5,
    baseP: 0.48,
    x: [-0.08, 1, -0.2, 0.2, -0.1, -0.2],
    bayesLow: 0.36,
    bayesHigh: 0.6,
  },
  {
    id: "multi-head",
    front: "What does multi-head attention add over one head?",
    back: "Several learned projections attend to different subspaces in parallel, then concatenate.",
    minutes: 6,
    baseP: 0.71,
    x: [0.8954, 1, 0.3, 0.4, 0.5, 0.0],
    bayesLow: 0.6,
    bayesHigh: 0.8,
  },
  {
    id: "positional",
    front: "Why do transformers need positional encodings?",
    back: "Self-attention is permutation-invariant, so position must be injected into the inputs.",
    minutes: 3,
    baseP: 0.35,
    x: [-0.619, 1, -0.4, 0.1, -0.5, -0.4],
    bayesLow: 0.22,
    bayesHigh: 0.48,
  },
  {
    id: "causal-mask",
    front: "What is a causal mask for?",
    back: "It blocks attention to future positions so each token only sees its past.",
    minutes: 5,
    baseP: 0.8,
    x: [1.3863, 1, 0.5, 0.5, 0.7, 0.2],
    bayesLow: 0.7,
    bayesHigh: 0.9,
  },
  {
    id: "kv-cache",
    front: "Why does autoregressive decoding use a KV cache?",
    back: "Keys and values of past tokens are reused, so each new step avoids recomputing them.",
    minutes: 4,
    baseP: 0.55,
    x: [0.2007, 1, 0.0, 0.2, 0.0, -0.4],
    bayesLow: 0.42,
    bayesHigh: 0.68,
  },
];

export function sigmoid(z: number): number {
  const zc = Math.min(Z_LIMIT, Math.max(-Z_LIMIT, z));
  return 1 / (1 + Math.exp(-zc));
}

export function logit(p: number): number {
  return Math.log(p / (1 - p));
}

export function dot(w: readonly number[], x: readonly number[]): number {
  if (w.length !== x.length) {
    throw new Error(`dimension mismatch: w has ${w.length}, x has ${x.length}`);
  }
  let z = 0;
  for (let j = 0; j < w.length; j++) z += w[j] * x[j];
  return z;
}

/** Predicted recall probability for features x under weights w. */
export function predict(w: readonly number[], x: readonly number[]): number {
  return sigmoid(dot(w, x));
}

/**
 * One online step on the log loss of a single answer. Pure: returns new weights.
 * p must be the prediction made before the answer (predict(w, x)).
 */
export function update(
  w: readonly number[],
  x: readonly number[],
  y: 0 | 1,
  p: number,
  eta: number,
  lam: number = LAMBDA,
  w0: readonly number[] = W0,
): number[] {
  if (w.length !== x.length || w.length !== w0.length) {
    throw new Error("dimension mismatch in update");
  }
  const g = y - p;
  return w.map((wj, j) => wj + eta * (g * x[j] - lam * (wj - w0[j])));
}

/** Apply one answer on a card: predict, then update. */
export function answer(
  w: readonly number[],
  card: Flashcard,
  remembered: boolean,
  eta: number = ETA,
): number[] {
  const y = remembered ? 1 : 0;
  return update(w, card.x, y, predict(w, card.x), eta);
}

/** Cards with their current predicted recall, riskiest (lowest p) first; ties by id. */
export function riskQueue(
  cards: readonly Flashcard[],
  w: readonly number[],
): { card: Flashcard; p: number }[] {
  return cards
    .map((card) => ({ card, p: predict(w, card.x) }))
    .sort((a, b) => a.p - b.p || a.card.id.localeCompare(b.card.id));
}
