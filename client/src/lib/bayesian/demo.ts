export const topics = [
  {
    title: "Bayes’ theorem",
    detail: "Conditional probability & prior beliefs",
    minutes: 20,
    question: "What connects a prior to a posterior?",
    answer:
      "Posterior ∝ likelihood × prior. The evidence normalizes the result.",
  },
  {
    title: "Linear algebra",
    detail: "Eigenvectors & transformations",
    minutes: 25,
    question: "What makes a vector an eigenvector of A?",
    answer:
      "A v = λ v: the transformation scales the vector without changing its direction.",
  },
  {
    title: "Gradient descent",
    detail: "Optimization & learning rates",
    minutes: 15,
    question: "Why move against the gradient?",
    answer:
      "The gradient points toward steepest local increase. A small step in the opposite direction reduces the objective.",
  },
  {
    title: "Markov chains",
    detail: "State transitions & stationarity",
    minutes: 20,
    question: "What is the Markov property?",
    answer:
      "The next state depends on the present state, conditionally independent of the earlier history.",
  },
  {
    title: "Hypothesis testing",
    detail: "Evidence & statistical decisions",
    minutes: 15,
    question: "What does a p-value describe?",
    answer:
      "Under the null model, the probability of a result at least as extreme as the observed result.",
  },
  {
    title: "Neural networks",
    detail: "Backpropagation & representations",
    minutes: 25,
    question: "What does backpropagation compute?",
    answer:
      "Gradients of the loss with respect to parameters, using the chain rule through the network.",
  },
];

export const initialHistories: number[][] = [
  [1, 1, 1, 0, 0, 0, 0, 0],
  [0, 0, 1, 1, 1, 1, 1, 1],
  [1, 0, 1, 0, 1, 0, 1, 0],
  [1, 1, 1, 1, 0, 0, 1, 0],
  [0, 1, 1, 1, 1, 1, 1, 1],
  [0, 0, 0, 1, 0, 0, 0, 0],
];

export type Prediction = {
  id: number;
  recall: number;
  low: number;
  high: number;
  static: number;
};
export type Fit = {
  items: Prediction[];
  draws: number;
  seed: number;
  phi: number;
  observations: number;
};

// A transparent risk-first heuristic; no claim of optimality or measured gain.
export function makeQueue(
  items: Prediction[],
  budget: number,
  adaptive = true,
) {
  const ordered = [...items].sort((a, b) =>
    adaptive ? a.recall - b.recall || a.id - b.id : a.id - b.id,
  );
  let used = 0;
  return ordered.filter((item) => {
    const duration = topics[item.id].minutes;
    if (used + duration > budget) return false;
    used += duration;
    return true;
  });
}
