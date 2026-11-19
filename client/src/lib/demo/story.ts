// Seeded story for the /demo route. Persona: Riya, preparing for an ML interview.
// Pure data and pure functions, no imports, so node --experimental-strip-types can load it.
// Everything here is illustrative seed data except where noted.

export type AgentId =
  | "roadmap"
  | "course"
  | "assessment"
  | "cheat-sheet"
  | "flashcard"
  | "short-bit"
  | "story";

export const AGENTS: { id: AgentId; label: string }[] = [
  { id: "roadmap", label: "Roadmap" },
  { id: "course", label: "Course" },
  { id: "assessment", label: "Assessment" },
  { id: "cheat-sheet", label: "Cheat-sheet" },
  { id: "flashcard", label: "Flashcard" },
  { id: "short-bit", label: "Short-bit" },
  { id: "story", label: "Story" },
];

// ── Step 1: trends ───────────────────────────────────────

export type Source = "Hacker News" | "Reddit" | "NewsAPI";

export interface Topic {
  id: string;
  name: string;
  source: Source;
  /** Frequency, engagement, recency. Each in [0, 1]. */
  F: number;
  E: number;
  R: number;
  /** 12 points, oldest first, each in [0, 1]. */
  spark: number[];
}

export function scoreTopic(t: { F: number; E: number; R: number }): number {
  return Math.round(100 * (0.4 * t.F + 0.4 * t.E + 0.2 * t.R));
}

export const TOPICS: Topic[] = [
  {
    id: "transformers",
    name: "Transformers & attention",
    source: "Hacker News",
    F: 0.95,
    E: 0.9,
    R: 0.92,
    spark: [
      0.22, 0.25, 0.31, 0.36, 0.42, 0.5, 0.58, 0.66, 0.74, 0.82, 0.88, 0.92,
    ],
  },
  {
    id: "rag",
    name: "RAG systems",
    source: "Reddit",
    F: 0.85,
    E: 0.8,
    R: 0.88,
    spark: [
      0.18, 0.2, 0.27, 0.33, 0.4, 0.47, 0.56, 0.63, 0.7, 0.76, 0.81, 0.84,
    ],
  },
  {
    id: "diffusion",
    name: "Diffusion models",
    source: "NewsAPI",
    F: 0.78,
    E: 0.74,
    R: 0.7,
    spark: [0.3, 0.34, 0.4, 0.42, 0.48, 0.52, 0.55, 0.6, 0.64, 0.68, 0.7, 0.72],
  },
  {
    id: "rlhf",
    name: "RL from human feedback",
    source: "Hacker News",
    F: 0.72,
    E: 0.58,
    R: 0.8,
    spark: [0.4, 0.38, 0.42, 0.45, 0.44, 0.5, 0.55, 0.52, 0.6, 0.63, 0.66, 0.7],
  },
  {
    id: "vector-db",
    name: "Vector databases",
    source: "Reddit",
    F: 0.66,
    E: 0.7,
    R: 0.58,
    spark: [
      0.35, 0.4, 0.38, 0.46, 0.5, 0.49, 0.55, 0.58, 0.57, 0.62, 0.64, 0.66,
    ],
  },
  {
    id: "gnn",
    name: "Graph neural nets",
    source: "NewsAPI",
    F: 0.55,
    E: 0.6,
    R: 0.65,
    spark: [
      0.42, 0.4, 0.44, 0.43, 0.47, 0.48, 0.5, 0.52, 0.51, 0.55, 0.57, 0.59,
    ],
  },
];

/** The topic with the highest score. Selected by default. */
export function defaultTopicId(): string {
  return TOPICS.reduce((best, t) =>
    scoreTopic(t) > scoreTopic(best) ? t : best,
  ).id;
}

// ── Step 2: roadmap ──────────────────────────────────────

export interface RoadmapNode {
  id: string;
  label: string;
  /** 0 = prerequisites (left) ... 3 = advanced (right). */
  col: number;
  row: number;
  outputs: Record<AgentId, string>;
}

export interface RoadmapEdge {
  from: string;
  to: string;
}

export interface Roadmap {
  topicId: string;
  nodes: RoadmapNode[];
  edges: RoadmapEdge[];
}

function outs(titles: string[]): Record<AgentId, string> {
  const result = {} as Record<AgentId, string>;
  AGENTS.forEach((a, i) => {
    result[a.id] = titles[i];
  });
  return result;
}

const TRANSFORMER_NODES: RoadmapNode[] = [
  {
    id: "linalg",
    label: "Linear algebra",
    col: 0,
    row: 0,
    outputs: outs([
      "Vectors, matrices and dot products in 2 hours",
      "Linear algebra for attention, 25-min read",
      "5 checks on matrix shapes",
      "Shape rules for matrix products",
      "What is a dot product? (3 cards)",
      "Why a dot product measures similarity",
      "The geometry of 'how aligned are these?'",
    ]),
  },
  {
    id: "embed",
    label: "Embeddings",
    col: 0,
    row: 1,
    outputs: outs([
      "From one-hot tokens to dense vectors",
      "Embeddings as learned lookup tables",
      "4 checks: embedding dimensions",
      "Embedding lookup in one diagram",
      "What does an embedding table store? (3 cards)",
      "Words become points in space",
      "The day word2vec made maths of meaning",
    ]),
  },
  {
    id: "rnn",
    label: "RNN limits",
    col: 0,
    row: 2,
    outputs: outs([
      "Why recurrence struggles with long context",
      "Sequence models and the bottleneck",
      "3 checks: vanishing gradients",
      "RNN vs attention at a glance",
      "Why does an RNN forget? (3 cards)",
      "One hidden state is not enough",
      "The bottleneck that started it all",
    ]),
  },
  {
    id: "attn",
    label: "Attention (QKV)",
    col: 1,
    row: 1,
    outputs: outs([
      "Queries, keys, values, from first principles",
      "Scaled dot-product attention, 40-min course",
      "6 questions on the softmax step",
      "QKV formula on one page",
      "Q, K, V: what each one does (6 cards)",
      "Attention as a weighted lookup",
      "Why 'attention is all you need' was bold",
    ]),
  },
  {
    id: "mh",
    label: "Multi-head + pos.",
    col: 2,
    row: 1,
    outputs: outs([
      "Several heads, and where word order goes",
      "Multi-head attention and positional encoding",
      "4 questions on head dimensions",
      "Head split and positional formulas",
      "Why multiple heads? (4 cards)",
      "Heads specialise, position is added",
      "Telling a model where each word sits",
    ]),
  },
  {
    id: "kv",
    label: "KV cache",
    col: 3,
    row: 1,
    outputs: outs([
      "Fast decoding with a KV cache, and its cost",
      "KV cache and memory-efficient attention",
      "4 questions on latency trade-offs",
      "KV cache sizing in one table",
      "Why cache keys and values? (3 cards)",
      "Remembering so you do not recompute",
      "The trick that makes chatbots fast",
    ]),
  },
];

const TRANSFORMER_EDGES: RoadmapEdge[] = [
  { from: "linalg", to: "attn" },
  { from: "embed", to: "attn" },
  { from: "rnn", to: "attn" },
  { from: "attn", to: "mh" },
  { from: "mh", to: "kv" },
];

const GENERIC_LABELS = [
  "Background",
  "Core idea",
  "Key parts",
  "Build it",
  "Trade-offs",
  "Open problems",
];

/** Roadmap for a topic. Transformers has real seeded content; others reuse a generic plan. */
export function roadmapFor(topicId: string): Roadmap {
  if (topicId === "transformers") {
    return { topicId, nodes: TRANSFORMER_NODES, edges: TRANSFORMER_EDGES };
  }
  const name = TOPICS.find((t) => t.id === topicId)?.name ?? topicId;
  const nodes: RoadmapNode[] = TRANSFORMER_NODES.map((n, i) => ({
    id: n.id,
    label: GENERIC_LABELS[i],
    col: n.col,
    row: n.row,
    outputs: outs(
      AGENTS.map((a) => `${a.label}: ${GENERIC_LABELS[i]} (${name})`),
    ),
  }));
  return { topicId, nodes, edges: TRANSFORMER_EDGES };
}

// ── Step 3: scheduler ────────────────────────────────────
// Times are minutes after midnight.

export const DAY_START = 7 * 60 + 30; // 07:30
export const DAY_END = 23 * 60 + 30; // 23:30

export interface Block {
  id: string;
  label: string;
  start: number;
  end: number;
}

export const PEAK_WINDOWS: Block[] = [
  { id: "peak-am", label: "Peak", start: 10 * 60, end: 13 * 60 },
  { id: "peak-pm", label: "Peak", start: 15 * 60, end: 18 * 60 },
];

export const LOW_WINDOWS: Block[] = [
  { id: "low-pm", label: "Low", start: 14 * 60, end: 15 * 60 },
  { id: "low-eve", label: "Low", start: 19 * 60, end: 20 * 60 },
];

export const COMMITMENTS: Block[] = [
  { id: "standup", label: "Standup", start: 10 * 60, end: 10 * 60 + 30 },
  { id: "capstone", label: "Capstone meeting", start: 16 * 60, end: 17 * 60 },
];

export const EXCLUSIONS: Block[] = [
  { id: "workout", label: "Workout", start: 8 * 60, end: 9 * 60 },
  { id: "lunch", label: "Lunch", start: 13 * 60, end: 14 * 60 },
  { id: "dinner", label: "Dinner", start: 20 * 60, end: 21 * 60 },
];

/** Illustrative bandwidth profile (minutes, value in [0, 1]), 30-minute steps. */
export const BANDWIDTH: [number, number][] = [
  [450, 0.4],
  [480, 0.55],
  [510, 0.6],
  [540, 0.7],
  [570, 0.78],
  [600, 0.92],
  [630, 0.95],
  [660, 0.92],
  [690, 0.9],
  [720, 0.85],
  [750, 0.8],
  [780, 0.65],
  [810, 0.6],
  [840, 0.4],
  [870, 0.35],
  [900, 0.85],
  [930, 0.9],
  [960, 0.8],
  [990, 0.75],
  [1020, 0.82],
  [1050, 0.85],
  [1080, 0.7],
  [1110, 0.55],
  [1140, 0.3],
  [1170, 0.28],
  [1200, 0.45],
  [1230, 0.55],
  [1260, 0.6],
  [1290, 0.5],
  [1320, 0.4],
  [1350, 0.35],
  [1380, 0.3],
  [1410, 0.25],
];

export interface Task {
  id: string;
  label: string;
  start: number;
  end: number;
  /** True when the task is fed by the memory engine (step 4). */
  fromMemory: boolean;
  reasoning: string[];
}

export const TASKS: Task[] = [
  {
    id: "read",
    label: "Read: Attention Is All You Need, section 3",
    start: 9 * 60,
    end: 9 * 60 + 50,
    fromMemory: false,
    reasoning: [
      "Dense new notation, so cognitive load is high (0.62)",
      "Energy is rising from 0.70 at 09:00 towards the first peak",
      "Placed 09:00–09:50: after the workout, before standup",
      "A 50-minute slot fits the paper section",
    ],
  },
  {
    id: "flashcards",
    label: "Review: 6 flashcards (Transformers)",
    start: 10 * 60 + 30,
    end: 11 * 60,
    fromMemory: true,
    reasoning: [
      "Memory engine: 6 Transformer cards are due today",
      "Three of them have predicted recall below 0.70",
      "Flashcard review is short and recall-heavy, so it fits a gap",
      "Placed 10:30–11:00, between standup and the course block",
    ],
  },
  {
    id: "course",
    label: "Course: scaled dot-product attention",
    start: 11 * 60,
    end: 12 * 60,
    fromMemory: false,
    reasoning: [
      "High cognitive load (0.9) for new derivations",
      "Peak window 10:00–13:00 has bandwidth 0.92–0.95",
      "Placed in the peak window at 11:00",
      "Lunch at 13:00 is a hard exclusion, so the block ends by 12:00",
    ],
  },
  {
    id: "problems",
    label: "Problem set: 3 attention derivations",
    start: 12 * 60,
    end: 12 * 60 + 50,
    fromMemory: false,
    reasoning: [
      "Derivations need sustained focus (load 0.85)",
      "Still inside the 10:00–13:00 peak window",
      "Starts at 12:00 and ends before lunch at 13:00",
    ],
  },
  {
    id: "cheatsheet",
    label: "Cheat-sheet: transformer block shapes",
    start: 14 * 60,
    end: 14 * 60 + 30,
    fromMemory: false,
    reasoning: [
      "Recall-only task with low load (0.3)",
      "The 14:00–15:00 low window suits light work",
      "Placed at 14:00, straight after lunch",
    ],
  },
  {
    id: "mock",
    label: "Mock interview: explain the KV cache",
    start: 17 * 60,
    end: 17 * 60 + 45,
    fromMemory: false,
    reasoning: [
      "Spoken explanation needs high working memory (0.8)",
      "The 15:00–18:00 peak window has bandwidth 0.75–0.85",
      "Starts after the capstone meeting ends at 17:00",
      "Ends 17:45, leaving a buffer before dinner",
    ],
  },
];

export function formatTime(minutes: number): string {
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}
