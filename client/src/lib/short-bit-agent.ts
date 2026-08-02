import { getGeminiResponse } from "./gemini";

export type ShortBitItem = {
  id: string;
  imageKeyword: string;
  blogContent: string;
};

export type ShortBitSet = {
  title: string;
  items: ShortBitItem[];
};

const COMMON_KEYWORD_WORDS = new Set([
  "book",
  "notebook",
  "pen",
  "pencil",
  "paper",
  "laptop",
  "keyboard",
  "screen",
  "phone",
  "desk",
  "table",
  "chair",
  "classroom",
  "whiteboard",
  "teacher",
  "student",
  "school",
  "library",
  "coffee",
  "clock",
  "window",
  "lamp",
  "road",
  "city",
  "home",
  "room",
  "computer",
  "headphones",
  "notes",
  "backpack",
  "calculator",
  "folder",
  "checklist",
]);

const COMMON_KEYWORD_FALLBACKS = [
  "book",
  "desk",
  "notebook",
  "laptop",
  "classroom",
  "whiteboard",
  "coffee",
  "road",
];

function simplifyImageKeyword(input: string, index = 0): string {
  const cleaned = input
    .toLowerCase()
    .replace(/[^a-z\s]/g, " ")
    .trim();

  const words = cleaned
    .split(/\s+/)
    .filter((word) => word.length > 1 && COMMON_KEYWORD_WORDS.has(word));

  if (words.length === 1) {
    return words[0];
  }

  if (words.length > 1) {
    return words[0];
  }

  return COMMON_KEYWORD_FALLBACKS[index % COMMON_KEYWORD_FALLBACKS.length];
}

function fallbackSet(title: string, description: string): ShortBitSet {
  return {
    title: `${title} · Short Bit`,
    items: [
      {
        id: "sb-1",
        imageKeyword: "desk",
        blogContent: description
          ? `${description.trim()} Start by turning that idea into one clear concept you can explain without jargon. Use a simple real-life analogy and rewrite it in your own words so it feels intuitive instead of abstract. Then test yourself with one practical mini-example to make sure the understanding sticks.\n\nTurn that idea into one tiny action you can do in the next 10 minutes, and define a visible outcome before you begin. Keep your scope intentionally small, then review what worked and what felt confusing right after the attempt. This loop builds clarity faster than long passive reading.`
          : `A sharp understanding of ${title} starts with one simple idea and one practical example. Break the concept into plain language, identify the key moving parts, and map each part to something you already know. That bridge reduces cognitive load and makes advanced details easier to absorb.\n\nTurn that idea into one tiny action you can do in the next 10 minutes, and define a visible outcome before you begin. Keep your scope intentionally small, then review what worked and what felt confusing right after the attempt. This loop builds clarity faster than long passive reading.`,
      },
      {
        id: "sb-2",
        imageKeyword: "whiteboard",
        blogContent:
          "When a concept feels hard, explain it out loud in plain language before diving into details. Speak as if you are teaching a beginner, and avoid technical shortcuts in your first pass. This reveals what you truly understand and what only feels familiar.\n\nThen re-open your notes and map each term to your own words with one concrete example per term. Mark the exact sentence where confusion appears, and immediately rewrite it in clearer language. This single rewrite habit compounds into stronger retention.",
      },
      {
        id: "sb-3",
        imageKeyword: "notebook",
        blogContent:
          "Keep each revision block short, specific, and outcome-based. Choose one objective, set a quick timer, and define what done looks like before you start. This keeps attention locked and prevents shallow multitasking.\n\nSmall feedback loops improve retention and reduce burnout when you measure progress by clarity, not hours. End each block with a two-line summary and one unresolved question to carry forward. That makes your next session start with momentum.",
      },
      {
        id: "sb-4",
        imageKeyword: "checklist",
        blogContent:
          "Always end with one concrete next step that is small enough to begin immediately. If a task feels heavy, split it until the first action takes less than five minutes. Consistent starts beat occasional perfect sessions.\n\nMomentum matters more than perfect planning when building a habit over time. Keep a visible checklist, celebrate completion signals, and prune low-impact tasks quickly. This protects focus and keeps progress emotionally sustainable.",
      },
    ],
  };
}

function normalizeSet(
  payload: unknown,
  title: string,
  description: string,
): ShortBitSet {
  const fallback = fallbackSet(title, description);
  if (!payload || typeof payload !== "object") return fallback;

  const candidate = payload as Partial<ShortBitSet>;
  const rawItems = Array.isArray(candidate.items) ? candidate.items : [];

  const items = rawItems
    .map((entry, index) => {
      if (!entry || typeof entry !== "object") return null;
      const item = entry as Partial<ShortBitItem>;
      const imageKeywordRaw =
        typeof item.imageKeyword === "string" ? item.imageKeyword.trim() : "";
      const blogContent =
        typeof item.blogContent === "string" ? item.blogContent.trim() : "";
      if (!blogContent) return null;

      const imageKeyword = simplifyImageKeyword(imageKeywordRaw, index);

      return {
        id:
          typeof item.id === "string" && item.id.trim().length > 0
            ? item.id.trim()
            : `sb-${index + 1}`,
        imageKeyword,
        blogContent,
      };
    })
    .filter((item): item is ShortBitItem => Boolean(item))
    .slice(0, 4);

  return {
    title:
      typeof candidate.title === "string" && candidate.title.trim().length > 0
        ? candidate.title.trim()
        : fallback.title,
    items: items.length >= 3 ? items : fallback.items,
  };
}

export async function generateShortBitSet(
  title: string,
  description: string,
  additionalInstructions: string,
): Promise<ShortBitSet> {
  const prompt = `
You are an expert micro-content writer.
Generate short educational visual bites.

Inputs:
- Title: ${title}
- Description: ${description}
- Additional instructions: ${additionalInstructions || "None"}

Task:
- Generate an array of short items.
- Each item must include:
  1) imageKeyword: one common everyday keyword for image generation
  2) blogContent: exactly 2 detailed paragraphs in clean flow

Hard rules:
- Return ONLY valid JSON.
- No markdown, no code fences, no explanations.
- blogContent must be exactly 2 paragraphs separated by one blank line (\\n\\n).
- Keep each paragraph substantial (about 3-5 sentences).
- Total blogContent should feel meaningful and expanded, not brief.
- imageKeyword must be exactly 1 word.
- Use only simple common words (easy visual nouns).
- Avoid abstract, technical, poetic, or rare wording.
- Good imageKeyword examples: "book", "laptop", "coffee", "notebook", "classroom", "whiteboard", "road", "desk".
- Bad imageKeyword examples: "epistemic reflection", "cognitive resonance", "metacognitive lattice", "surreal symbolism".
- Generate between 3 and 4 items.

JSON shape:
{
  "title": "string",
  "items": [
    {
      "id": "sb-1",
      "imageKeyword": "string",
      "blogContent": "string"
    }
  ]
}
`;

  try {
    const raw = await getGeminiResponse(prompt, true);
    const parsed = JSON.parse(raw);
    if (parsed?.error) {
      throw new Error(parsed.message || parsed.error);
    }
    return normalizeSet(parsed, title, description);
  } catch (error) {
    console.error("[ShortBitAgent] generation failed:", error);
    throw error;
  }
}
