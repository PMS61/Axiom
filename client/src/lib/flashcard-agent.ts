import { getGeminiResponse } from "./gemini";

export type Flashcard = {
  id: string;
  front: string;
  back: string;
};

export type FlashcardSet = {
  title: string;
  cards: Flashcard[];
};

function fallbackSet(title: string, description: string): FlashcardSet {
  return {
    title: `${title} Flashcards`,
    cards: [
      {
        id: "fc-1",
        front: `What is ${title}?`,
        back:
          description ||
          `${title} is a course topic with core concepts and practical applications.`,
      },
      {
        id: "fc-2",
        front: `Why is ${title} important?`,
        back: "It helps solve real-world problems and builds deeper domain understanding.",
      },
      {
        id: "fc-3",
        front: `Give one practical use of ${title}.`,
        back: "Apply it to a small project and explain the result in your own words.",
      },
      {
        id: "fc-4",
        front: "What is one common mistake learners make?",
        back: "Memorizing definitions without practicing retrieval and application.",
      },
      {
        id: "fc-5",
        front: "How should I revise this topic quickly?",
        back: "Use active recall: cover answers, test yourself, then verify.",
      },
    ],
  };
}

function normalizeSet(
  payload: unknown,
  title: string,
  description: string,
): FlashcardSet {
  const fallback = fallbackSet(title, description);
  if (!payload || typeof payload !== "object") return fallback;

  const candidate = payload as Partial<FlashcardSet>;
  const rawCards = Array.isArray(candidate.cards) ? candidate.cards : [];

  const cards = rawCards
    .map((entry, index) => {
      if (!entry || typeof entry !== "object") return null;
      const item = entry as Partial<Flashcard>;
      const front = typeof item.front === "string" ? item.front.trim() : "";
      const back = typeof item.back === "string" ? item.back.trim() : "";
      if (!front || !back) return null;

      return {
        id:
          typeof item.id === "string" && item.id.trim()
            ? item.id.trim()
            : `fc-${index + 1}`,
        front,
        back,
      };
    })
    .filter((card): card is Flashcard => Boolean(card))
    .slice(0, 24);

  return {
    title:
      typeof candidate.title === "string" && candidate.title.trim().length > 0
        ? candidate.title.trim()
        : fallback.title,
    cards: cards.length >= 3 ? cards : fallback.cards,
  };
}

export async function generateFlashcardSet(
  title: string,
  description: string,
  additionalInstructions: string,
): Promise<FlashcardSet> {
  const prompt = `
You are an expert instructional designer.
Create a high-quality flashcard set for study.

Inputs:
- Course title: ${title}
- Course description: ${description}
- Additional instructions: ${additionalInstructions || "None"}

Hard rules:
- Return ONLY valid JSON.
- No markdown and no code fences.
- Generate concise, clear, exam-friendly flashcards.
- Mix concept, application, and common-mistake cards.
- Keep card front short and back precise.
- Generate between 10 and 18 cards.

JSON format:
{
  "title": "string",
  "cards": [
    { "id": "fc-1", "front": "question", "back": "answer" }
  ]
}
`;

  try {
    const raw = await getGeminiResponse(prompt, true);
    const parsed = JSON.parse(raw);
    return normalizeSet(parsed, title, description);
  } catch (error) {
    console.error("[FlashcardAgent] generation failed:", error);
    return fallbackSet(title, description);
  }
}
