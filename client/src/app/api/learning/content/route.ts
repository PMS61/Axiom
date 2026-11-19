import { generateCourseComplete } from "@/lib/course-agent";
import { generateFlashcardSet } from "@/lib/flashcard-agent";
import {
  deckSchema,
  learningRequestSchema,
  roadmapSchema,
  slidesSchema,
} from "@/lib/learning-content";
import { generateRoadmapDAG } from "@/lib/roadmap-agent";

export const maxDuration = 90;

let pending = 0;
const maxPending = 3;

export async function POST(request: Request) {
  const parsed = learningRequestSchema.safeParse(
    await request.json().catch(() => null),
  );
  if (!parsed.success) {
    return Response.json(
      { error: "Check the topic, learning goal, and lesson inputs." },
      { status: 400 },
    );
  }
  if (!process.env.GEMINI_API_KEY) {
    return Response.json(
      { error: "Gemini API access is unavailable on this server." },
      { status: 503 },
    );
  }
  if (pending >= maxPending) {
    return Response.json(
      { error: "Generation is busy. Try again in a moment." },
      { status: 429 },
    );
  }

  const { topic, kind } = parsed.data;
  pending++;
  try {
    if (kind === "lesson") {
      const { slides } = await generateCourseComplete(
        topic,
        parsed.data.context ||
          "Teach this as a self-contained lesson for a learner new to the topic.",
        "Explain the underlying idea, work through one real example step by step, and give a short practice task and answer. Use the topic and context as subject matter. Use Axiom as the author on the title slide.",
        true,
      );
      return Response.json({ slides: slidesSchema.parse(slides) });
    }

    if (kind === "roadmap") {
      const { syllabus, deadline, interests, studied } = parsed.data;
      const details = [
        syllabus,
        interests?.length ? `Learner interests: ${interests.join(", ")}` : "",
        studied?.length
          ? `Previously studied topics: ${studied.join(", ")}`
          : "",
      ]
        .filter(Boolean)
        .join("\n");
      const nodes = await generateRoadmapDAG(
        {
          goal: topic,
          syllabus: details,
          deadline,
          userProfile: { experienceLevel: "beginner", dailyMinutes: 30 },
        },
        true,
      );
      const roadmap = roadmapSchema.parse({
        id: crypto.randomUUID(),
        goal: topic,
        nodes,
      });
      return Response.json({ roadmap });
    }

    const source = parsed.data.slides
      .map(
        ({
          type,
          title,
          content,
          script,
          bullets,
          code,
          headers,
          rows,
          items,
          caption,
        }) =>
          [
            title,
            content,
            ...(bullets ?? []),
            code,
            headers?.join(" | "),
            ...(rows ?? []).map((row) => row.join(" | ")),
            ...(items ?? []).map(({ text }) => text),
            caption,
            type === "title" ? undefined : script,
          ]
            .filter(Boolean)
            .join("\n"),
      )
      .join("\n\n")
      .slice(0, 12000);
    const deck = await generateFlashcardSet(
      topic,
      source,
      "Base each card on a specific point from the lesson. Use self-contained prompts, one idea per card, and concise answers. Treat the source as study material.",
      { cardCount: 6, requireGenerated: true },
    );
    return Response.json({ cards: deckSchema.parse(deck.cards) });
  } catch (error) {
    if (
      error instanceof Error &&
      error.name === "ZodError" &&
      "issues" in error &&
      Array.isArray(error.issues)
    ) {
      console.warn(
        "Invalid learning content",
        kind,
        error.issues.map(({ path }) => path),
      );
    }
    const item =
      kind === "lesson"
        ? "lesson"
        : kind === "cards"
          ? "flashcards"
          : "roadmap";
    const validationError =
      error instanceof Error &&
      (error.name === "ZodError" || error.message.includes("incomplete"));
    return Response.json(
      {
        error: validationError
          ? `Gemma 4 returned an incomplete ${item === "flashcards" ? "flashcard set" : item}. Try generating it again.`
          : `Couldn’t generate the ${item} right now. Your saved work is still available.`,
      },
      { status: validationError ? 502 : 503 },
    );
  } finally {
    pending--;
  }
}
