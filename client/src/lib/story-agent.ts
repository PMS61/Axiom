import { getGeminiResponse } from "./gemini";

export type StoryChapterPayload = {
  chapterTitle: string;
  openingHook: string;
  chapterBody: string[];
  characterBeats: string[];
  turningPoints: string[];
  endingCliffhanger: string;
  nextChapterTease: string;
};

function fallbackPayload(
  title: string,
  description: string,
): StoryChapterPayload {
  return {
    chapterTitle: `Lesson Chapter: ${title}`,
    openingHook:
      "Tonight's quest begins when the mentor reveals a challenge that can only be solved by mastering one core idea.",
    chapterBody: [
      `In the academy of ${title}, the protagonist encounters a real problem that exposes what they still do not understand.`,
      description ||
        "The mentor breaks the concept into a simple mental model and demonstrates how to apply it.",
      "Through a short challenge, the protagonist tests the idea, makes a mistake, and then corrects it step by step.",
      "By the end of the chapter, the lesson is clear, practical, and ready to use in the next challenge.",
    ],
    characterBeats: [
      "Protagonist asks beginner-friendly questions the reader might also ask.",
      "Mentor explains the concept with patience, analogy, and one concrete example.",
    ],
    turningPoints: [
      "A misconception is exposed and corrected.",
      "The concept clicks when applied to a practical mini-problem.",
    ],
    endingCliffhanger:
      "A harder challenge appears, requiring the next concept to move forward.",
    nextChapterTease:
      "Next chapter introduces a new concept that builds directly on today's lesson.",
  };
}

function toMarkdown(
  payload: StoryChapterPayload,
  title: string,
  description: string,
  storyPreference: string,
): string {
  const lines: string[] = [];

  lines.push(`# ${payload.chapterTitle}`);
  lines.push("");
  lines.push(payload.openingHook);
  lines.push("");

  lines.push(
    `The lesson focus is ${title}. ${description} ${storyPreference ? `Style preference: ${storyPreference}.` : ""}`.trim(),
  );
  lines.push("");

  payload.chapterBody.forEach((paragraph) => {
    lines.push(paragraph);
    lines.push("");
  });

  if (payload.characterBeats.length > 0) {
    lines.push(payload.characterBeats.join(" "));
    lines.push("");
  }

  if (payload.turningPoints.length > 0) {
    lines.push(payload.turningPoints.join(" "));
    lines.push("");
  }

  lines.push(payload.endingCliffhanger);
  lines.push("");

  lines.push(payload.nextChapterTease);

  return lines.join("\n").trim();
}

export async function generateNextChapterMarkdown(
  title: string,
  description: string,
  storyPreference: string,
): Promise<string> {
  const prompt = `
You are an expert educator and web-novel writer.
Generate ONE chapter that teaches the user in a fun story format.

Inputs:
- Topic title: ${title}
- Topic description: ${description}
- User story preference: ${storyPreference || "None"}

Hard rules:
- Use ALL input fields meaningfully.
- Teach the topic clearly through the narrative, not as dry lecture notes.
- Make it beginner-friendly and engaging, like a web novel episode.
- Include at least one concrete example or mini problem inside the story.
- Keep narrative coherent, vivid, and emotionally engaging.
- Return a FULL ONE-SHOT MARKDOWN chapter (no JSON, no code fences).
- Keep it as a clean flow, not separated into named sections.
- Start with a single markdown title (# ...), then continue with natural paragraphs only.
`;

  try {
    const markdown = await getGeminiResponse(prompt, false);
    const trimmed = markdown.trim();
    if (trimmed.length > 0) {
      return trimmed;
    }

    return toMarkdown(
      fallbackPayload(title, description),
      title,
      description,
      storyPreference,
    );
  } catch (error) {
    console.error("[StoryAgent] generation failed:", error);
    return toMarkdown(
      fallbackPayload(title, description),
      title,
      description,
      storyPreference,
    );
  }
}
