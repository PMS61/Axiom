import { getGeminiResponse } from "./gemini";

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

    throw new Error("Story generation returned empty content.");
  } catch (error) {
    console.error("[StoryAgent] generation failed:", error);
    throw error;
  }
}
