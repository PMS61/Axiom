"use server";

import { generateNextChapterMarkdown } from "@/lib/story-agent";

export async function generateStoryForTask(input: {
  title: string;
  description: string;
  storyPreference?: string;
}) {
  const title = input.title.trim();
  const description = input.description.trim();
  const storyPreference = (input.storyPreference ?? "").trim();

  if (!title || !description) {
    return {
      success: false,
      error: "Task title and description are required.",
    };
  }

  try {
    const markdown = await generateNextChapterMarkdown(
      title,
      description,
      storyPreference,
    );

    const safeName = title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/(^-|-$)/g, "");

    return {
      success: true,
      data: {
        title,
        markdown,
        fileName: `${safeName || "story"}-next-chapter.pdf`,
      },
    };
  } catch (error) {
    console.error("[StoryTaskAction] failed:", error);
    return { success: false, error: "Failed to generate story chapter." };
  }
}
