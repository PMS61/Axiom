"use server";

import { generateNextChapterMarkdown } from "@/lib/story-agent";

export async function generateStoryChapter(formData: FormData) {
  const title = String(formData.get("title") ?? "").trim();
  const description = String(formData.get("description") ?? "").trim();
  const storyPreference = String(formData.get("storyPreference") ?? "").trim();

  if (!title || !description) {
    return {
      success: false,
      error: "Story title and description are required.",
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
    console.error("[StoryAction] failed:", error);
    return { success: false, error: "Failed to generate next chapter." };
  }
}
