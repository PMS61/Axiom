"use server";

import { generateCheatSheetMarkdown } from "@/lib/cheatsheet-agent";

export async function generateCheatSheetForTask(input: {
  title: string;
  description: string;
  additionalInstructions?: string;
}) {
  const title = input.title.trim();
  const description = input.description.trim();
  const additionalInstructions = (input.additionalInstructions ?? "").trim();

  if (!title || !description) {
    return {
      success: false,
      error: "Task title and description are required.",
    };
  }

  try {
    const markdown = await generateCheatSheetMarkdown(
      title,
      description,
      additionalInstructions,
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
        fileName: `${safeName || "course"}-cheatsheet.pdf`,
      },
    };
  } catch (error) {
    console.error("[CheatSheetTaskAction] failed:", error);
    return { success: false, error: "Failed to generate cheat sheet." };
  }
}
