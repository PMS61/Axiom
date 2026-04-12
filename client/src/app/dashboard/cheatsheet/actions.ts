"use server";

import { generateCheatSheetMarkdown } from "@/lib/cheatsheet-agent";

export async function generateCheatSheet(formData: FormData) {
  const title = String(formData.get("title") ?? "").trim();
  const description = String(formData.get("description") ?? "").trim();
  const additionalInstructions = String(
    formData.get("additionalInstructions") ?? "",
  ).trim();

  if (!title || !description) {
    return {
      success: false,
      error: "Course title and description are required.",
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
    console.error("[CheatSheetAction] failed:", error);
    return { success: false, error: "Failed to generate cheat sheet." };
  }
}
