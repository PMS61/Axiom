"use server";

import { generateFlashcardSet } from "@/lib/flashcard-agent";

export async function generateFlashcards(formData: FormData) {
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
    const set = await generateFlashcardSet(
      title,
      description,
      additionalInstructions,
    );
    return { success: true, data: set };
  } catch (error) {
    console.error("[FlashcardDemo] generation failed:", error);
    return { success: false, error: "Failed to generate flashcards." };
  }
}
