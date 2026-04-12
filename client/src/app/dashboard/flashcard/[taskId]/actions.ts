"use server";

import { generateFlashcardSet } from "@/lib/flashcard-agent";

export async function generateFlashcardsForTask(input: {
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
    const set = await generateFlashcardSet(
      title,
      description,
      additionalInstructions,
    );
    return { success: true, data: set };
  } catch (error) {
    console.error("[FlashcardTaskAction] generation failed:", error);
    return { success: false, error: "Failed to generate flashcards." };
  }
}
