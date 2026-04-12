"use server";

import { generateShortBitSet } from "@/lib/short-bit-agent";

export async function generateShortBitsForTask(input: {
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
    const set = await generateShortBitSet(
      title,
      description,
      additionalInstructions,
    );
    return { success: true, data: set };
  } catch (error) {
    console.error("[ShortBitTaskAction] generation failed:", error);
    return { success: false, error: "Failed to generate short bits." };
  }
}
