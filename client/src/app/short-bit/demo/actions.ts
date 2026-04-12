"use server";

import { generateShortBitSet } from "@/lib/short-bit-agent";

export async function generateShortBits(formData: FormData) {
  const title = String(formData.get("title") ?? "").trim();
  const description = String(formData.get("description") ?? "").trim();
  const additionalInstructions = String(
    formData.get("additionalInstructions") ?? "",
  ).trim();

  if (!title || !description) {
    return {
      success: false,
      error: "Title and description are required.",
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
    console.error("[ShortBitDemo] generation failed:", error);
    return { success: false, error: "Failed to generate short bits." };
  }
}
