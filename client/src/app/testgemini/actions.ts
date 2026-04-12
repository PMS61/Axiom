"use server";

import { getGeminiResponse } from "@/lib/gemini";

export async function sendChatMessage(prompt: string) {
  try {
    // Calling getGeminiResponse with isJson = false for standard chat response
    const response = await getGeminiResponse(prompt, false);
    return { success: true, text: response };
  } catch (error: any) {
    console.error("Gemini Chat Error:", error);
    return { success: false, error: error.message || "Failed to get response from Gemini" };
  }
}
