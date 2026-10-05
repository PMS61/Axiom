"use server";

import { GoogleGenAI } from "@google/genai";

const MAX_CALLS_PER_MINUTE = 30;
const CALL_INTERVAL_MS = Math.ceil(60_000 / MAX_CALLS_PER_MINUTE);
const MODEL = "gemma-4-31b-it";
let client: GoogleGenAI | null = null;
let nextCallAt = 0;

function getClient(): GoogleGenAI | null {
  if (client) return client;
  const key = process.env.GEMINI_API_KEY;
  if (!key) return null;
  client = new GoogleGenAI({ apiKey: key });
  return client;
}

function extractJson(text: string): string {
  const start = text.indexOf("```json");
  const end = text.lastIndexOf("```");
  return start >= 0 && end > start
    ? text.slice(start + 7, end).trim()
    : text.trim();
}

export async function getGeminiResponse(
  prompt: string,
  isJson = true,
): Promise<string> {
  const ai = getClient();
  if (!ai)
    return isJson
      ? JSON.stringify({ error: "ai_unavailable" })
      : "AI is unavailable.";

  let lastError: unknown;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      const wait = Math.max(0, nextCallAt - Date.now());
      nextCallAt = Math.max(Date.now(), nextCallAt) + CALL_INTERVAL_MS;
      if (wait) await new Promise((resolve) => setTimeout(resolve, wait));
      const response = await ai.models.generateContent({
        model: MODEL,
        contents: prompt,
      });
      if (!response.text)
        throw new Error("No response text received from Gemini API.");
      let output = response.text.trim();
      if (isJson) {
        try {
          JSON.parse(output);
        } catch {
          output = extractJson(output);
          JSON.parse(output);
        }
      }
      return output;
    } catch (error) {
      lastError = error;
      const status =
        typeof error === "object" && error !== null
          ? Number((error as { status?: unknown }).status)
          : 0;
      const retryable =
        status === 429 ||
        status === 500 ||
        status === 503 ||
        String(error).toLowerCase().includes("overloaded");
      if (!retryable || attempt === 3) break;
      await new Promise((resolve) => setTimeout(resolve, attempt * 1_000));
    }
  }
  console.error("Gemini task estimate failed:", lastError);
  return isJson
    ? JSON.stringify({ error: "model_unavailable" })
    : "AI is unavailable.";
}
