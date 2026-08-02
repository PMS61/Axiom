"use server";

import { GoogleGenAI } from "@google/genai";

// Rate limiter configuration
const MAX_CALLS_PER_MINUTE = 30;
const MINUTE_IN_MS = 60 * 1000;
const GEMMA_4_MODEL = "gemma-4-31b-it";

let genAI: GoogleGenAI | null = null;

/**
 * Sleep for the specified duration
 * @param {number} ms - Duration to sleep in milliseconds
 * @returns {Promise<void>}
 */
const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

function extractJsonFromResponse(text: string): string {
  const startIndex = text.indexOf("```json");
  const endIndex = text.lastIndexOf("```");

  if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
    return text.substring(startIndex + 7, endIndex).trim(); // 7 = length of "```json"
  }

  return text.trim(); // fallback: return full text
}

function getGeminiClient(): GoogleGenAI | null {
  if (genAI) return genAI;

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.warn(
      "GEMINI_API_KEY is not configured; AI generation is unavailable.",
    );
    return null;
  }

  genAI = new GoogleGenAI({ apiKey });
  return genAI;
}

// Improved rate limiter: avoid hitting the limit by spacing calls evenly
const CALL_INTERVAL_MS = Math.ceil(MINUTE_IN_MS / MAX_CALLS_PER_MINUTE);
let nextAvailableCallTime = 0;

async function rateLimitSafe(): Promise<void> {
  const now = Date.now();
  if (now < nextAvailableCallTime) {
    const delay = nextAvailableCallTime - now;
    nextAvailableCallTime += CALL_INTERVAL_MS;
    await sleep(delay);
  } else {
    nextAvailableCallTime = now + CALL_INTERVAL_MS;
  }
}

function isRateLimitError(error: unknown): boolean {
  if (!error) return false;
  if (typeof error === "object" && error !== null) {
    const candidate = error as {
      message?: unknown;
      response?: { status?: unknown };
    };
    if (candidate.response?.status === 429) return true;
    if (
      typeof candidate.message === "string" &&
      candidate.message.toLowerCase().includes("rate limit")
    )
      return true;
  }
  return false;
}

function isServiceUnavailableError(error: unknown): boolean {
  if (!error) return false;
  if (typeof error === "object" && error !== null) {
    const candidate = error as { message?: unknown; status?: unknown };
    if (candidate.status === 503) return true;
    if (
      typeof candidate.message === "string" &&
      candidate.message.toLowerCase().includes("overloaded")
    )
      return true;
    if (
      typeof candidate.message === "string" &&
      candidate.message.toLowerCase().includes("unavailable")
    )
      return true;
  }
  return false;
}

/**
 * Function that takes a prompt as input and returns the output from Gemini-1.5-flash model
 * Each call creates a new chat instance, so there's no persistent memory between calls
 * Rate limited to 14 calls per minute, will wait if rate limit is reached
 */
export async function getGeminiResponse(
  prompt: string,
  isJson: boolean = true,
  model: string = GEMMA_4_MODEL,
): Promise<string> {
  const client = getGeminiClient();
  if (!client) {
    return isJson
      ? JSON.stringify({
          error: "ai_unavailable",
          message:
            "Gemma 4 is unavailable because GEMINI_API_KEY is not configured.",
        })
      : "AI generation is unavailable because GEMINI_API_KEY is not configured.";
  }

  if (model !== GEMMA_4_MODEL) {
    console.warn(
      `Ignoring requested model "${model}". Axiom is configured to use ${GEMMA_4_MODEL} only.`,
    );
  }

  let retryCount = 0;
  const maxRetries = 3;

  while (retryCount < maxRetries) {
    try {
      await rateLimitSafe();

      const response = await client.models.generateContent({
        model: GEMMA_4_MODEL,
        contents: prompt,
      });

      if (!response || !response.text) {
        throw new Error("No response text received from Gemini API");
      }

      let outputText = response.text.trim();

      // console.log("Raw output text:", outputText);

      if (isJson) {
        try {
          JSON.parse(outputText);
        } catch {
          outputText = extractJsonFromResponse(outputText);
          // Try parsing again after extraction
          try {
            JSON.parse(outputText);
          } catch {
            throw new Error("Could not extract valid JSON from response");
          }
        }
      }

      // console.log("Prompt:", prompt.substring(0, 200) + "...");
      // console.log("Response:", outputText.substring(0, 200) + "...");

      return outputText;
    } catch (error) {
      retryCount++;

      if (isRateLimitError(error)) {
        console.warn("Rate limit hit, retrying in 10s...");
        await sleep(10000);
        retryCount--; // Don't count rate limit as a retry
        continue;
      }

      if (isServiceUnavailableError(error)) {
        console.warn(
          `Service unavailable (attempt ${retryCount}/${maxRetries}), retrying in ${5 * retryCount}s...`,
        );
        if (retryCount < maxRetries) {
          await sleep(5000 * retryCount);
          continue;
        }
      }

      console.error(
        `Error in Gemini API (attempt ${retryCount}/${maxRetries}):`,
        error,
      );

      if (retryCount >= maxRetries) {
        console.error("Max retries exceeded, falling back to default content");
        if (isJson) {
          // Return a basic error structure for JSON requests
          return JSON.stringify({
            error: "service_unavailable",
            message:
              "Gemma 4 is currently unavailable. Using fallback content.",
          });
        } else {
          return "I apologize, but the AI service is currently unavailable. Please try again later.";
        }
      }

      // Wait before retry
      await sleep(2000 * retryCount);
    }
  }

  // This should never be reached, but just in case
  console.error("Unexpected end of retry loop");
  return isJson
    ? JSON.stringify({ error: "max_retries_exceeded" })
    : "Sorry, I encountered an error processing your request.";
}
