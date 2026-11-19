"use server";

import { GoogleGenAI, ThinkingLevel } from "@google/genai";

// Rate limiter configuration
const MAX_CALLS_PER_MINUTE = 30;
const MINUTE_IN_MS = 60 * 1000;
const GEMMA_4_MODEL = "gemma-4-26b-a4b-it";
const GEMMA_4_ALTERNATE = "gemma-4-31b-it";

let genAI: GoogleGenAI | null = null;

/**
 * Sleep for the specified duration
 * @param {number} ms - Duration to sleep in milliseconds
 * @returns {Promise<void>}
 */
const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

function extractJsonFromResponse(text: string): string {
  const fenced = [
    ...text.matchAll(/```(?:json)?[ \t]*\r?\n([\s\S]*?)```/gi),
  ].map((match) => match[1].trim());
  const start = text.search(/[\[{]/);
  const end =
    start < 0 ? -1 : text.lastIndexOf(text[start] === "[" ? "]" : "}");
  const candidates = [text, ...fenced];
  if (start >= 0 && end > start) candidates.push(text.slice(start, end + 1));
  for (const candidate of candidates) {
    try {
      JSON.parse(candidate);
      return candidate;
    } catch {
      // Try the next complete output block.
    }
  }
  throw new SyntaxError("Gemma 4 did not return valid JSON.");
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
      status?: unknown;
      response?: { status?: unknown };
    };
    if (candidate.status === 429 || candidate.response?.status === 429)
      return true;
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
    if (candidate.status === 503 || candidate.status === 504) return true;
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

/** Generate with Gemma 4 through the Gemini API. Each request is independent. */
export async function getGeminiResponse(
  prompt: string,
  isJson: boolean = true,
  model: string = GEMMA_4_MODEL,
  fast = false,
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
      `Ignoring requested model "${model}". Axiom uses the Gemma 4 family.`,
    );
  }

  let retryCount = 0;
  const maxRetries = 3;

  let activeModel = GEMMA_4_MODEL;
  while (retryCount < maxRetries) {
    try {
      await rateLimitSafe();

      const response = await client.models.generateContent({
        model: activeModel,
        contents: prompt,
        config: {
          httpOptions: { timeout: 25000 },
          ...(fast
            ? {
                thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL },
                maxOutputTokens: 1800,
              }
            : {}),
        },
      });

      if (!response || !response.text) {
        throw new Error("No response text received from Gemini API");
      }

      const outputText = response.text.trim();
      return isJson ? extractJsonFromResponse(outputText) : outputText;
    } catch (error) {
      retryCount++;

      if (isRateLimitError(error) || isServiceUnavailableError(error)) {
        activeModel = retryCount === 1 ? GEMMA_4_ALTERNATE : GEMMA_4_MODEL;
        if (retryCount < maxRetries) {
          await sleep(1500 * retryCount);
          continue;
        }
      }

      console.error(
        `Error in Gemini API (attempt ${retryCount}/${maxRetries}):`,
        error,
      );

      if (retryCount >= maxRetries) {
        console.error("Gemma 4 retry limit reached");
        if (isJson) {
          // Return a basic error structure for JSON requests
          if (error instanceof SyntaxError) {
            return JSON.stringify({
              error: "invalid_output",
              message: error.message,
            });
          }
          return JSON.stringify({
            error: "service_unavailable",
            message: "Gemma 4 is currently unavailable. Try again in a moment.",
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
