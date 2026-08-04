"use server";

import { spawn } from "node:child_process";
import crypto from "node:crypto";
import { mkdir, stat } from "node:fs/promises";
import path from "node:path";

const PIPER_BINARY = process.env.PIPER_BINARY_PATH || "piper";
const PIPER_MODEL = process.env.PIPER_MODEL_PATH;
const PIPER_ESPEAK_DATA = process.env.PIPER_ESPEAK_DATA_PATH;
const AUDIO_OUTPUT_DIR = path.join(
  process.cwd(),
  "public",
  "audio",
  "generated",
);
const AUDIO_PUBLIC_PREFIX = "/audio/generated";

let availabilityChecked = false;
let isAvailable = false;

async function checkAvailability(): Promise<boolean> {
  if (availabilityChecked) return isAvailable;
  availabilityChecked = true;

  if (!PIPER_MODEL) {
    console.warn(
      "[PiperService] PIPER_MODEL_PATH is not configured; TTS is unavailable.",
    );
    isAvailable = false;
    return false;
  }

  isAvailable = await new Promise((resolve) => {
    const probe = spawn(PIPER_BINARY, ["--help"]);
    probe.on("error", () => resolve(false));
    probe.on("close", () => resolve(true));
  });

  if (!isAvailable) {
    console.warn(
      `[PiperService] piper binary not found at "${PIPER_BINARY}"; TTS is unavailable.`,
    );
  }
  return isAvailable;
}

function runPiper(args: string[], stdin: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(PIPER_BINARY, args);
    let stderr = "";
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`piper exited with code ${code}: ${stderr}`));
    });
    child.stdin.write(stdin);
    child.stdin.end();
  });
}

function cacheKey(text: string, voice?: string): string {
  return crypto
    .createHash("sha256")
    .update(`${voice ?? "default"}::${text}`)
    .digest("hex")
    .slice(0, 24);
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await stat(filePath);
    return true;
  } catch {
    return false;
  }
}

export interface SynthesizeResult {
  audioUrl?: string;
  audioPath?: string;
  error?: string;
}

/**
 * Synthesizes speech via a self-hosted Piper process. Results are cached to
 * disk by a hash of (voice, text) — repeat calls for the same slide/card
 * text reuse the existing file instead of re-invoking Piper.
 */
export async function synthesizeSpeech(
  text: string,
  voice?: string,
): Promise<SynthesizeResult> {
  const trimmed = text.trim();
  if (!trimmed) return { error: "empty_text" };

  const available = await checkAvailability();
  if (!available) return { error: "piper_unavailable" };

  await mkdir(AUDIO_OUTPUT_DIR, { recursive: true });

  const fileName = `${cacheKey(trimmed, voice)}.wav`;
  const outputPath = path.join(AUDIO_OUTPUT_DIR, fileName);
  const audioUrl = `${AUDIO_PUBLIC_PREFIX}/${fileName}`;

  if (await fileExists(outputPath)) {
    return { audioUrl, audioPath: outputPath };
  }

  const args = ["--model", PIPER_MODEL as string, "--output_file", outputPath];
  if (voice) args.push("--speaker", voice);
  if (PIPER_ESPEAK_DATA) args.push("--espeak_data", PIPER_ESPEAK_DATA);

  try {
    await runPiper(args, trimmed);
    return { audioUrl, audioPath: outputPath };
  } catch (error) {
    console.error("[PiperService] synthesis failed:", error);
    return { error: "synthesis_failed" };
  }
}
