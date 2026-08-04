"use server";

import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const RHUBARB_BINARY = process.env.RHUBARB_BINARY_PATH || "rhubarb";

export type Viseme = "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "X";

export interface VisemeCue {
  start: number;
  end: number;
  viseme: Viseme;
}

export interface VisemeExtractionResult {
  timeline?: VisemeCue[];
  error?: string;
}

let availabilityChecked = false;
let isAvailable = false;

async function checkAvailability(): Promise<boolean> {
  if (availabilityChecked) return isAvailable;
  availabilityChecked = true;

  isAvailable = await new Promise((resolve) => {
    const probe = spawn(RHUBARB_BINARY, ["--version"]);
    probe.on("error", () => resolve(false));
    probe.on("close", () => resolve(true));
  });

  if (!isAvailable) {
    console.warn(
      `[RhubarbService] rhubarb binary not found at "${RHUBARB_BINARY}"; lip sync is unavailable.`,
    );
  }
  return isAvailable;
}

function runRhubarb(args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(RHUBARB_BINARY, args);
    let stderr = "";
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`rhubarb exited with code ${code}: ${stderr}`));
    });
  });
}

/**
 * Extracts a viseme (mouth-shape) timeline from narration audio + its exact
 * spoken text via a self-hosted Rhubarb Lip Sync process. Decoupled from the
 * TTS engine — Rhubarb reads the audio itself, doesn't need Piper to expose
 * phoneme timestamps.
 */
export async function extractVisemeTimeline(
  audioPath: string,
  dialogText: string,
): Promise<VisemeExtractionResult> {
  const available = await checkAvailability();
  if (!available) return { error: "rhubarb_unavailable" };

  const tmpDir = await mkdtemp(path.join(os.tmpdir(), "rhubarb-"));
  const dialogPath = path.join(tmpDir, "dialog.txt");
  const outputPath = path.join(tmpDir, "output.json");

  try {
    await writeFile(dialogPath, dialogText, "utf-8");
    await runRhubarb([
      "-f",
      "json",
      "-o",
      outputPath,
      audioPath,
      "--dialogFile",
      dialogPath,
    ]);

    const raw = await readFile(outputPath, "utf-8");
    const parsed = JSON.parse(raw) as {
      mouthCues?: Array<{ start: number; end: number; value: string }>;
    };

    const timeline: VisemeCue[] = (parsed.mouthCues ?? []).map((cue) => ({
      start: cue.start,
      end: cue.end,
      viseme: cue.value as Viseme,
    }));

    return { timeline };
  } catch (error) {
    console.error("[RhubarbService] viseme extraction failed:", error);
    return { error: "extraction_failed" };
  } finally {
    await rm(tmpDir, { recursive: true, force: true }).catch(() => {});
  }
}
