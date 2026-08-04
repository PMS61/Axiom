"use server";

import { readFile, writeFile } from "node:fs/promises";
import { synthesizeSpeech } from "@/lib/piper-service";
import { extractVisemeTimeline, type VisemeCue } from "@/lib/rhubarb-service";

export interface NarrationResult {
  audioUrl?: string;
  visemeTimeline?: VisemeCue[];
  error?: string;
}

async function readCachedVisemes(
  cachePath: string,
): Promise<VisemeCue[] | null> {
  try {
    const raw = await readFile(cachePath, "utf-8");
    return JSON.parse(raw) as VisemeCue[];
  } catch {
    return null;
  }
}

/**
 * Synthesizes narration audio for a slide's script (Piper) and extracts its
 * viseme timeline (Rhubarb) for CourseNarrator's avatar. Both stages are
 * cached to disk keyed off the audio file Piper already hashes by
 * (voice, text) — a repeat call for the same slide script reuses both the
 * WAV and the viseme JSON instead of re-running either engine.
 */
export async function narrateSlideAction(
  scriptText: string,
): Promise<NarrationResult> {
  const trimmed = scriptText.trim();
  if (!trimmed) return { error: "empty_script" };

  const synth = await synthesizeSpeech(trimmed);
  if (synth.error || !synth.audioUrl || !synth.audioPath) {
    return { error: synth.error || "synthesis_failed" };
  }

  const visemeCachePath = `${synth.audioPath}.visemes.json`;
  const cached = await readCachedVisemes(visemeCachePath);
  if (cached) {
    return { audioUrl: synth.audioUrl, visemeTimeline: cached };
  }

  const visemes = await extractVisemeTimeline(synth.audioPath, trimmed);
  const timeline = visemes.timeline ?? [];

  // Viseme extraction failing (e.g. Rhubarb unavailable) shouldn't block
  // narration audio — CourseNarrator already falls back to a closed-mouth
  // idle when visemeTimeline is empty, so this degrades gracefully.
  if (visemes.timeline) {
    await writeFile(visemeCachePath, JSON.stringify(timeline), "utf-8").catch(
      () => {},
    );
  }

  return { audioUrl: synth.audioUrl, visemeTimeline: timeline };
}
