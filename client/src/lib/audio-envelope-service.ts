"use server";

import { readFile } from "node:fs/promises";
import type { Viseme, VisemeCue } from "./rhubarb-service";

// Fast, subprocess-free alternative to Rhubarb: reads the WAV's PCM samples
// directly and buckets loudness (RMS) into the same viseme categories
// CourseNarrator already knows how to render, so mouth-openness tracks the
// audio's actual amplitude envelope instead of running a recognizer.
// No speech/phoneme awareness at all — just "louder = mouth more open" —
// but for this avatar's rough per-category mouth-plane mapping that reads
// as real sync, and it's orders of magnitude faster than any recognizer
// (single-digit milliseconds vs. multi-second Rhubarb runs).

const WINDOW_SECONDS = 0.08;

interface WavPcm16Mono {
  sampleRate: number;
  dataOffset: number;
  sampleCount: number;
  buffer: Buffer;
}

function parseWavPcm16Mono(buffer: Buffer): WavPcm16Mono {
  if (
    buffer.toString("ascii", 0, 4) !== "RIFF" ||
    buffer.toString("ascii", 8, 12) !== "WAVE"
  ) {
    throw new Error("not a RIFF/WAVE file");
  }

  const sampleRate = buffer.readUInt32LE(24);

  let offset = 12;
  let dataOffset = -1;
  let dataSize = 0;
  while (offset + 8 <= buffer.length) {
    const chunkId = buffer.toString("ascii", offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    if (chunkId === "data") {
      dataOffset = offset + 8;
      dataSize = chunkSize;
      break;
    }
    offset += 8 + chunkSize + (chunkSize % 2);
  }
  if (dataOffset === -1) throw new Error("no data chunk found in WAV");

  return {
    sampleRate,
    dataOffset,
    sampleCount: Math.floor(dataSize / 2),
    buffer,
  };
}

function bucketAmplitudeToViseme(normalizedAmplitude: number): Viseme {
  if (normalizedAmplitude < 0.05) return "X";
  if (normalizedAmplitude < 0.15) return "A";
  if (normalizedAmplitude < 0.3) return "B";
  if (normalizedAmplitude < 0.45) return "H";
  if (normalizedAmplitude < 0.6) return "C";
  return "D";
}

export async function extractAmplitudeVisemeTimeline(
  audioPath: string,
): Promise<VisemeCue[]> {
  const raw = await readFile(audioPath);
  const wav = parseWavPcm16Mono(raw);

  const windowSamples = Math.max(
    1,
    Math.round(wav.sampleRate * WINDOW_SECONDS),
  );
  const rmsPerWindow: number[] = [];

  for (let start = 0; start < wav.sampleCount; start += windowSamples) {
    const end = Math.min(start + windowSamples, wav.sampleCount);
    let sumSquares = 0;
    for (let i = start; i < end; i++) {
      const sample = wav.buffer.readInt16LE(wav.dataOffset + i * 2) / 32768;
      sumSquares += sample * sample;
    }
    rmsPerWindow.push(Math.sqrt(sumSquares / (end - start)));
  }

  const maxRms = Math.max(...rmsPerWindow, 1e-6);

  return rmsPerWindow.map((rms, index) => ({
    start: index * WINDOW_SECONDS,
    end: (index + 1) * WINDOW_SECONDS,
    viseme: bucketAmplitudeToViseme(rms / maxRms),
  }));
}
