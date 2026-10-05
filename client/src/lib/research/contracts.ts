import { z } from "zod";

export const AdaptiveProfileSchema = z.object({
  schemaVersion: z.literal(1),
  version: z.number().int().nonnegative(),
  experienceLevel: z.enum(["unknown", "beginner", "intermediate", "advanced"]),
  learningStyle: z.enum([
    "unknown",
    "visual",
    "reading",
    "practice",
    "balanced",
  ]),
  peakFocusWindows: z.array(z.string().max(60)).max(20),
  minimumFocusWindows: z.array(z.string().max(60)).max(20),
  nonAvailabilityWindows: z.array(z.string().max(100)).max(30),
  skillByTopic: z.record(z.string(), z.number().min(0).max(1)),
  taskHistory: z
    .array(
      z.object({
        taskId: z.string().min(1).max(80),
        name: z.string().min(1).max(160),
        subject: z.string().min(1).max(100),
        type: z.enum([
          "learning",
          "problem_solving",
          "writing",
          "revision",
          "reading",
          "administrative",
          "recreational",
        ]),
        estimatedDifficulty: z.number().int().min(1).max(10),
        estimatedDurationMinutes: z.number().int().min(10).max(360),
        cognitiveLoadProxy: z.number().min(0).max(1000),
        actualDurationMinutes: z.number().int().min(1).max(1440),
        reportedDifficulty: z.number().int().min(1).max(10),
        recordedAt: z.string().datetime(),
      }),
    )
    .max(200),
  updatedAt: z.string().datetime().nullable(),
});

export type AdaptiveProfile = z.infer<typeof AdaptiveProfileSchema>;
export type TaskMemory = AdaptiveProfile["taskHistory"][number];
export const DEFAULT_ADAPTIVE_PROFILE: AdaptiveProfile = {
  schemaVersion: 1,
  version: 0,
  experienceLevel: "unknown",
  learningStyle: "unknown",
  peakFocusWindows: [],
  minimumFocusWindows: [],
  nonAvailabilityWindows: [],
  skillByTopic: {},
  taskHistory: [],
  updatedAt: null,
};
