import {
  type AdaptiveProfile,
  AdaptiveProfileSchema,
  DEFAULT_ADAPTIVE_PROFILE,
  type TaskMemory,
} from "./contracts";

const PROFILE_KEY = "axiom_research_profile_v1";
const BASELINE_KEY = "axiom_research_profile_baseline_v1";

function write(profile: AdaptiveProfile): AdaptiveProfile {
  const next = {
    ...profile,
    version: profile.version + 1,
    updatedAt: new Date().toISOString(),
  };
  window.localStorage.setItem(PROFILE_KEY, JSON.stringify(next));
  return next;
}

/** M3 is the only module that writes adaptive profile state. */
export const M3Profile = {
  read(): AdaptiveProfile {
    if (typeof window === "undefined") return DEFAULT_ADAPTIVE_PROFILE;
    try {
      const saved = window.localStorage.getItem(PROFILE_KEY);
      if (saved) {
        const result = AdaptiveProfileSchema.safeParse(JSON.parse(saved));
        if (result.success) return result.data;
      }
    } catch {
      /* Recover as a cold-start profile. */
    }
    window.localStorage.setItem(
      PROFILE_KEY,
      JSON.stringify(DEFAULT_ADAPTIVE_PROFILE),
    );
    window.localStorage.setItem(
      BASELINE_KEY,
      JSON.stringify(DEFAULT_ADAPTIVE_PROFILE),
    );
    return DEFAULT_ADAPTIVE_PROFILE;
  },
  baseline(): AdaptiveProfile {
    if (typeof window === "undefined") return DEFAULT_ADAPTIVE_PROFILE;
    try {
      const saved = window.localStorage.getItem(BASELINE_KEY);
      if (saved) {
        const result = AdaptiveProfileSchema.safeParse(JSON.parse(saved));
        if (result.success) return result.data;
      }
    } catch {
      /* Set a baseline from the current profile. */
    }
    const profile = this.read();
    window.localStorage.setItem(BASELINE_KEY, JSON.stringify(profile));
    return profile;
  },
  updatePreferences(
    profile: AdaptiveProfile,
    updates: Partial<
      Pick<
        AdaptiveProfile,
        | "experienceLevel"
        | "learningStyle"
        | "peakFocusWindows"
        | "minimumFocusWindows"
        | "nonAvailabilityWindows"
      >
    >,
  ): AdaptiveProfile {
    return write({ ...profile, ...updates });
  },
  updateTopicSkill(
    profile: AdaptiveProfile,
    topic: string,
    score: number,
  ): AdaptiveProfile {
    const key = topic.trim().slice(0, 100);
    if (!key) return profile;
    return write({
      ...profile,
      skillByTopic: {
        ...profile.skillByTopic,
        [key]: Math.max(0, Math.min(1, score)),
      },
    });
  },
  recordTaskOutcome(
    profile: AdaptiveProfile,
    task: Omit<
      TaskMemory,
      "actualDurationMinutes" | "reportedDifficulty" | "recordedAt"
    >,
    actualDurationMinutes: number,
    reportedDifficulty: number,
  ): AdaptiveProfile {
    const observation: TaskMemory = {
      ...task,
      actualDurationMinutes: Math.round(actualDurationMinutes),
      reportedDifficulty: Math.round(reportedDifficulty),
      recordedAt: new Date().toISOString(),
    };
    const taskHistory = [
      ...profile.taskHistory.filter((item) => item.taskId !== task.taskId),
      observation,
    ].slice(-200);
    return write({ ...profile, taskHistory });
  },
};
