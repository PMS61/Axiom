/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Personalization Projections
   Pure functions slicing the canonical LearnerProfile into the
   exact shape each pillar needs. No I/O — mirrors the existing
   toDashboardProfilePayload()/toReportProfile() pattern in
   userProfileStorage.ts. See plans/04-personalization-unification-plan.md.
   ═══════════════════════════════════════════════════════════ */

import { normalizeQueryToTopic } from "./trend-engine/topicMapper";
import type { LearnerProfile, RoadmapUserProfile, TopicMastery } from "./types";

/** Normalises a free-text topic string onto the same canonical label space
 *  the trend engine uses, so mastery keys, roadmap topics, and trend topics
 *  converge instead of drifting into three incompatible vocabularies. */
export function normalizeTopicKey(topic: string): string {
  return normalizeQueryToTopic(topic);
}

// ── Roadmap Agent (What -> How bridge) ────────────────────

export function toRoadmapPersonalizationPayload(profile: LearnerProfile): {
  userProfile: RoadmapUserProfile;
  existingMastery: TopicMastery[];
} {
  return {
    userProfile: {
      experienceLevel: profile.experienceLevel,
      dailyMinutes: profile.dailyMinutes,
      learningStyle: profile.learningStyle,
    },
    existingMastery: Object.entries(profile.masteryByTopic).map(
      ([topic, score]) => ({
        topic,
        score,
      }),
    ),
  };
}

// ── Course / content generators (When -> How bridge) ──────
//
// Reserved for plans/02-course-generation-agentic-video.md's staged pipeline,
// which is what will actually thread a profile parameter into
// generateCourseSkeleton/generateCourseSlide. Not wired into course-agent.ts
// yet — building it now against a pipeline that's about to be restructured
// would just mean rebuilding it again once Plan 2 lands.

export interface CoursePersonalizationPayload {
  experienceLevel: LearnerProfile["experienceLevel"];
  learningStyle: LearnerProfile["learningStyle"];
  /** 0 if the learner has no recorded mastery for this topic yet. */
  masteryScore: number;
}

export function toCoursePersonalizationPayload(
  profile: LearnerProfile,
  topic: string,
): CoursePersonalizationPayload {
  return {
    experienceLevel: profile.experienceLevel,
    learningStyle: profile.learningStyle,
    masteryScore: profile.masteryByTopic[normalizeTopicKey(topic)] ?? 0,
  };
}

// ── Trend ranking (When -> What bridge) ───────────────────
//
// Reserved for plans/03-trends-accuracy-plan.md's item E (mastery-weighted
// ranking). Not wired into fetchTopTrends()'s scoring yet — sequence after
// Plan 3's core scoring/grounding fixes land, so personalization isn't
// layered on top of a score still being recalibrated.

export interface TrendPersonalizationPayload {
  interestDomains: string[];
  profileType: string;
  /** Topics below this mastery threshold, sorted lowest-first — candidates for a ranking boost. */
  strugglingTopics: string[];
}

const STRUGGLING_MASTERY_THRESHOLD = 0.5;

export function toTrendPersonalizationPayload(
  profile: LearnerProfile,
): TrendPersonalizationPayload {
  const strugglingTopics = Object.entries(profile.masteryByTopic)
    .filter(([, score]) => score < STRUGGLING_MASTERY_THRESHOLD)
    .sort(([, a], [, b]) => a - b)
    .map(([topic]) => topic);

  return {
    interestDomains: profile.interestDomains,
    profileType: profile.profileType,
    strugglingTopics,
  };
}
