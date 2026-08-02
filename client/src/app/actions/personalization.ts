"use server";

/**
 * ═══════════════════════════════════════════════════════════
 *  THE AXIOM — Unified Personalization Server Actions
 *  Single fetch/write path for the cross-pillar LearnerProfile.
 *  See plans/04-personalization-unification-plan.md.
 * ═══════════════════════════════════════════════════════════
 */

import { sql } from "@vercel/postgres";
import { verify } from "jsonwebtoken";
import { cookies } from "next/headers";
import type { LearnerProfile } from "@/lib/types";

// ── Auth ──────────────────────────────────────────────────

export async function getPersonalizationUserId(): Promise<number | null> {
  const cookieStore = await cookies();
  const token = cookieStore.get("token")?.value;
  if (!token || !process.env.JWT_SECRET) return null;
  try {
    const decoded = verify(token, process.env.JWT_SECRET) as { userId: number };
    return decoded.userId;
  } catch {
    return null;
  }
}

// ── Schema Bootstrap ──────────────────────────────────────
// Tables/columns this module reads or writes. Each is IF NOT EXISTS so it's
// safe to call redundantly alongside the other actions files that also
// touch `users` / `user_interests` (auth.ts, trends.ts) — same pattern
// already used throughout the codebase.

async function ensurePersonalizationSchema(): Promise<void> {
  await sql`ALTER TABLE users ADD COLUMN IF NOT EXISTS experience_level VARCHAR(20) DEFAULT 'beginner'`;
  await sql`ALTER TABLE users ADD COLUMN IF NOT EXISTS learning_style VARCHAR(20) DEFAULT 'balanced'`;

  await sql`
    CREATE TABLE IF NOT EXISTS user_interests (
      user_id INT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
      domains TEXT[] NOT NULL DEFAULT '{}',
      profile_type TEXT NOT NULL DEFAULT 'developer',
      updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )
  `;

  await sql`
    CREATE TABLE IF NOT EXISTS section_weights (
      user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
      morning DOUBLE PRECISION NOT NULL DEFAULT 0.40,
      afternoon DOUBLE PRECISION NOT NULL DEFAULT 0.35,
      evening DOUBLE PRECISION NOT NULL DEFAULT 0.25,
      updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )
  `;

  await sql`
    CREATE TABLE IF NOT EXISTS user_mastery (
      id SERIAL PRIMARY KEY,
      user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      topic TEXT NOT NULL,
      score DOUBLE PRECISION NOT NULL DEFAULT 0,
      source VARCHAR(20) NOT NULL DEFAULT 'assessment',
      updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
      UNIQUE (user_id, topic)
    )
  `;
}

// ── Mastery: read ─────────────────────────────────────────

export async function getMasteryMap(
  userId: number,
): Promise<Record<string, number>> {
  await ensurePersonalizationSchema();
  try {
    const result =
      await sql`SELECT topic, score FROM user_mastery WHERE user_id = ${userId}`;
    const map: Record<string, number> = {};
    for (const row of result.rows as { topic: string; score: number }[]) {
      map[row.topic] = row.score;
    }
    return map;
  } catch (err) {
    console.error("getMasteryMap failed:", err);
    return {};
  }
}

// ── Mastery: write ────────────────────────────────────────
//
// Merge policy (deliberate, not left implicit):
//   - source="assessment" is a direct measurement — always overwrites.
//   - source="roadmap" is a coarse completion signal (see completeNodeAction,
//     which only ever reports mastery=1 on task completion) — it must never
//     downgrade or overwrite a real assessment-sourced score. It only writes
//     when no score exists yet, or the existing score also came from "roadmap".

export async function upsertMastery(
  userId: number,
  topic: string,
  score: number,
  source: "assessment" | "roadmap",
): Promise<void> {
  if (!topic.trim()) return;
  await ensurePersonalizationSchema();
  const clamped = Math.max(0, Math.min(1, score));

  try {
    if (source === "assessment") {
      await sql`
        INSERT INTO user_mastery (user_id, topic, score, source, updated_at)
        VALUES (${userId}, ${topic}, ${clamped}, 'assessment', NOW())
        ON CONFLICT (user_id, topic) DO UPDATE SET
          score = EXCLUDED.score,
          source = 'assessment',
          updated_at = NOW()
      `;
    } else {
      await sql`
        INSERT INTO user_mastery (user_id, topic, score, source, updated_at)
        VALUES (${userId}, ${topic}, ${clamped}, 'roadmap', NOW())
        ON CONFLICT (user_id, topic) DO UPDATE SET
          score = CASE WHEN user_mastery.source = 'roadmap' THEN EXCLUDED.score ELSE user_mastery.score END,
          updated_at = CASE WHEN user_mastery.source = 'roadmap' THEN NOW() ELSE user_mastery.updated_at END
      `;
    }
  } catch (err) {
    console.error("upsertMastery failed:", err);
  }
}

// ── Learning preferences: write ───────────────────────────

export async function saveLearningPreferences(updates: {
  experience_level?: "beginner" | "intermediate" | "advanced";
  learning_style?: "visual" | "reading" | "practice" | "balanced";
}): Promise<{ error?: string }> {
  const userId = await getPersonalizationUserId();
  if (!userId) return { error: "Not authenticated" };

  await ensurePersonalizationSchema();

  try {
    if (updates.experience_level !== undefined) {
      await sql`UPDATE users SET experience_level = ${updates.experience_level} WHERE id = ${userId}`;
    }
    if (updates.learning_style !== undefined) {
      await sql`UPDATE users SET learning_style = ${updates.learning_style} WHERE id = ${userId}`;
    }
    return {};
  } catch (err) {
    console.error("saveLearningPreferences failed:", err);
    return { error: "Failed to save learning preferences." };
  }
}

// ── Unified profile: read ─────────────────────────────────

export async function getLearnerProfile(
  userId: number,
): Promise<LearnerProfile> {
  await ensurePersonalizationSchema();

  const defaults: LearnerProfile = {
    userId,
    experienceLevel: "beginner",
    learningStyle: "balanced",
    interestDomains: [],
    profileType: "developer",
    dailyMinutes: 120,
    masteryByTopic: {},
  };

  try {
    const userRow = await sql`
      SELECT experience_level, learning_style FROM users WHERE id = ${userId} LIMIT 1
    `;
    if (userRow.rows.length > 0) {
      const row = userRow.rows[0] as {
        experience_level: string | null;
        learning_style: string | null;
      };
      if (row.experience_level)
        defaults.experienceLevel =
          row.experience_level as LearnerProfile["experienceLevel"];
      if (row.learning_style)
        defaults.learningStyle =
          row.learning_style as LearnerProfile["learningStyle"];
    }

    const interestRow = await sql`
      SELECT domains, profile_type FROM user_interests WHERE user_id = ${userId} LIMIT 1
    `;
    if (interestRow.rows.length > 0) {
      const row = interestRow.rows[0] as {
        domains: string[];
        profile_type: string;
      };
      defaults.interestDomains = row.domains ?? [];
      defaults.profileType = row.profile_type ?? defaults.profileType;
    }

    const sectionRow = await sql`
      SELECT morning, afternoon, evening FROM section_weights WHERE user_id = ${userId} LIMIT 1
    `;
    if (sectionRow.rows.length > 0) {
      const { morning, afternoon, evening } = sectionRow.rows[0] as {
        morning: number;
        afternoon: number;
        evening: number;
      };
      // Same derivation trends.ts already uses: 50 axioms/day, ~4 min/axiom.
      defaults.dailyMinutes = Math.round(
        (morning + afternoon + evening) * 50 * 4,
      );
    }

    defaults.masteryByTopic = await getMasteryMap(userId);
  } catch (err) {
    console.error("getLearnerProfile failed, returning defaults:", err);
  }

  return defaults;
}

/** Convenience wrapper: resolves the current cookie-authenticated user's profile, or null if unauthenticated. */
export async function getCurrentLearnerProfile(): Promise<LearnerProfile | null> {
  const userId = await getPersonalizationUserId();
  if (!userId) return null;
  return getLearnerProfile(userId);
}
