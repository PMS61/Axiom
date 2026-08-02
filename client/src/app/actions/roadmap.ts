"use server";

/**
 * ═══════════════════════════════════════════════════════════
 *  THE AXIOM — Roadmap Agent Server Actions
 * ═══════════════════════════════════════════════════════════
 */

import { sql } from "@vercel/postgres";
import { verify } from "jsonwebtoken";
import { cookies } from "next/headers";
import {
  normalizeTopicKey,
  toRoadmapPersonalizationPayload,
} from "@/lib/personalization";
import { generateRoadmapDAG, recomputeNodeStates } from "@/lib/roadmap-agent";
import type { AdaptiveRoadmap, RoadmapDAGNode, RoadmapGoal } from "@/lib/types";
import { getLearnerProfile, upsertMastery } from "./personalization";
import { updateTaskStateAndSlot } from "./tasks";

// ── Auth Helper ───────────────────────────────────────────

async function getUserId(): Promise<number | null> {
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

async function createRoadmapsTable(): Promise<void> {
  await sql`
    CREATE TABLE IF NOT EXISTS roadmaps (
      id VARCHAR(255) PRIMARY KEY,
      user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      goal JSONB NOT NULL,
      nodes JSONB NOT NULL DEFAULT '[]'::jsonb,
      generated_at TIMESTAMPTZ NOT NULL,
      last_updated_at TIMESTAMPTZ NOT NULL
    )
  `;
  try {
    // Drop any unique constraints on the roadmaps table to allow multiple roadmaps
    await sql`
      DO $$
      DECLARE
          r RECORD;
      BEGIN
          FOR r IN (
            SELECT constraint_name 
            FROM information_schema.table_constraints 
            WHERE table_name = 'roadmaps' AND constraint_type = 'UNIQUE'
          )
          LOOP
              EXECUTE 'ALTER TABLE roadmaps DROP CONSTRAINT ' || quote_ident(r.constraint_name);
          END LOOP;
      END
      $$;
    `;
  } catch (e) {
    console.error("Failed to drop unique constraints:", e);
  }
}

// ── Actions ───────────────────────────────────────────────

export async function saveRoadmapAction(
  roadmap: AdaptiveRoadmap,
): Promise<{ success?: boolean; error?: string }> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    await createRoadmapsTable();
    await sql`
      INSERT INTO roadmaps (id, user_id, goal, nodes, generated_at, last_updated_at)
      VALUES (
        ${roadmap.id}, ${userId}, ${JSON.stringify(roadmap.goal)}::jsonb,
        ${JSON.stringify(roadmap.nodes)}::jsonb, ${roadmap.generatedAt}, ${roadmap.lastUpdatedAt}
      )
      ON CONFLICT (id) DO UPDATE SET
        goal = EXCLUDED.goal,
        nodes = EXCLUDED.nodes,
        generated_at = EXCLUDED.generated_at,
        last_updated_at = EXCLUDED.last_updated_at
    `;
    return { success: true };
  } catch (err) {
    console.error("[saveRoadmapAction] Error:", err);
    return { error: "Failed to persist roadmap." };
  }
}

export async function getRoadmapAction(roadmapId?: string): Promise<{
  roadmap?: AdaptiveRoadmap | null;
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    await createRoadmapsTable();
    const result = roadmapId
      ? await sql`SELECT id, goal, nodes, generated_at, last_updated_at FROM roadmaps WHERE user_id = ${userId} AND id = ${roadmapId} LIMIT 1`
      : await sql`SELECT id, goal, nodes, generated_at, last_updated_at FROM roadmaps WHERE user_id = ${userId} ORDER BY last_updated_at DESC LIMIT 1`;

    if (result.rows.length === 0) return { roadmap: null };
    const row = result.rows[0];
    return {
      roadmap: {
        id: row.id,
        goal: row.goal,
        nodes: row.nodes,
        generatedAt: new Date(row.generated_at).toISOString(),
        lastUpdatedAt: new Date(row.last_updated_at).toISOString(),
      },
    };
  } catch (err) {
    console.error("[getRoadmapAction] Error:", err);
    return { error: "Failed to fetch roadmap." };
  }
}

export async function listRoadmapsAction(): Promise<{
  roadmaps?: {
    id: string;
    goal: RoadmapGoal;
    generatedAt: string;
    lastUpdatedAt: string;
  }[];
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    await createRoadmapsTable();
    const result = await sql`
      SELECT id, goal, generated_at, last_updated_at FROM roadmaps WHERE user_id = ${userId} ORDER BY last_updated_at DESC
    `;
    return {
      roadmaps: result.rows.map((row) => ({
        id: row.id,
        goal: row.goal,
        generatedAt: new Date(row.generated_at).toISOString(),
        lastUpdatedAt: new Date(row.last_updated_at).toISOString(),
      })),
    };
  } catch (err) {
    console.error("[listRoadmapsAction] Error:", err);
    return { error: "Failed to fetch roadmaps." };
  }
}

export async function getLatestRoadmapAction(): Promise<{
  roadmap?: AdaptiveRoadmap;
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    await createRoadmapsTable();
    const result = await sql`
      SELECT id, goal, nodes, generated_at, last_updated_at 
      FROM roadmaps 
      WHERE user_id = ${userId} 
      ORDER BY last_updated_at DESC 
      LIMIT 1
    `;
    if (result.rows.length === 0) return {};
    const row = result.rows[0];
    return {
      roadmap: {
        id: row.id,
        goal: row.goal,
        nodes: row.nodes,
        generatedAt: new Date(row.generated_at).toISOString(),
        lastUpdatedAt: new Date(row.last_updated_at).toISOString(),
      },
    };
  } catch (err) {
    console.error("[getLatestRoadmapAction] Error:", err);
    return { error: "Failed to fetch latest roadmap." };
  }
}

export async function clearRoadmapAction(): Promise<{
  success?: boolean;
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    await createRoadmapsTable();
    await sql`DELETE FROM roadmaps WHERE user_id = ${userId}`;
    return { success: true };
  } catch (err) {
    console.error("[clearRoadmapAction] Error:", err);
    return { error: "Failed to clear roadmaps." };
  }
}

export async function generateRoadmapAction(goal: RoadmapGoal): Promise<{
  roadmap?: AdaptiveRoadmap;
  error?: string;
}> {
  try {
    // Fill in personalization the caller didn't already specify explicitly —
    // an explicit userProfile/existingMastery on the goal always wins.
    const userId = await getUserId();
    let personalizedGoal = goal;
    if (userId && (!goal.userProfile || !goal.existingMastery)) {
      const profile = await getLearnerProfile(userId);
      const { userProfile, existingMastery } =
        toRoadmapPersonalizationPayload(profile);
      personalizedGoal = {
        ...goal,
        userProfile: goal.userProfile ?? userProfile,
        existingMastery: goal.existingMastery ?? existingMastery,
      };
    }

    const nodes = await generateRoadmapDAG(personalizedGoal);
    const roadmap: AdaptiveRoadmap = {
      id: `roadmap_${Date.now()}`,
      goal: personalizedGoal,
      nodes,
      generatedAt: new Date().toISOString(),
      lastUpdatedAt: new Date().toISOString(),
    };

    // Auto-persist when generated
    const saveRes = await saveRoadmapAction(roadmap);
    if (saveRes.error) {
      console.warn("Roadmap generated but failed to persist:", saveRes.error);
    }

    return { roadmap };
  } catch (err) {
    console.error("[generateRoadmapAction] Error:", err);
    return { error: err instanceof Error ? err.message : "Unknown error" };
  }
}

export async function generateContentAction(
  title: string,
  topic: string,
  contentType: "ppt" | "one-shot" | "shortbits" | "storytelling" = "ppt",
): Promise<{ data?: any; error?: string }> {
  try {
    if (contentType === "shortbits") {
      const { generateShortBitSet } = await import("@/lib/short-bit-agent");
      return { data: await generateShortBitSet(title, topic, "") };
    }

    if (contentType === "storytelling") {
      const { generateNextChapterMarkdown } = await import("@/lib/story-agent");
      return { data: await generateNextChapterMarkdown(title, topic, "") };
    }

    if (contentType === "one-shot") {
      const { generateCheatSheetMarkdown } = await import(
        "@/lib/cheatsheet-agent"
      );
      return { data: await generateCheatSheetMarkdown(title, topic, "") };
    }

    const { generateCourseComplete } = await import("@/lib/course-agent");
    return { data: await generateCourseComplete(title, topic) };
  } catch (err: any) {
    console.error("[generateContentAction] Error:", err);
    return { error: err.message };
  }
}

export async function completeNodeAction(taskId: string): Promise<{
  success?: boolean;
  error?: string;
  roadmap?: AdaptiveRoadmap;
  completedNodeId?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    await createRoadmapsTable();

    const taskResult = await sql`
      SELECT id, name
      FROM tasks
      WHERE id = ${taskId} AND user_id = ${userId}
      LIMIT 1
    `;

    if (taskResult.rows.length === 0) {
      return { error: "Task not found" };
    }

    const taskName = (taskResult.rows[0]?.name as string | undefined) ?? "";

    // 1. Mark the task as completed in the DB
    await updateTaskStateAndSlot(taskId, "completed");

    // 2. Prefer roadmap that explicitly links this taskId
    const linkedRoadmapResult = await sql`
      SELECT id, goal, nodes, generated_at, last_updated_at 
      FROM roadmaps 
      WHERE user_id = ${userId} 
      AND nodes @> ${JSON.stringify([{ taskId }])}::jsonb
      ORDER BY last_updated_at DESC
      LIMIT 1
    `;

    let roadmapRow = linkedRoadmapResult.rows[0] as
      | {
          id: string;
          goal: any;
          nodes: RoadmapDAGNode[];
          generated_at: string;
          last_updated_at: string;
        }
      | undefined;

    // 3. Fallback: scan recent roadmaps and match node by title when taskId link is absent.
    if (!roadmapRow) {
      const candidates = await sql`
        SELECT id, goal, nodes, generated_at, last_updated_at 
        FROM roadmaps 
        WHERE user_id = ${userId} 
        ORDER BY last_updated_at DESC 
        LIMIT 10
      `;

      const normalize = (value: string): string =>
        value.trim().toLowerCase().replace(/\s+/g, " ");

      const normalizedTaskName = normalize(taskName);

      roadmapRow = candidates.rows.find((row) => {
        const nodes = (row.nodes as RoadmapDAGNode[]) ?? [];
        return nodes.some(
          (node) => normalize(node.title) === normalizedTaskName,
        );
      }) as typeof roadmapRow;
    }

    if (!roadmapRow) {
      // Task completion succeeded, but there is no roadmap to update.
      return { success: true };
    }

    const nodes = (roadmapRow.nodes as RoadmapDAGNode[]) ?? [];

    const normalize = (value: string): string =>
      value.trim().toLowerCase().replace(/\s+/g, " ");

    const targetNode =
      nodes.find((node) => node.taskId === taskId) ??
      nodes.find((node) => normalize(node.title) === normalize(taskName));

    if (!targetNode) {
      // If we cannot map the task to a node, keep task as completed and return success.
      return { success: true };
    }

    // 4. Update mastery for the target node and recompute others.
    const masteryUpdates: Record<string, number> = {
      [targetNode.id]: 1,
    };

    const updatedNodes = recomputeNodeStates(nodes, masteryUpdates);

    // Cross-pillar mastery signal (plans/04). This is a coarse "task done"
    // flag, not a graded score — upsertMastery's merge policy makes sure it
    // never overwrites a real assessment-sourced score for the same topic.
    await upsertMastery(
      userId,
      normalizeTopicKey(targetNode.title),
      1,
      "roadmap",
    );

    const updatedRoadmap: AdaptiveRoadmap = {
      id: roadmapRow.id,
      goal: roadmapRow.goal,
      nodes: updatedNodes,
      generatedAt: new Date(roadmapRow.generated_at).toISOString(),
      lastUpdatedAt: new Date().toISOString(),
    };

    // 5. Persist updated roadmap.
    const saveRes = await saveRoadmapAction(updatedRoadmap);
    if (saveRes.error) {
      return { error: saveRes.error };
    }

    return {
      success: true,
      roadmap: updatedRoadmap,
      completedNodeId: targetNode.id,
    };
  } catch (err) {
    console.error("[completeNodeAction] Error:", err);
    return { error: "Failed to process course completion." };
  }
}
