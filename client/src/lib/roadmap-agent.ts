/**
 * ═══════════════════════════════════════════════════════════
 *  THE AXIOM — Adaptive Roadmap Agent
 *  Uses Gemini API (gemma-4-31b-it) to generate a
 *  personalised DAG of learning topics based on user profile,
 *  goals, and mastery scores. Each node maps to a CogFlow task.
 * ═══════════════════════════════════════════════════════════
 */

import { getGeminiResponse } from "./gemini";
import type { RoadmapDAGNode, RoadmapGoal } from "./types";

// ── Prompt engineering ────────────────────────────────────

const SYSTEM_RULES = `
You are an expert adaptive learning curriculum designer. Your job is to generate a 
personalised learning roadmap as a Directed Acyclic Graph (DAG) based on the learner's 
profile, stated goal, and current mastery scores.

MANDATORY RULES:
- Respond ONLY with valid JSON. No markdown fences, no commentary.
- The JSON must be an array of node objects matching the schema exactly.
- Each node has a unique "id" string (snake_case, e.g. "intro_algorithms").
- "prerequisites" is an array of other node "id"s that MUST be completed first. 
  Root nodes have an empty prerequisites array [].
- "type" must be exactly "course" or "assessment".
- "priority" must be "low", "normal", or "high".
- "sequence" must be an integer indicating execution order.
- "difficulty" is 1–10.
- "estimatedMinutes" is realistic study time in minutes (30–180).
- "masteryScore" starts at 0.0 (defaults for new topics).
- Limit total nodes to 12–18 for a focused learning path.
- Ensure the DAG has no cycles.
- Use exactly 1 assessement for every 3 course nodes (ratio enforcement).
- "subject" should be a single thematic grouping tag (e.g. "Foundations", "Core Concepts", "Advanced", "Applied").
- Sequence must be logical: prerequisites come before dependents.
`;

const NODE_SCHEMA = `
{
  "id": "unique_snake_case_identifier",
  "title": "Human readable topic title",
  "description": "2–3 sentence summary of what the learner will study or be assessed on",
  "type": "course",
  "priority": "normal",
  "sequence": 1,
  "difficulty": 1-10,
  "estimatedMinutes": 30-180,
  "prerequisites": ["other_node_id", ...],
  "masteryScore": 0.0,
  "subject": "Thematic tag/group",
  "nodeState": "locked"
}
`;

function buildPrompt(goal: RoadmapGoal): string {
  const masteryContext =
    goal.existingMastery && goal.existingMastery.length > 0
      ? `\nCurrent Mastery Scores (skip or fast-track topics where mastery > 0.75):\n${goal.existingMastery.map((m) => `  - ${m.topic}: ${(m.score * 100).toFixed(0)}%`).join("\n")}`
      : "";

  const syllabusContext = goal.syllabus
    ? `\nProvided Syllabus / Reference Material:\n${goal.syllabus}`
    : "";

  const profileContext = goal.userProfile
    ? `\nLearner Profile:
  - Experience Level: ${goal.userProfile.experienceLevel}
  - Available Time Per Day: ${goal.userProfile.dailyMinutes} minutes
  - Learning Style: ${goal.userProfile.learningStyle || "balanced"}
  - Deadline: ${goal.deadline || "flexible"}`
    : "";

  return `${SYSTEM_RULES}

GOAL: ${goal.goal}
${profileContext}
${masteryContext}
${syllabusContext}

Generate a complete adaptive learning roadmap DAG for this goal.
Each node schema:
${NODE_SCHEMA}

Respond with ONLY the JSON array.`;
}

// ── Helpers ───────────────────────────────────────────────

/**
 * Aggressively strip markdown code-fences, leading/trailing whitespace,
 * and any preamble text before the first `[`.
 */
function cleanJsonResponse(raw: string): string {
  let cleaned = raw.trim();
  // Remove ```json ... ``` or ``` ... ``` (multiline-safe)
  cleaned = cleaned.replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/, "");
  // If there's preamble text before the JSON array, strip it
  const arrayStart = cleaned.indexOf("[");
  if (arrayStart > 0) {
    cleaned = cleaned.slice(arrayStart);
  }
  // If there's trailing text after the last `]`, strip it
  const arrayEnd = cleaned.lastIndexOf("]");
  if (arrayEnd !== -1 && arrayEnd < cleaned.length - 1) {
    cleaned = cleaned.slice(0, arrayEnd + 1);
  }
  return cleaned.trim();
}

/**
 * Parse and validate that the response is a proper JSON array of nodes.
 * Throws descriptive errors on failure.
 */
function parseAndValidateNodes(raw: string): RoadmapDAGNode[] {
  const cleaned = cleanJsonResponse(raw);
  const parsed = JSON.parse(cleaned);

  // Guard: must be an array
  if (!Array.isArray(parsed)) {
    throw new Error(
      `Expected a JSON array of nodes but received ${typeof parsed}: ${JSON.stringify(parsed).slice(0, 200)}`,
    );
  }

  if (parsed.length === 0) {
    throw new Error("Received an empty node array from the model.");
  }

  return parsed;
}

/**
 * Hydrate and validate individual node fields, ensuring sane defaults.
 */
function hydrateNodes(parsed: RoadmapDAGNode[]): RoadmapDAGNode[] {
  const validatedNodes = parsed.map((node, idx) => ({
    id: node.id || `node_${idx}`,
    title: node.title || "Untitled Topic",
    description: node.description || "",
    type: (node.type === "assessment" ? "assessment" : "course") as
      | "course"
      | "assessment",
    priority: ["low", "normal", "high"].includes(node.priority as string)
      ? (node.priority as "low" | "normal" | "high")
      : ("normal" as const),
    sequence: typeof node.sequence === "number" ? node.sequence : idx + 1,
    difficulty: Math.min(10, Math.max(1, node.difficulty ?? 5)),
    estimatedMinutes: Math.min(180, Math.max(15, node.estimatedMinutes ?? 60)),
    prerequisites: Array.isArray(node.prerequisites) ? node.prerequisites : [],
    masteryScore: node.masteryScore ?? 0,
    subject: node.subject || "Core",
    nodeState: "locked" as const,
  }));

  // Compute initial nodeState: root nodes (no prerequisites) start as "available"
  const nodeIds = new Set(validatedNodes.map((n) => n.id));
  return validatedNodes.map((node) => ({
    ...node,
    // Ensure prerequisites reference valid nodes only
    prerequisites: node.prerequisites.filter((p: string) => nodeIds.has(p)),
    nodeState:
      node.prerequisites.length === 0
        ? ("available" as const)
        : ("locked" as const),
  }));
}

// ── Main agent function ───────────────────────────────────

const MAX_ROADMAP_ATTEMPTS = 2;

export async function generateRoadmapDAG(
  goal: RoadmapGoal,
): Promise<RoadmapDAGNode[]> {
  const prompt = buildPrompt(goal);
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= MAX_ROADMAP_ATTEMPTS; attempt++) {
    try {
      console.log(`[RoadmapAgent] Attempt ${attempt}/${MAX_ROADMAP_ATTEMPTS}…`);
      const raw = await getGeminiResponse(prompt, true);
      const parsed = parseAndValidateNodes(raw);
      const nodes = hydrateNodes(parsed);
      console.log(
        `[RoadmapAgent] Successfully generated ${nodes.length} nodes.`,
      );
      return nodes;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      console.error(
        `[RoadmapAgent] Attempt ${attempt} failed:`,
        lastError.message,
      );
      // Brief pause before retry
      if (attempt < MAX_ROADMAP_ATTEMPTS) {
        await new Promise((r) => setTimeout(r, 2000));
      }
    }
  }

  throw new Error(
    `Roadmap Agent failed after ${MAX_ROADMAP_ATTEMPTS} attempts: ${lastError?.message ?? "Unknown error"}`,
  );
}

// ── Mastery → Node State propagation ─────────────────────

/**
 * Given updated mastery scores (from Assessment Agent), re-compute
 * the nodeState for every node in the DAG.
 * - mastery >= 0.8  → "mastered"
 * - mastery > 0  → "in_progress"
 * - all prerequisites mastered → "available"
 * - otherwise → "locked"
 */
export function recomputeNodeStates(
  nodes: RoadmapDAGNode[],
  masteryUpdates: Record<string, number>,
): RoadmapDAGNode[] {
  const updatedNodes = nodes.map((node) => ({
    ...node,
    masteryScore: masteryUpdates[node.id] ?? node.masteryScore,
  }));

  return updatedNodes.map((node) => {
    const mastery = node.masteryScore;

    if (mastery >= 0.8) {
      return { ...node, nodeState: "mastered" as const };
    }

    // Check if all prerequisites are mastered
    const allPrereqsMet = node.prerequisites.every((prereqId) => {
      const prereq = updatedNodes.find((n) => n.id === prereqId);
      return prereq && prereq.masteryScore >= 0.8;
    });

    if (allPrereqsMet && node.prerequisites.length > 0) {
      return { ...node, nodeState: mastery > 0 ? "in_progress" : "available" };
    }

    if (node.prerequisites.length === 0) {
      return { ...node, nodeState: mastery > 0 ? "in_progress" : "available" };
    }

    return { ...node, nodeState: "locked" as const };
  });
}
