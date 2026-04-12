/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Trend Intelligence Server Actions
   All LLM calls use Gemini. All DB via Vercel Postgres.
   Cache TTL: 2 hours.
   ═══════════════════════════════════════════════════════════ */

"use server";

import { sql } from "@vercel/postgres";
import { cookies } from "next/headers";
import { verify } from "jsonwebtoken";
import { getGeminiResponse } from "@/lib/gemini";
import { processTrends, processSingleTopic } from "@/lib/trend-engine/processTrends";
import { normalizeQueryToTopic } from "@/lib/trend-engine/topicMapper";
import type {
  TrendResult,
  TopicSearchResult,
  RoadmapPayload,
  UserInterests,
  TopicOverview,
  ExploreSubtopic,
  ExploreSubtopicGraph,
} from "@/lib/trend-engine/types";

// ── Auth ──────────────────────────────────────────────────

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

async function ensureTrendTables(): Promise<void> {
  await sql`
    CREATE TABLE IF NOT EXISTS trend_cache (
      id SERIAL PRIMARY KEY,
      cache_key TEXT NOT NULL UNIQUE,
      data JSONB NOT NULL,
      created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )
  `;

  await sql`
    CREATE TABLE IF NOT EXISTS user_interests (
      user_id INT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
      domains TEXT[] NOT NULL DEFAULT '{}',
      profile_type TEXT NOT NULL DEFAULT 'developer',
      updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )
  `;

  await sql`
    CREATE TABLE IF NOT EXISTS shared_topic_subtopic_graphs (
      id SERIAL PRIMARY KEY,
      topic_key TEXT NOT NULL UNIQUE,
      topic_label TEXT NOT NULL,
      subtopics JSONB NOT NULL,
      graph JSONB NOT NULL,
      created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )
  `;
}

// ── Cache helpers ─────────────────────────────────────────

const CACHE_TTL_MS = 2 * 60 * 60 * 1000; // 2 hours

function topicKeyFromTopic(topic: string): string {
  return topic.toLowerCase().trim().replace(/\s+/g, "_");
}

function buildSubtopicGraph(subtopics: ExploreSubtopic[]): ExploreSubtopicGraph {
  return {
    nodes: subtopics.map((s) => ({
      id: s.id,
      name: s.name,
      prerequisites: s.prerequisites,
    })),
    edges: subtopics.flatMap((s) =>
      s.prerequisites.map((from) => ({
        from,
        to: s.id,
      })),
    ),
    generated_at: new Date().toISOString(),
  };
}

async function getSharedSubtopics(topicKey: string): Promise<{
  subtopics: ExploreSubtopic[];
  graph: ExploreSubtopicGraph;
} | null> {
  try {
    const result = await sql`
      SELECT subtopics, graph
      FROM shared_topic_subtopic_graphs
      WHERE topic_key = ${topicKey}
      LIMIT 1
    `;

    if (!result.rows.length) return null;

    return {
      subtopics: result.rows[0].subtopics as ExploreSubtopic[],
      graph: result.rows[0].graph as ExploreSubtopicGraph,
    };
  } catch {
    return null;
  }
}

async function upsertSharedSubtopics(
  topicKey: string,
  topicLabel: string,
  subtopics: ExploreSubtopic[],
  graph: ExploreSubtopicGraph,
): Promise<void> {
  try {
    await sql`
      INSERT INTO shared_topic_subtopic_graphs (topic_key, topic_label, subtopics, graph, created_at, updated_at)
      VALUES (
        ${topicKey},
        ${topicLabel},
        ${JSON.stringify(subtopics)}::jsonb,
        ${JSON.stringify(graph)}::jsonb,
        NOW(),
        NOW()
      )
      ON CONFLICT (topic_key) DO UPDATE SET
        topic_label = EXCLUDED.topic_label,
        subtopics = EXCLUDED.subtopics,
        graph = EXCLUDED.graph,
        updated_at = NOW()
    `;
  } catch (err) {
    console.error("upsertSharedSubtopics failed:", err);
  }
}

async function getCached<T>(key: string): Promise<T | null> {
  try {
    const result = await sql`
      SELECT data, created_at FROM trend_cache WHERE cache_key = ${key} LIMIT 1
    `;
    if (!result.rows.length) return null;
    const row = result.rows[0] as { data: T; created_at: string };
    const age = Date.now() - new Date(row.created_at).getTime();
    if (age > CACHE_TTL_MS) return null;
    return row.data;
  } catch {
    return null;
  }
}

async function setCached<T>(key: string, data: T): Promise<void> {
  try {
    await sql`
      INSERT INTO trend_cache (cache_key, data, created_at)
      VALUES (${key}, ${JSON.stringify(data)}::jsonb, NOW())
      ON CONFLICT (cache_key) DO UPDATE SET
        data       = EXCLUDED.data,
        created_at = NOW()
    `;
  } catch (err) {
    console.error("setCached failed:", err);
  }
}

// ── Action 1: Fetch Top Trends ────────────────────────────

export async function fetchTopTrends(forceRefresh = false): Promise<{
  trends?: TrendResult[];
  error?: string;
  cached?: boolean;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  await ensureTrendTables();

  const cacheKey = "top_trends_v1";

  if (!forceRefresh) {
    const cached = await getCached<TrendResult[]>(cacheKey);
    if (cached) return { trends: cached, cached: true };
  }

  try {
    // 1. Get raw scored trends (no LLM)
    const { trends: rawTrends } = await processTrends(10);

    // 2. Generate LLM insights for all topics in one batch call
    const insightPrompt = `You are a tech trend analyst for a personalized learning platform called Axiom.

For each of the following trending tech topics, generate:
1. A brief "insight" (1-2 sentences): why this topic is trending RIGHT NOW and why a learner should care.
2. A "future_outlook" (1 sentence): where this technology is heading in the next 6-12 months.

Topics: ${rawTrends.map((t, i) => `${i + 1}. ${t.topic} (trend score: ${t.trend_score}, direction: ${t.direction})`).join("\n")}

Respond with ONLY a JSON array in this exact format:
[
  { "topic": "...", "insight": "...", "future_outlook": "..." },
  ...
]`;

    const insightRaw = await getGeminiResponse(insightPrompt, true);
    let insights: Array<{ topic: string; insight: string; future_outlook: string }> = [];

    try {
      insights = JSON.parse(insightRaw);
    } catch {
      console.error("Failed to parse trend insights JSON");
    }

    // 3. Merge insights into trend results
    const trends: TrendResult[] = rawTrends.map((t) => {
      const match = insights.find(
        (ins) => ins.topic.toLowerCase() === t.topic.toLowerCase(),
      );
      return {
        ...t,
        insight: match?.insight ?? `${t.topic} is gaining significant traction in the tech community.`,
        future_outlook: match?.future_outlook ?? "This area is expected to continue growing.",
      };
    });

    await setCached(cacheKey, trends);
    return { trends, cached: false };
  } catch (err) {
    console.error("fetchTopTrends failed:", err);
    return { error: "Failed to fetch trends. Check API keys and try again." };
  }
}

// ── Action 2: Search Topic ────────────────────────────────

export async function searchTopic(query: string): Promise<{
  result?: TopicSearchResult;
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  if (!query.trim()) return { error: "Query cannot be empty" };

  await ensureTrendTables();

  const normalizedTopic = normalizeQueryToTopic(query);
  const cacheKey = `search_${normalizedTopic.toLowerCase().replace(/\s+/g, "_")}`;

  const cached = await getCached<TopicSearchResult>(cacheKey);
  if (cached) return { result: cached };

  try {
    // Get real signal data for this topic
    const trendData = await processSingleTopic(query);

    // Use Gemini to generate the full assessment
    const prompt = `You are a learning advisor on Axiom, a personalized learning OS.

A user asked: "Should I learn ${query}?"

${trendData ? `Current trend data:
- Trend score: ${trendData.trend_score}/100
- Direction: ${trendData.direction}
- Momentum: ${trendData.momentum}
- News mentions (48h): ${trendData.signals.news_mentions}
- Reddit posts (48h): ${trendData.signals.reddit_posts}
- Avg engagement: ${trendData.signals.avg_engagement}` : "No live trend data available — use your knowledge."}

Provide a comprehensive learning recommendation. Respond with ONLY this JSON:
{
  "relevance_score": <0-100, how relevant this skill is in 2025-2026>,
  "effort_estimate": <"low" | "medium" | "high" | "very_high">,
  "roi_estimate": <"low" | "medium" | "high" | "very_high">,
  "competition_level": <"low" | "medium" | "high">,
  "recommendation": <one clear sentence: should they learn it or not, and why>,
  "suggested_depth": <"overview" | "working_knowledge" | "proficient" | "expert">,
  "reasoning": <2-3 sentences explaining effort vs ROI, job market, and future potential>
}`;

    const raw = await getGeminiResponse(prompt, true);
    const parsed = JSON.parse(raw);

    const result: TopicSearchResult = {
      topic: normalizedTopic,
      relevance_score: parsed.relevance_score ?? 50,
      effort_estimate: parsed.effort_estimate ?? "medium",
      roi_estimate: parsed.roi_estimate ?? "medium",
      competition_level: parsed.competition_level ?? "medium",
      recommendation: parsed.recommendation ?? "Consider learning this topic.",
      suggested_depth: parsed.suggested_depth ?? "working_knowledge",
      reasoning: parsed.reasoning ?? "",
      trend_data: trendData
        ? { ...trendData, insight: parsed.recommendation, future_outlook: parsed.reasoning }
        : undefined,
    };

    await setCached(cacheKey, result);
    return { result };
  } catch (err) {
    console.error("searchTopic failed:", err);
    return { error: "Failed to analyze topic. Please try again." };
  }
}

// ── Action 3: Generate Deep Insight ──────────────────────

export async function generateTopicInsight(topic: string): Promise<{
  insight?: {
    why_it_matters: string;
    should_learn: boolean;
    effort_vs_roi: string;
    learning_path_hint: string;
  };
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  const cacheKey = `insight_${topic.toLowerCase().replace(/\s+/g, "_")}`;
  const cached = await getCached<{
    why_it_matters: string;
    should_learn: boolean;
    effort_vs_roi: string;
    learning_path_hint: string;
  }>(cacheKey);
  if (cached) return { insight: cached };

  const prompt = `You are a senior tech educator on Axiom, a personalized learning OS.

Generate a deep insight for the topic: "${topic}"

Respond with ONLY this JSON:
{
  "why_it_matters": "<2-3 sentences: why this topic matters in 2025-2026, real-world impact>",
  "should_learn": <true or false>,
  "effort_vs_roi": "<1-2 sentences: honest assessment of learning curve vs career/project value>",
  "learning_path_hint": "<1 sentence: what to learn first to get started with this topic>"
}`;

  try {
    const raw = await getGeminiResponse(prompt, true);
    const insight = JSON.parse(raw);
    await setCached(cacheKey, insight);
    return { insight };
  } catch (err) {
    console.error("generateTopicInsight failed:", err);
    return { error: "Failed to generate insight." };
  }
}

// ── Action 4: Add to Roadmap ──────────────────────────────

export async function addTopicToRoadmap(topic: string, trendScore: number): Promise<{
  payload?: RoadmapPayload;
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  // Compute priority from trend score (1–10 scale)
  const priority = Math.max(1, Math.min(10, Math.round(trendScore / 10)));

  // Determine depth from trend direction / score
  let recommended_depth = "working_knowledge";
  if (trendScore >= 75) recommended_depth = "proficient";
  if (trendScore >= 90) recommended_depth = "expert";
  if (trendScore < 40) recommended_depth = "overview";

  const payload: RoadmapPayload = {
    topic,
    recommended_depth,
    priority,
  };

  // In the future, this will write to the roadmap_nodes table.
  // For now, return the structured payload for the Roadmap Agent to consume.
  return { payload };
}

// ── Action 5: Save User Interests ────────────────────────

export async function saveUserInterests(interests: UserInterests): Promise<{
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  await ensureTrendTables();

  try {
    await sql`
      INSERT INTO user_interests (user_id, domains, profile_type, updated_at)
      VALUES (${userId}, ${interests.domains as unknown as string}, ${interests.profile_type}, NOW())
      ON CONFLICT (user_id) DO UPDATE SET
        domains      = EXCLUDED.domains,
        profile_type = EXCLUDED.profile_type,
        updated_at   = NOW()
    `;
    return {};
  } catch (err) {
    console.error("saveUserInterests failed:", err);
    return { error: "Failed to save interests." };
  }
}

// ── Action 6: Topic Overview (Explore page) ───────────────

export async function getTopicOverview(topic: string): Promise<{
  overview?: TopicOverview;
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  await ensureTrendTables();

  const cacheKey = `overview_${topic.toLowerCase().replace(/\s+/g, "_")}`;
  const cached = await getCached<TopicOverview>(cacheKey);
  if (cached) return { overview: cached };

  // Get user's daily study hours from section_weights
  let dailyHours = 2;
  try {
    const sw = await sql`SELECT morning, afternoon, evening FROM section_weights WHERE user_id = ${userId} LIMIT 1`;
    if (sw.rows.length > 0) {
      const { morning, afternoon, evening } = sw.rows[0] as { morning: number; afternoon: number; evening: number };
      // 50 axioms/day * (morning+afternoon+evening weights) * ~4 min/axiom → hours
      const dailyMinutes = (morning + afternoon + evening) * 50 * 4;
      dailyHours = Math.round((dailyMinutes / 60) * 10) / 10;
    }
  } catch { /* use default */ }

  const prompt = `You are a tech learning advisor on Axiom, a personalised learning OS.

Topic: "${topic}"
User studies approximately ${dailyHours} hours/day.

Generate a comprehensive learning overview. Respond ONLY with this JSON:
{
  "description": "<2-3 sentences: what this topic is and what you can build or do with it>",
  "market_relevance": <0-100, current employer/industry demand score>,
  "why_now": "<2 sentences: why 2025-2026 is a good time to learn this, citing real market signals>",
  "total_hours": <realistic total hours to reach working-knowledge level, integer>,
  "personalised_weeks": <weeks at ${dailyHours} hours/day to complete total_hours>,
  "daily_hours_assumed": ${dailyHours},
  "difficulty": "<beginner|intermediate|advanced|expert>",
  "job_demand": "<low|moderate|high|very_high>",
  "key_use_cases": ["<use case 1>", "<use case 2>", "<use case 3>", "<use case 4>"],
  "references": ["<book or course title 1>", "<book or course title 2>", "<official doc or resource 3>"]
}`;

  try {
    const raw = await getGeminiResponse(prompt, true);
    const overview: TopicOverview = JSON.parse(raw);
    await setCached(cacheKey, overview);
    return { overview };
  } catch (err) {
    console.error("getTopicOverview failed:", err);
    return { error: "Failed to generate topic overview." };
  }
}

// ── Action 7: Topic Subtopics (Explore page) ──────────────

export async function getTopicSubtopics(topic: string, totalHours?: number): Promise<{
  subtopics?: ExploreSubtopic[];
  graph?: ExploreSubtopicGraph;
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  await ensureTrendTables();

  const topicKey = topicKeyFromTopic(topic);
  const cacheKey = `subtopics_${topicKey}`;

  // Shared persistent store (all users) takes priority.
  const shared = await getSharedSubtopics(topicKey);
  if (shared) {
    return { subtopics: shared.subtopics, graph: shared.graph };
  }

  const cached = await getCached<ExploreSubtopic[]>(cacheKey);
  if (cached) {
    const graph = buildSubtopicGraph(cached);
    await upsertSharedSubtopics(topicKey, topic, cached, graph);
    return { subtopics: cached, graph };
  }

  const hoursHint = totalHours ? `Total hours across all subtopics should sum to approximately ${totalHours} hours.` : "";

  const prompt = `You are a curriculum designer on Axiom, a personalised learning OS.

Parent topic: "${topic}"

Generate exactly 10 subtopics that form a complete, ordered learning path.

STRICT RULES:
- Each subtopic must have a unique snake_case "id"
- "prerequisites" must ONLY contain ids from this same list of 10 (closed graph — no external references)
- No cycles allowed in the prerequisite graph
- Root subtopics (no prerequisites) come first logically
- "hours" is realistic focused study time per subtopic (range: 2–25 hours)
- "difficulty" is 1–10 integer
${hoursHint}

Respond ONLY with a JSON array of exactly 10 objects:
[
  {
    "id": "snake_case_id",
    "name": "Human Readable Name",
    "description": "<2-3 sentences: what the learner will study and understand>",
    "hours": <integer>,
    "difficulty": <1-10>,
    "prerequisites": ["id_of_prereq"],
    "why_important": "<1 sentence: why this subtopic matters for mastering ${topic}>"
  }
]`;

  try {
    const raw = await getGeminiResponse(prompt, true);
    const subtopics: ExploreSubtopic[] = JSON.parse(raw);

    // Validate: ensure prerequisites only reference ids in the list
    const idSet = new Set(subtopics.map((s) => s.id));
    const cleaned = subtopics.map((s) => ({
      ...s,
      prerequisites: s.prerequisites.filter((p) => idSet.has(p) && p !== s.id),
    }));

    const graph = buildSubtopicGraph(cleaned);

    await setCached(cacheKey, cleaned);
    await upsertSharedSubtopics(topicKey, topic, cleaned, graph);

    return { subtopics: cleaned, graph };
  } catch (err) {
    console.error("getTopicSubtopics failed:", err);
    return { error: "Failed to generate subtopics." };
  }
}

export async function getUserInterests(): Promise<{
  interests?: UserInterests;
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  await ensureTrendTables();

  try {
    const result = await sql`
      SELECT domains, profile_type FROM user_interests WHERE user_id = ${userId} LIMIT 1
    `;
    if (!result.rows.length) return { interests: { domains: [], profile_type: "developer" } };
    const row = result.rows[0] as { domains: string[]; profile_type: string };
    return { interests: { domains: row.domains, profile_type: row.profile_type } };
  } catch (err) {
    console.error("getUserInterests failed:", err);
    return { error: "Failed to fetch interests." };
  }
}
