/* ═══════════════════════════════════════════════════════════
   Trend Engine — Type Definitions
   ═══════════════════════════════════════════════════════════ */

export interface RawSignal {
  title: string;
  source: "news" | "reddit" | "hackernews";
  publishedAt: string;   // ISO timestamp
  engagement: number;    // upvotes / comments / shares
  url: string;
}

export interface TopicSignals {
  topic: string;
  news_mentions: number;
  reddit_posts: number;
  hackernews_posts: number;
  avg_engagement: number;
  total_engagement: number;
  recency_score: number;   // 0–1, weighted by how recent articles are
  raw: RawSignal[];
}

export interface TrendResult {
  topic: string;
  trend_score: number;    // 0–100
  direction: "rising" | "stable" | "declining";
  momentum: "accelerating" | "steady" | "slowing";
  signals: {
    news_mentions: number;
    reddit_posts: number;
    hackernews_posts: number;
    avg_engagement: number;
  };
  insight: string;         // LLM-generated
  future_outlook: string;  // LLM-generated
  // Time series for graphs (last 7 data points)
  mentions_over_time: number[];
  engagement_over_time: number[];
}

export interface TopicSearchResult {
  topic: string;
  relevance_score: number;   // 0–100
  effort_estimate: "low" | "medium" | "high" | "very_high";
  roi_estimate: "low" | "medium" | "high" | "very_high";
  competition_level: "low" | "medium" | "high";
  recommendation: string;
  suggested_depth: "overview" | "working_knowledge" | "proficient" | "expert";
  reasoning: string;
  trend_data?: TrendResult;
}

export interface RoadmapPayload {
  topic: string;
  recommended_depth: string;
  priority: number;   // 1–10
}

export interface UserInterests {
  domains: string[];
  profile_type: string;
}

// Domains available for user selection
export const DOMAINS = [
  "AI & Machine Learning",
  "Web Development",
  "Mobile Development",
  "Cloud & DevOps",
  "Data Science",
  "Cybersecurity",
  "Blockchain & Web3",
  "Systems Programming",
  "Game Development",
  "UI/UX Design",
  "Finance & FinTech",
  "Entrepreneurship",
] as const;

export type Domain = (typeof DOMAINS)[number];

// Profile types for personalization
export const PROFILE_TYPES = [
  "student",
  "developer",
  "data_scientist",
  "designer",
  "entrepreneur",
  "finance_professional",
  "content_creator",
  "researcher",
] as const;

export type ProfileType = (typeof PROFILE_TYPES)[number];

// ── Explore page types ────────────────────────────────────

export interface TopicOverview {
  description: string;
  market_relevance: number;      // 0–100
  why_now: string;
  total_hours: number;
  personalised_weeks: number;
  daily_hours_assumed: number;
  difficulty: "beginner" | "intermediate" | "advanced" | "expert";
  job_demand: "low" | "moderate" | "high" | "very_high";
  key_use_cases: string[];
  references: string[];
}

export interface ExploreSubtopic {
  id: string;                    // snake_case unique within this list
  name: string;
  description: string;
  hours: number;
  difficulty: number;            // 1–10
  prerequisites: string[];       // ids from same list only (closed graph)
  why_important: string;
}

export interface ExploreSubtopicGraph {
  nodes: Array<{
    id: string;
    name: string;
    prerequisites: string[];
  }>;
  edges: Array<{
    from: string;
    to: string;
  }>;
  generated_at: string;
}
