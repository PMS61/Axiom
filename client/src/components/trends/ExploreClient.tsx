/* ═══════════════════════════════════════════════════════════
   Trend Explore — Client Component
   Progressively loads:
     1. Topic overview (market relevance, hours, use cases)
     2. 10 subtopics + SVG dependency graph
   ═══════════════════════════════════════════════════════════ */

"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Header from "@/components/Header";
import SubtopicGraph from "./SubtopicGraph";
import { getTopicOverview, getTopicSubtopics } from "@/app/actions/trends";
import type { TopicOverview, ExploreSubtopic, ExploreSubtopicGraph } from "@/lib/trend-engine/types";

// ── Helpers ───────────────────────────────────────────────

const DIFFICULTY_LABEL: Record<string, string> = {
  beginner: "Beginner",
  intermediate: "Intermediate",
  advanced: "Advanced",
  expert: "Expert",
};

const JOB_DEMAND_COLOR: Record<string, string> = {
  low: "var(--muted)",
  moderate: "var(--ink)",
  high: "var(--watch)",
  very_high: "var(--safe)",
};

const JOB_DEMAND_LABEL: Record<string, string> = {
  low: "Low",
  moderate: "Moderate",
  high: "High",
  very_high: "Very High",
};

function difficultyBarColor(d: number): string {
  if (d <= 3) return "var(--safe)";
  if (d <= 6) return "var(--watch)";
  return "var(--vermillion)";
}

// ── Skeleton blocks ───────────────────────────────────────

function SkeletonBlock({ h, w = "100%" }: { h: number; w?: string | number }) {
  return (
    <div
      style={{
        height: h,
        width: w,
        background: "var(--rule)",
        opacity: 0.4,
        animation: "skeletonPulse 1.4s ease-in-out infinite",
      }}
    />
  );
}

// ── Subtopic Card ─────────────────────────────────────────

function SubtopicCard({
  subtopic,
  index,
  allSubtopics,
  added,
  onAdd,
}: {
  subtopic: ExploreSubtopic;
  index: number;
  allSubtopics: ExploreSubtopic[];
  added: boolean;
  onAdd: () => void;
}) {
  const prereqNames = subtopic.prerequisites.map(
    (id) => allSubtopics.find((s) => s.id === id)?.name ?? id,
  );

  return (
    <div
      style={{
        border: "0.5px solid var(--rule)",
        borderLeft: `3px solid ${difficultyBarColor(subtopic.difficulty)}`,
        background: added ? "var(--card-bg)" : "var(--bg)",
        padding: "20px",
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      {/* Header row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            className="meta-text"
            style={{ fontSize: 9, color: "var(--muted)", marginBottom: 4 }}
          >
            {String(index + 1).padStart(2, "0")}
          </div>
          <div style={{ fontWeight: 700, fontSize: 14, lineHeight: 1.3 }}>
            {subtopic.name}
          </div>
        </div>
        <div style={{ display: "flex", gap: 10, flexShrink: 0, alignItems: "center" }}>
          <div style={{ textAlign: "right" }}>
            <div
              className="meta-text"
              style={{ fontSize: 9, color: "var(--muted)" }}
            >
              HOURS
            </div>
            <div
              style={{
                fontFamily: "var(--mono)",
                fontWeight: 700,
                fontSize: 16,
                color: "var(--ink)",
              }}
            >
              {subtopic.hours}h
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div
              className="meta-text"
              style={{ fontSize: 9, color: "var(--muted)" }}
            >
              DIFF
            </div>
            <div
              style={{
                fontFamily: "var(--mono)",
                fontWeight: 700,
                fontSize: 16,
                color: difficultyBarColor(subtopic.difficulty),
              }}
            >
              {subtopic.difficulty}/10
            </div>
          </div>
        </div>
      </div>

      {/* Description */}
      <p style={{ fontSize: 12, color: "var(--muted)", margin: 0, lineHeight: 1.6 }}>
        {subtopic.description}
      </p>

      {/* Why important */}
      <div
        style={{
          fontSize: 11,
          color: "var(--ink)",
          fontStyle: "italic",
          borderLeft: "2px solid var(--rule)",
          paddingLeft: 10,
        }}
      >
        {subtopic.why_important}
      </div>

      {/* Prerequisites */}
      {prereqNames.length > 0 && (
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
          <span className="meta-text" style={{ fontSize: 9, color: "var(--muted)" }}>
            REQUIRES:
          </span>
          {prereqNames.map((name) => (
            <span
              key={name}
              style={{
                fontFamily: "var(--mono)",
                fontSize: 9,
                border: "0.5px solid var(--rule)",
                padding: "2px 6px",
                color: "var(--muted)",
                background: "var(--card-bg)",
              }}
            >
              {name}
            </span>
          ))}
        </div>
      )}
      {prereqNames.length === 0 && (
        <div className="meta-text" style={{ fontSize: 9, color: "var(--safe)" }}>
          No prerequisites — start here
        </div>
      )}

      {/* Add to Roadmap */}
      <button
        className={`btn btn-sm ${added ? "" : "btn-primary"}`}
        style={
          added
            ? { color: "var(--safe)", borderColor: "var(--safe)", marginTop: 4 }
            : { marginTop: 4 }
        }
        onClick={onAdd}
        disabled={added}
      >
        {added ? "✓ Added to Roadmap" : "+ Add to Roadmap"}
      </button>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────

export default function ExploreClient({ topic }: { topic: string }) {
  const router = useRouter();
  const [overview, setOverview] = useState<TopicOverview | null>(null);
  const [subtopics, setSubtopics] = useState<ExploreSubtopic[] | null>(null);
  const [subtopicGraph, setSubtopicGraph] = useState<ExploreSubtopicGraph | null>(null);
  const [overviewError, setOverviewError] = useState<string | null>(null);
  const [subtopicsError, setSubtopicsError] = useState<string | null>(null);
  const [overviewLoading, setOverviewLoading] = useState(true);
  const [subtopicsLoading, setSubtopicsLoading] = useState(true);
  const [addedIds, setAddedIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!topic) return;

    // Fire both calls truly in parallel — subtopics doesn't need the hours hint to start
    getTopicOverview(topic).then(({ overview: ov, error }) => {
      setOverviewLoading(false);
      if (error || !ov) { setOverviewError(error ?? "Failed"); return; }
      setOverview(ov);
    });

    getTopicSubtopics(topic).then(({ subtopics: st, graph, error: se }) => {
      setSubtopicsLoading(false);
      if (se || !st) { setSubtopicsError(se ?? "Failed"); return; }
      setSubtopics(st);
      setSubtopicGraph(graph ?? null);
    });
  }, [topic]);

  function buildRoadmapRedirect(goal: string, syllabus?: string): string {
    const params = new URLSearchParams();
    params.set("goal", goal);
    params.set("autogenerate", "1");
    if (syllabus?.trim()) {
      params.set("syllabus", syllabus.trim());
    }
    return `/dashboard/roadmap?${params.toString()}`;
  }

  function handleAddToRoadmap(subtopicId: string) {
    const subtopic = subtopics?.find((s) => s.id === subtopicId);
    if (!subtopic) return;

    setAddedIds((prev) => new Set([...prev, subtopicId]));

    const goalText = `Master ${subtopic.name} (${topic})`;
    const syllabusText = [
      `Topic context: ${topic}`,
      `Primary focus subtopic: ${subtopic.name}`,
      `Description: ${subtopic.description}`,
      `Why it matters: ${subtopic.why_important}`,
      `Difficulty: ${subtopic.difficulty}/10`,
      `Estimated effort: ${subtopic.hours}h`,
      subtopic.prerequisites.length > 0
        ? `Prerequisites: ${subtopic.prerequisites
            .map((id) => subtopics?.find((s) => s.id === id)?.name ?? id)
            .join(", ")}`
        : "Prerequisites: none",
    ].join("\n");

    router.push(buildRoadmapRedirect(goalText, syllabusText));
  }

  function handleGenerateFromAllSubtopics() {
    if (!subtopics || subtopics.length === 0) return;

    const goalText = `Master ${topic} with a structured subtopic roadmap`;
    const syllabusText = [
      `Generate roadmap strictly from these subtopics for ${topic}:`,
      ...subtopics.map(
        (s, idx) =>
          `${idx + 1}. ${s.name} | difficulty ${s.difficulty}/10 | ${s.hours}h | prereqs: ${
            s.prerequisites.length > 0
              ? s.prerequisites
                  .map((id) => subtopics.find((p) => p.id === id)?.name ?? id)
                  .join(", ")
              : "none"
          }`,
      ),
    ].join("\n");

    router.push(buildRoadmapRedirect(goalText, syllabusText));
  }

  function handleGraphNodeClick(id: string) {
    // Scroll to the subtopic card
    const el = document.getElementById(`subtopic-card-${id}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  const totalSubtopicHours = subtopics?.reduce((s, t) => s + t.hours, 0) ?? 0;

  return (
    <div style={{ minHeight: "100vh" }}>
      <Header />

      <style>{`
        @keyframes skeletonPulse {
          0%, 100% { opacity: 0.4; }
          50%       { opacity: 0.15; }
        }
      `}</style>

      {/* ── Breadcrumb ── */}
      <div
        className="container meta-text"
        style={{
          paddingTop: 24,
          paddingBottom: 0,
          display: "flex",
          gap: 8,
          alignItems: "center",
          color: "var(--muted)",
        }}
      >
        <a href="/dashboard/trends" style={{ color: "var(--muted)", textDecoration: "none" }}>
          Trend Intelligence
        </a>
        <span>→</span>
        <span style={{ color: "var(--ink)" }}>{topic}</span>
      </div>

      {/* ── Hero: Topic Overview ── */}
      <section className="container section-rule" style={{ paddingTop: 40, paddingBottom: 40 }}>
        {overviewLoading ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <SkeletonBlock h={12} w={120} />
            <SkeletonBlock h={40} w="60%" />
            <SkeletonBlock h={16} w="80%" />
            <SkeletonBlock h={16} w="70%" />
            <div style={{ display: "flex", gap: 24, marginTop: 8 }}>
              <SkeletonBlock h={56} w={100} />
              <SkeletonBlock h={56} w={100} />
              <SkeletonBlock h={56} w={100} />
              <SkeletonBlock h={56} w={100} />
            </div>
          </div>
        ) : overviewError ? (
          <div style={{ color: "var(--vermillion)", fontFamily: "var(--mono)", fontSize: 12 }}>
            [!] {overviewError}
          </div>
        ) : overview ? (
          <>
            <div
              className="responsive-grid-hero"
              style={{ display: "grid", gridTemplateColumns: "1fr 1.4fr", gap: 60, alignItems: "start" }}
            >
              {/* Left: title + description */}
              <div>
                <div className="meta-text" style={{ marginBottom: 14 }}>
                  TOPIC DEEP DIVE
                </div>
                <h1 style={{ fontSize: 38, marginBottom: 16, lineHeight: 1.15 }}>{topic}</h1>
                <p style={{ color: "var(--muted)", lineHeight: 1.7, marginBottom: 24, fontSize: 14 }}>
                  {overview.description}
                </p>
                <div
                  style={{
                    borderLeft: "2px solid var(--ink)",
                    paddingLeft: 16,
                    fontSize: 13,
                    color: "var(--ink)",
                    lineHeight: 1.7,
                    marginBottom: 24,
                  }}
                >
                  {overview.why_now}
                </div>

                {/* References */}
                {overview.references?.length > 0 && (
                  <div>
                    <div className="meta-text" style={{ marginBottom: 10 }}>
                      RECOMMENDED RESOURCES
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                      {overview.references.map((ref, i) => (
                        <div
                          key={i}
                          style={{
                            fontSize: 12,
                            color: "var(--muted)",
                            display: "flex",
                            gap: 8,
                            alignItems: "baseline",
                          }}
                        >
                          <span
                            style={{
                              fontFamily: "var(--mono)",
                              fontSize: 9,
                              color: "var(--rule)",
                            }}
                          >
                            {String(i + 1).padStart(2, "0")}
                          </span>
                          {ref}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Right: stats grid */}
              <div>
                <div
                  className="meta-text"
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 1,
                    border: "0.5px solid var(--rule)",
                    background: "var(--rule)",
                  }}
                >
                  {[
                    {
                      label: "Market Relevance",
                      value: `${overview.market_relevance}/100`,
                      color: overview.market_relevance >= 70 ? "var(--safe)" : "var(--ink)",
                    },
                    {
                      label: "Job Demand",
                      value: JOB_DEMAND_LABEL[overview.job_demand] ?? overview.job_demand,
                      color: JOB_DEMAND_COLOR[overview.job_demand] ?? "var(--ink)",
                    },
                    {
                      label: "Total Hours to Learn",
                      value: `~${overview.total_hours}h`,
                      color: "var(--ink)",
                    },
                    {
                      label: `At ${overview.daily_hours_assumed}h/day`,
                      value: `${overview.personalised_weeks} weeks`,
                      color: "var(--ink)",
                    },
                    {
                      label: "Difficulty",
                      value: DIFFICULTY_LABEL[overview.difficulty] ?? overview.difficulty,
                      color: overview.difficulty === "expert" || overview.difficulty === "advanced"
                        ? "var(--vermillion)"
                        : overview.difficulty === "intermediate"
                          ? "var(--watch)"
                          : "var(--safe)",
                    },
                  ].map(({ label, value, color }) => (
                    <div
                      key={label}
                      style={{
                        background: "var(--card-bg)",
                        padding: "20px",
                      }}
                    >
                      <div style={{ fontSize: 9, color: "var(--muted)", letterSpacing: "0.12em" }}>
                        {label.toUpperCase()}
                      </div>
                      <div
                        style={{
                          fontWeight: 700,
                          fontSize: 22,
                          marginTop: 6,
                          fontFamily: "var(--mono)",
                          color,
                        }}
                      >
                        {value}
                      </div>
                    </div>
                  ))}

                  {/* Key use cases — full width */}
                  <div
                    style={{
                      background: "var(--card-bg)",
                      padding: "20px",
                      gridColumn: "1 / -1",
                    }}
                  >
                    <div style={{ fontSize: 9, color: "var(--muted)", letterSpacing: "0.12em", marginBottom: 12 }}>
                      KEY USE CASES
                    </div>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {overview.key_use_cases.map((uc, i) => (
                        <span
                          key={i}
                          style={{
                            fontSize: 11,
                            border: "0.5px solid var(--rule)",
                            padding: "4px 10px",
                            color: "var(--ink)",
                          }}
                        >
                          {uc}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        ) : null}
      </section>

      {/* ── Dependency Graph ── */}
      <section className="container section-rule" style={{ paddingTop: 40, paddingBottom: 40 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
            marginBottom: 24,
          }}
        >
          <div>
            <div className="meta-text" style={{ marginBottom: 6 }}>
              DEPENDENCY GRAPH
            </div>
            <h2 style={{ fontSize: 20, margin: 0 }}>
              {subtopicsLoading ? "Building subtopic graph…" : `10 Subtopics · ${totalSubtopicHours}h total`}
            </h2>
          </div>
          <div className="meta-text" style={{ color: "var(--muted)" }}>
            Left = prerequisites · Right = advanced
          </div>
        </div>

        {subtopicsLoading ? (
          <div
            style={{
              height: 280,
              border: "0.5px solid var(--rule)",
              background: "var(--bg)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 12,
            }}
          >
            <div
              style={{
                width: 8,
                height: 8,
                background: "var(--ink)",
                animation: "skeletonPulse 1s ease-in-out infinite",
              }}
            />
            <span className="meta-text" style={{ color: "var(--muted)" }}>
              Generating subtopics with Gemma…
            </span>
          </div>
        ) : subtopicsError ? (
          <div style={{ color: "var(--vermillion)", fontFamily: "var(--mono)", fontSize: 12 }}>
            [!] {subtopicsError}
          </div>
        ) : subtopics ? (
          <SubtopicGraph
            subtopics={subtopics}
            graph={subtopicGraph ?? undefined}
            addedIds={addedIds}
            onNodeClick={handleGraphNodeClick}
          />
        ) : null}
      </section>

      {/* ── Subtopic Cards ── */}
      {(subtopics || subtopicsLoading) && (
        <section className="container" style={{ paddingTop: 0, paddingBottom: 80 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              paddingBottom: 20,
              borderBottom: "0.5px solid var(--rule)",
              marginBottom: 24,
            }}
          >
            <h2 style={{ fontSize: 18, margin: 0 }}>Subtopics</h2>
            {subtopics && (
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <span className="meta-text" style={{ color: "var(--muted)" }}>
                  Click graph nodes to highlight · Generate directly from listed subtopics
                </span>
                <button className="btn btn-sm btn-primary" onClick={handleGenerateFromAllSubtopics}>
                  Generate Roadmap from All
                </button>
              </div>
            )}
          </div>

          {subtopicsLoading ? (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 16,
              }}
            >
              {Array.from({ length: 6 }).map((_, i) => (
                <div
                  key={i}
                  style={{
                    border: "0.5px solid var(--rule)",
                    padding: 20,
                    display: "flex",
                    flexDirection: "column",
                    gap: 12,
                  }}
                >
                  <SkeletonBlock h={12} w={60} />
                  <SkeletonBlock h={18} w="70%" />
                  <SkeletonBlock h={12} w="90%" />
                  <SkeletonBlock h={12} w="80%" />
                </div>
              ))}
            </div>
          ) : subtopics ? (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(2, 1fr)",
                gap: 16,
              }}
            >
              {subtopics.map((s, i) => (
                <div id={`subtopic-card-${s.id}`} key={s.id}>
                  <SubtopicCard
                    subtopic={s}
                    index={i}
                    allSubtopics={subtopics}
                    added={addedIds.has(s.id)}
                    onAdd={() => handleAddToRoadmap(s.id)}
                  />
                </div>
              ))}
            </div>
          ) : null}
        </section>
      )}
    </div>
  );
}
