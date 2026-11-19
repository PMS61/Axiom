"use client";

import Link from "next/link";
import { useState } from "react";
import RoadmapDAGView from "@/components/RoadmapDAGView";
import RoadmapGoalModal from "@/components/RoadmapGoalModal";
import { roadmapSchema } from "@/lib/learning-content";
import { updateLibrary, useLibrary } from "@/lib/learning-library";
import type { RoadmapGoal } from "@/lib/types";
import s from "../demo.module.css";
import { learnerName, milestones, roadmapTitle } from "../demo-data";
import l from "../learning.module.css";

export default function RoadmapWorkspace({ prefill }: { prefill?: string }) {
  const library = useLibrary();
  const [showForm, setShowForm] = useState(Boolean(prefill));
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const selected = library.roadmaps.find(
    ({ id }) => id === library.activeRoadmapId,
  );
  const title = selected?.goal ?? roadmapTitle;
  const studied = new Set(
    library.lessons
      .filter((lesson) => lesson.finished)
      .map((lesson) => lesson.id),
  );
  const nodes = (selected?.nodes ?? milestones).map((node) => {
    if (!selected) return node;
    const completed = studied.has(node.title.toLowerCase());
    const prerequisites = node.prerequisites.every((id) => {
      const prerequisite = selected.nodes.find((item) => item.id === id);
      return prerequisite && studied.has(prerequisite.title.toLowerCase());
    });
    const started = library.lessons.some(
      (lesson) => lesson.id === node.title.toLowerCase(),
    );
    return {
      ...node,
      masteryScore: completed ? 1 : 0,
      nodeState: completed
        ? ("mastered" as const)
        : !prerequisites
          ? ("locked" as const)
          : started
            ? ("in_progress" as const)
            : ("available" as const),
    };
  });

  async function createRoadmap(goal: RoadmapGoal) {
    setError("");
    if (library.roadmaps.length >= 20) {
      setError("Your library already has 20 roadmaps.");
      return;
    }
    if (
      goal.goal.length < 3 ||
      goal.goal.length > 300 ||
      (goal.syllabus?.length ?? 0) > 2500
    ) {
      setError(
        "Use 3–300 characters for your goal and up to 2,500 for the syllabus.",
      );
      return;
    }
    setPending(true);
    try {
      const response = await fetch("/api/learning/content", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kind: "roadmap",
          topic: goal.goal,
          syllabus: goal.syllabus,
          deadline: goal.deadline,
          interests: library.interests,
          studied: library.lessons
            .filter((lesson) => lesson.finished)
            .map(({ topic }) => topic)
            .slice(-12),
        }),
      });
      const data = await response.json();
      if (!response.ok)
        throw new Error(data.error ?? "Roadmap generation failed.");
      const roadmap = roadmapSchema.parse(data.roadmap);
      updateLibrary((previous) => ({
        ...previous,
        roadmaps: [...previous.roadmaps, roadmap],
        activeRoadmapId: roadmap.id,
      }));
      setShowForm(false);
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "The roadmap could not be generated.",
      );
    } finally {
      setPending(false);
    }
  }

  return (
    <main className="container">
      <section className={s.roadmapHead}>
        <p className="meta-text">{learnerName}’s learning roadmap</p>
        <h1>{title}</h1>
        <p>
          {nodes.filter((node) => node.nodeState === "mastered").length} of{" "}
          {nodes.length} topics completed
        </p>
        <div className={l.roadmapActions}>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => {
              setError("");
              setShowForm(true);
            }}
          >
            Generate roadmap
          </button>
          {selected && (
            <a href="#roadmap-steps" className="btn">
              Go to study plan
            </a>
          )}
          <Link href="/dashboard/demo" className="btn">
            Back to my schedule
          </Link>
        </div>
      </section>
      {library.roadmaps.length > 0 && (
        <div className={l.roadmapPicker}>
          <label htmlFor="saved-roadmap">My roadmaps</label>
          <select
            id="saved-roadmap"
            value={selected?.id ?? ""}
            onChange={(event) => {
              const activeRoadmapId = event.target.value || undefined;
              updateLibrary((previous) => ({ ...previous, activeRoadmapId }));
            }}
          >
            <option value="">{roadmapTitle}</option>
            {library.roadmaps.map((item) => (
              <option key={item.id} value={item.id}>
                {item.goal}
              </option>
            ))}
          </select>
        </div>
      )}
      <div className={s.roadmap}>
        <RoadmapDAGView
          key={selected?.id ?? "starter"}
          nodes={nodes}
          readOnly
        />
      </div>
      {selected && (
        <section className={l.roadmapSteps} aria-labelledby="roadmap-steps">
          <h2 id="roadmap-steps">Study plan</h2>
          <ol>
            {nodes.map((node) => (
              <li key={node.id}>
                <div>
                  <h3>{node.title}</h3>
                  <p>{node.description}</p>
                  <small>
                    {node.estimatedMinutes} min ·{" "}
                    {node.nodeState.replace("_", " ")}
                  </small>
                </div>
                {node.nodeState !== "locked" && (
                  <Link
                    className="btn"
                    href={{
                      pathname: "/dashboard/demo/study",
                      query: { topic: node.title, context: node.description },
                    }}
                  >
                    {node.type === "assessment"
                      ? "Review and self-check"
                      : "Open lesson"}
                  </Link>
                )}
              </li>
            ))}
          </ol>
        </section>
      )}
      {showForm && (
        <RoadmapGoalModal
          prefillGoal={prefill ?? library.activeTopic}
          description="Set a learning goal and Gemma 4 will organize it into lessons and checkpoints. Open the available topics to study them."
          error={error}
          isGenerating={pending}
          onSubmit={createRoadmap}
          onCancel={() => {
            if (!pending) setShowForm(false);
          }}
        />
      )}
    </main>
  );
}
