/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Roadmap Goal Modal
   User sets their learning goal + profile to seed the
   Adaptive Roadmap Agent.
   ═══════════════════════════════════════════════════════════ */

"use client";

import { useState } from "react";
import type { RoadmapGoal } from "@/lib/types";

interface Props {
  onSubmit: (goal: RoadmapGoal) => void;
  onCancel: () => void;
  isGenerating?: boolean;
  prefillGoal?: string;
}

export default function RoadmapGoalModal({
  onSubmit,
  onCancel,
  isGenerating,
  prefillGoal,
}: Props) {
  const [goal, setGoal] = useState(prefillGoal ?? "");
  const [syllabus, setSyllabus] = useState("");
  const [deadline, setDeadline] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!goal.trim()) return;

    onSubmit({
      goal: goal.trim(),
      syllabus: syllabus.trim() || undefined,
      deadline: deadline || undefined,
    });
  };

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(10,10,12,0.75)",
        backdropFilter: "blur(8px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 99999,
        padding: "24px",
      }}
    >
      <div
        style={{
          background: "var(--bg)",
          border: "0.5px solid var(--rule)",
          width: "100%",
          maxWidth: 620,
          maxHeight: "90vh",
          overflowY: "auto",
          padding: "48px",
          position: "relative",
        }}
      >
        {/* Close */}
        <button
          type="button"
          onClick={onCancel}
          style={{
            position: "absolute",
            top: 24,
            right: 24,
            background: "transparent",
            border: "none",
            color: "var(--muted)",
            fontSize: 20,
            cursor: "pointer",
          }}
        >
          ✕
        </button>

        {/* Header */}
        <div className="meta-text" style={{ marginBottom: 8 }}>
          ADAPTIVE ROADMAP AGENT
        </div>
        <h2
          style={{ fontSize: 28, marginBottom: 8, fontFamily: "var(--serif)" }}
        >
          Define Your Learning Goal
        </h2>
        <p
          style={{
            color: "var(--muted)",
            fontSize: 13,
            marginBottom: 40,
            lineHeight: 1.7,
          }}
        >
          The Roadmap Agent will generate a personalised DAG of topics with
          prerequisites, difficulty scores, and study times — automatically
          creating CogFlow tasks from each node.
        </p>

        <form onSubmit={handleSubmit}>
            <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
              {/* Goal */}
              <div>
                <label>Learning Goal *</label>
                <textarea
                  value={goal}
                  onChange={(e) => setGoal(e.target.value)}
                  placeholder="e.g. Master Data Structures & Algorithms for coding interviews"
                  rows={3}
                  required
                  style={{
                    width: "100%",
                    resize: "vertical",
                    padding: "12px 0",
                    fontFamily: "var(--mono)",
                    fontSize: 14,
                    background: "transparent",
                    color: "var(--ink)",
                    outline: "none",
                    border: "none",
                    borderBottom: "0.5px solid var(--rule)",
                  }}
                />
              </div>

              {/* Syllabus */}
              <div>
                <label>Syllabus / Reference Material (optional)</label>
                <textarea
                  value={syllabus}
                  onChange={(e) => setSyllabus(e.target.value)}
                  placeholder="Paste your course syllabus, topic list, or any reference material here..."
                  rows={6}
                  style={{
                    width: "100%",
                    resize: "vertical",
                    fontFamily: "var(--mono)",
                    fontSize: 13,
                    background: "transparent",
                    color: "var(--ink)",
                    outline: "none",
                    border: "none",
                    borderBottom: "0.5px solid var(--rule)",
                    padding: "12px 0",
                  }}
                />
              </div>

              {/* Deadline */}
              <div>
                <label>Target Deadline (optional)</label>
                <input
                  type="date"
                  value={deadline}
                  onChange={(e) => setDeadline(e.target.value)}
                  min={new Date().toISOString().split("T")[0]}
                />
              </div>

              <button
                type="submit"
                className="btn btn-primary"
                disabled={isGenerating || !goal.trim()}
                style={{
                  alignSelf: "flex-end",
                  opacity: isGenerating ? 0.6 : 1,
                  cursor: isGenerating ? "not-allowed" : "pointer",
                  minWidth: 180,
                }}
              >
                {isGenerating ? "Generating Roadmap…" : "Generate Roadmap →"}
              </button>
            </div>
        </form>
      </div>
    </div>
  );
}
