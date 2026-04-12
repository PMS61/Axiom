/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Adaptive Roadmap Page
   Primary navigation view: DAG of learning topics with
   prerequisites, mastery states, and CogFlow task generation.
   ═══════════════════════════════════════════════════════════ */

"use client";

import { Suspense, useCallback, useState, useEffect, useRef } from "react";
import { useApp } from "@/lib/store";
import { useRouter, useSearchParams } from "next/navigation";
import Header from "@/components/Header";
import RoadmapDAGView from "@/components/RoadmapDAGView";
import RoadmapGoalModal from "@/components/RoadmapGoalModal";
import {
  generateRoadmapAction,
  saveRoadmapAction,
  clearRoadmapAction,
  listRoadmapsAction,
  getRoadmapAction,
  getLatestRoadmapAction,
} from "@/app/actions/roadmap";
import { saveTask, syncTasks, getTasks } from "@/app/actions/tasks";
import { computeCL } from "@/lib/engine";
import type { RoadmapDAGNode, RoadmapGoal, Task } from "@/lib/types";

// ── Task type mapping ─────────────────────────────────────

function nodeToTask(node: RoadmapDAGNode): Task {
  const taskType = node.type === "assessment" ? "problem_solving" : "learning";
  const clResult = computeCL(
    node.difficulty,
    node.estimatedMinutes,
    0, // no deadline urgency yet
    taskType,
    node.priority,
  );

  return {
    id: crypto.randomUUID(),
    name: node.title,
    type: taskType,
    difficulty: node.difficulty,
    completionTime: node.estimatedMinutes,
    priority: node.priority,
    state: "unscheduled",
    subject: node.subject,
    etask: clResult.total,
    etaskBreakdown: clResult,
    sequence: node.sequence,
    createdAt: new Date().toISOString(),
  };
}

// ── Stats bar ─────────────────────────────────────────────

function RoadmapStats({ nodes }: { nodes: RoadmapDAGNode[] }) {
  const mastered = nodes.filter((n) => n.nodeState === "mastered").length;
  const inProgress = nodes.filter((n) => n.nodeState === "in_progress").length;
  const available = nodes.filter((n) => n.nodeState === "available").length;
  const locked = nodes.filter((n) => n.nodeState === "locked").length;
  const totalMinutes = nodes.reduce((s, n) => s + n.estimatedMinutes, 0);
  const masteredMinutes = nodes
    .filter((n) => n.nodeState === "mastered")
    .reduce((s, n) => s + n.estimatedMinutes, 0);
  const overallMastery =
    nodes.length > 0
      ? nodes.reduce((s, n) => s + n.masteryScore, 0) / nodes.length
      : 0;

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(6, 1fr)",
        borderBottom: "0.5px solid var(--rule)",
        background: "var(--card-bg)",
      }}
    >
      {[
        { label: "Total Nodes", value: nodes.length, color: "var(--ink)" },
        { label: "Mastered", value: mastered, color: "var(--safe)" },
        { label: "In Progress", value: inProgress, color: "var(--watch)" },
        { label: "Available", value: available, color: "var(--ink)" },
        { label: "Locked", value: locked, color: "var(--muted)" },
        {
          label: "Overall Mastery",
          value: `${Math.round(overallMastery * 100)}%`,
          color: overallMastery >= 0.8 ? "var(--safe)" : "var(--ink)",
        },
      ].map(({ label, value, color }) => (
        <div
          key={label}
          style={{
            padding: "16px 20px",
            borderRight: "0.5px solid var(--rule)",
          }}
        >
          <div
            style={{
              fontSize: 9,
              fontFamily: "var(--mono)",
              color: "var(--muted)",
              letterSpacing: "0.15em",
              marginBottom: 6,
            }}
          >
            {label.toUpperCase()}
          </div>
          <div
            style={{
              fontWeight: 700,
              fontSize: 24,
              fontFamily: "var(--mono)",
              color,
            }}
          >
            {value}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────

export default function RoadmapPage() {
  return (
    <Suspense
      fallback={
        <div
          style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}
        >
          <Header />
        </div>
      }
    >
      <RoadmapPageContent />
    </Suspense>
  );
}

function RoadmapPageContent() {
  const { state, dispatch } = useApp();
  const prevRoadmapDateRef = useRef<string>("");
  const autoGenerateFingerprintRef = useRef<string>("");
  const router = useRouter();
  const searchParams = useSearchParams();
  const [showGoalModal, setShowGoalModal] = useState(false);
  const [prefillGoal, setPrefillGoal] = useState<string | undefined>(undefined);
  const [showSavedModal, setShowSavedModal] = useState(false);
  const [savedRoadmaps, setSavedRoadmaps] = useState<any[]>([]);
  const [loadingRoadmaps, setLoadingRoadmaps] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ── Auto-open goal modal if ?goal param present ───────────
  useEffect(() => {
    const goalParam = searchParams.get("goal");
    const autoGenerate = searchParams.get("autogenerate") === "1";
    if (goalParam && !autoGenerate && !state.activeRoadmap) {
      setPrefillGoal(decodeURIComponent(goalParam));
      setShowGoalModal(true);
    }
  }, [searchParams, state.activeRoadmap]);

  // ── Make sure Tasks are loaded ───────────────────────────
  useEffect(() => {
    async function loadTasks() {
      // Fetch fresh tasks when entering roadmap to sync states
      const { tasks, error } = await getTasks();
      if (!error && tasks) {
        dispatch({ type: "SET_TASKS", payload: tasks });
      }
    }
    loadTasks();
  }, [dispatch]);

  // ── Load latest roadmap on mount ────────────────────────
  useEffect(() => {
    async function loadLatest() {
      // Only auto-load if we don't already have an active roadmap in store
      // or if we want to ensure it's the absolute freshest
      if (!state.activeRoadmap) {
        dispatch({ type: "SET_ROADMAP_GENERATING", payload: true });
        const { roadmap, error } = await getLatestRoadmapAction();
        if (roadmap) {
          dispatch({ type: "SET_ROADMAP", payload: roadmap });
        }
        dispatch({ type: "SET_ROADMAP_GENERATING", payload: false });
      }
    }
    loadLatest();
  }, [dispatch, state.activeRoadmap]);

  // ── Load saved roadmaps list ───────────────────────────
  const fetchSavedRoadmaps = useCallback(async () => {
    setLoadingRoadmaps(true);
    const result = await listRoadmapsAction();
    if (result.roadmaps) {
      setSavedRoadmaps(result.roadmaps);
    }
    setLoadingRoadmaps(false);
  }, []);

  const loadSpecificRoadmap = useCallback(async (id: string) => {
    dispatch({ type: "SET_ROADMAP_GENERATING", payload: true });
    setShowSavedModal(false);
    const result = await getRoadmapAction(id);
    if (result.roadmap) {
      dispatch({ type: "SET_ROADMAP", payload: result.roadmap });
      setShowGoalModal(false);
    } else {
      setError(result.error ?? "Failed to load roadmap");
    }
    dispatch({ type: "SET_ROADMAP_GENERATING", payload: false });
  }, [dispatch]);

  // ── Auto-save roadmap ──────────────────────────────────
  useEffect(() => {
    if (
      state.activeRoadmap &&
      state.activeRoadmap.lastUpdatedAt !== prevRoadmapDateRef.current
    ) {
      prevRoadmapDateRef.current = state.activeRoadmap.lastUpdatedAt;
      saveRoadmapAction(state.activeRoadmap).catch((err) => {
        console.error("Auto-save roadmap failed:", err);
      });
    }
  }, [state.activeRoadmap]);

  // ── Generate roadmap from goal ─────────────────────────
  const handleGoalSubmit = useCallback(
    async (goal: RoadmapGoal) => {
      dispatch({ type: "SET_ROADMAP_GENERATING", payload: true });
      setError(null);
      setShowGoalModal(false);

      const result = await generateRoadmapAction(goal);
      if (result.error || !result.roadmap) {
        setError(result.error ?? "Failed to generate roadmap.");
        dispatch({ type: "SET_ROADMAP_GENERATING", payload: false });
        setShowGoalModal(true);
        return;
      }

      // Auto-create a task for every node
      const newTasks: Task[] = [];
      const updatedNodes = result.roadmap.nodes.map(node => {
        const task = nodeToTask(node);
        newTasks.push(task);
        return { ...node, taskId: task.id };
      });
      result.roadmap.nodes = updatedNodes;
      result.roadmap.lastUpdatedAt = new Date().toISOString(); // Update timestamp to trigger auto-save

      dispatch({ type: "SET_ROADMAP", payload: result.roadmap });
      
      // Dispatch ADD_TASK sequentially for the reducer
      newTasks.forEach(t => dispatch({ type: "ADD_TASK", payload: t }));
      
      // Save them all to the database
      try {
        await syncTasks(newTasks);
      } catch (e) {
        console.error("Failed to sync auto-generated tasks:", e);
      }
    },
    [dispatch],
  );

  // ── Auto-generate from Explore payload (?autogenerate=1&goal=...&syllabus=...) ──
  useEffect(() => {
    const shouldAutoGenerate = searchParams.get("autogenerate") === "1";
    const goalParam = searchParams.get("goal");

    if (!shouldAutoGenerate || !goalParam) return;

    const syllabusParam = searchParams.get("syllabus") ?? undefined;
    const deadlineParam = searchParams.get("deadline") ?? undefined;
    const fingerprint = `${goalParam}::${syllabusParam ?? ""}::${deadlineParam ?? ""}`;

    if (autoGenerateFingerprintRef.current === fingerprint) {
      return;
    }

    autoGenerateFingerprintRef.current = fingerprint;

    handleGoalSubmit({
      goal: goalParam,
      syllabus: syllabusParam,
      deadline: deadlineParam,
    });
  }, [searchParams, handleGoalSubmit]);

  // ── Mastery update ─────────────────────────────────────
  const handleMasteryUpdate = useCallback(
    (nodeId: string, score: number) => {
      dispatch({ type: "UPDATE_NODE_MASTERY", payload: { nodeId, score } });
    },
    [dispatch],
  );

  // ── Convert node → CogFlow task ───────────────────────
  const handleConvertToTask = useCallback(
    async (node: RoadmapDAGNode) => {
      if (!state.activeRoadmap) return;

      const task = nodeToTask(node);

      dispatch({ type: "ADD_TASK", payload: task });
      dispatch({
        type: "LINK_NODE_TO_TASK",
        payload: { nodeId: node.id, taskId: task.id },
      });

      try {
        await saveTask(task);
      } catch (e) {
        console.error("Failed to persist task:", e);
      }
    },
    [state.activeRoadmap, state.tasks, dispatch],
  );

  const nodes = state.activeRoadmap?.nodes ?? [];

  return (
    <div
      style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}
    >
      <Header />

      {/* ── Page Header ── */}
      <div
        style={{
          borderBottom: "0.5px solid var(--rule)",
          padding: "0 24px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          height: 56,
          background: "var(--bg)",
          flexShrink: 0,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
          <div
            style={{
              fontFamily: "var(--mono)",
              fontSize: 10,
              letterSpacing: "0.15em",
              color: "var(--muted)",
            }}
          >
            ADAPTIVE ROADMAP
          </div>
          {state.activeRoadmap && (
            <div
              style={{
                fontFamily: "var(--mono)",
                fontSize: 12,
                color: "var(--ink)",
                maxWidth: 480,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {state.activeRoadmap.goal.goal}
            </div>
          )}
        </div>

        <div style={{ display: "flex", gap: 8 }}>
          {error && (
            <span
              style={{
                fontSize: 11,
                color: "var(--vermillion)",
                fontFamily: "var(--mono)",
                alignSelf: "center",
              }}
            >
              [!] {error}
            </span>
          )}
          {state.roadmapGenerating && (
            <span
              style={{
                fontSize: 11,
                color: "var(--muted)",
                fontFamily: "var(--mono)",
                alignSelf: "center",
              }}
            >
              Generating…
            </span>
          )}
          {state.activeRoadmap && (
            <>
              <button
                className="btn btn-sm"
                onClick={() => setShowGoalModal(true)}
              >
                New Goal
              </button>
              <button
                className="btn btn-sm"
                onClick={() => {
                  fetchSavedRoadmaps();
                  setShowSavedModal(true);
                }}
              >
                Load Saved
              </button>
              <button
                className="btn btn-sm"
                style={{ color: "var(--muted)" }}
                onClick={async () => {
                  if (confirm("Clear the current roadmap?")) {
                    dispatch({ type: "CLEAR_ROADMAP" });
                    setShowGoalModal(true);
                  }
                }}
              >
                Clear
              </button>
            </>
          )}
          <button
            className="btn btn-sm"
            onClick={() => router.push("/dashboard")}
          >
            ← Dashboard
          </button>
        </div>
      </div>

      {/* ── Stats Bar ── */}
      {nodes.length > 0 && <RoadmapStats nodes={nodes} />}

      {/* ── Main content ── */}
      <div style={{ flex: 1, overflow: "hidden", position: "relative" }}>
        {/* Generating state */}
        {state.roadmapGenerating && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              background: "var(--bg)",
              zIndex: 10,
              gap: 24,
            }}
          >
            {/* Animated trace lines */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 8,
                width: 360,
              }}
            >
              {[
                "Analysing goal parameters…",
                "Applying prerequisite topology…",
                "Calibrating difficulty gradient…",
                "Generating DAG with Gemini…",
              ].map((line, i) => (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    opacity: 0,
                    animation: `fadeIn 0.4s ${i * 0.5}s forwards`,
                  }}
                >
                  <div
                    style={{
                      width: 6,
                      height: 6,
                      background: "var(--ink)",
                      flexShrink: 0,
                      animation: "pulse 1s infinite",
                    }}
                  />
                  <span
                    style={{
                      fontFamily: "var(--mono)",
                      fontSize: 12,
                      color: "var(--muted)",
                    }}
                  >
                    {line}
                  </span>
                </div>
              ))}
            </div>
            <div
              style={{
                fontFamily: "var(--serif)",
                fontSize: 14,
                color: "var(--muted)",
                marginTop: 8,
              }}
            >
              Building your personalised learning path…
            </div>
          </div>
        )}

        {/* Empty state */}
        {!state.roadmapGenerating && nodes.length === 0 && (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              gap: 24,
              padding: 48,
              textAlign: "center",
            }}
          >
            {/* DAG illustration */}
            <svg width="180" height="120" viewBox="0 0 180 120">
              <rect
                x="10"
                y="40"
                width="60"
                height="40"
                fill="none"
                stroke="var(--rule)"
                strokeWidth="0.5"
              />
              <rect
                x="110"
                y="10"
                width="60"
                height="40"
                fill="none"
                stroke="var(--rule)"
                strokeWidth="0.5"
              />
              <rect
                x="110"
                y="70"
                width="60"
                height="40"
                fill="none"
                stroke="var(--rule)"
                strokeWidth="0.5"
              />
              <path
                d="M 70 60 C 90 60 90 30 110 30"
                fill="none"
                stroke="var(--rule)"
                strokeWidth="0.5"
                strokeDasharray="4 4"
              />
              <path
                d="M 70 60 C 90 60 90 90 110 90"
                fill="none"
                stroke="var(--rule)"
                strokeWidth="0.5"
                strokeDasharray="4 4"
              />
              <circle cx="40" cy="60" r="4" fill="var(--ink)" />
              <circle cx="140" cy="30" r="4" fill="var(--rule)" />
              <circle cx="140" cy="90" r="4" fill="var(--rule)" />
            </svg>

            <div>
              <h2
                style={{
                  fontFamily: "var(--serif)",
                  fontSize: 24,
                  marginBottom: 12,
                }}
              >
                No Roadmap Active
              </h2>
              <p
                style={{
                  color: "var(--muted)",
                  fontSize: 13,
                  maxWidth: 400,
                  lineHeight: 1.7,
                  marginBottom: 32,
                }}
              >
                Set a learning goal to generate a personalised DAG of topics.
                The Roadmap Agent will sequence prerequisites, assign difficulty
                scores, and create CogFlow tasks automatically.
              </p>
              <div style={{ display: "flex", gap: 16 }}>
                <button
                  className="btn btn-primary"
                  onClick={() => setShowGoalModal(true)}
                >
                  Set Learning Goal →
                </button>
                <button
                  className="btn"
                  onClick={() => {
                    fetchSavedRoadmaps();
                    setShowSavedModal(true);
                  }}
                >
                  Load Saved Roadmap
                </button>
              </div>
            </div>
          </div>
        )}

        {/* DAG View */}
        {!state.roadmapGenerating && nodes.length > 0 && (
          <RoadmapDAGView
            nodes={nodes}
            onMasteryUpdate={handleMasteryUpdate}
            onConvertToTask={handleConvertToTask}
          />
        )}
      </div>

      {/* ── Goal Modal ── */}
      {showGoalModal && !showSavedModal && (
        <RoadmapGoalModal
          onSubmit={handleGoalSubmit}
          onCancel={() => {
            setShowGoalModal(false);
            if (!state.activeRoadmap) router.push("/dashboard");
          }}
          isGenerating={state.roadmapGenerating}
          prefillGoal={prefillGoal}
        />
      )}

      {/* ── Saved Roadmaps Modal ── */}
      {showSavedModal && (
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
          }}
        >
          <div
            style={{
              background: "var(--bg)",
              border: "0.5px solid var(--rule)",
              width: "100%",
              maxWidth: 500,
              padding: "32px",
              position: "relative",
            }}
          >
            <button
              onClick={() => setShowSavedModal(false)}
              style={{
                position: "absolute",
                top: 16,
                right: 16,
                background: "none",
                border: "none",
                cursor: "pointer",
                color: "var(--muted)",
                fontSize: 16,
              }}
            >
              ✕
            </button>
            <h2 style={{ fontFamily: "var(--serif)", fontSize: 24, marginBottom: 24 }}>
              Saved Roadmaps
            </h2>
            {loadingRoadmaps ? (
              <p style={{ color: "var(--muted)", fontFamily: "var(--mono)", fontSize: 12 }}>Loading...</p>
            ) : savedRoadmaps.length === 0 ? (
              <p style={{ color: "var(--muted)", fontFamily: "var(--mono)", fontSize: 12 }}>No saved roadmaps found.</p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 12, maxHeight: 400, overflowY: "auto" }}>
                {savedRoadmaps.map((rm) => (
                  <div
                    key={rm.id}
                    style={{
                      border: "0.5px solid var(--rule)",
                      padding: 16,
                      cursor: "pointer",
                    }}
                    onClick={() => loadSpecificRoadmap(rm.id)}
                  >
                    <div style={{ fontWeight: "bold", marginBottom: 8, fontSize: 14 }}>{rm.goal.goal}</div>
                    <div style={{ display: "flex", gap: 16, fontSize: 10, fontFamily: "var(--mono)", color: "var(--muted)", textTransform: "uppercase" }}>
                      <span>{new Date(rm.lastUpdatedAt).toLocaleDateString()}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Inline CSS for animations ── */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateX(-8px); }
          to   { opacity: 1; transform: translateX(0); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.3; }
        }
      `}</style>
    </div>
  );
}
