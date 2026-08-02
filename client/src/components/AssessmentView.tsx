"use client";

import { useEffect, useState } from "react";
import {
  evaluateAssessmentAction,
  generateAssessmentAction,
} from "@/app/actions/assessment";
import {
  getStoredContentByTypeAction,
  saveCourseAction,
} from "@/app/actions/courses";
import type {
  Assessment,
  AssessmentQuestion,
  AssessmentResult,
} from "@/lib/assessment-type";
import { useApp } from "@/lib/store";
import type { Task } from "@/lib/types";

interface AssessmentViewProps {
  taskId: string;
}

export default function AssessmentView({ taskId }: AssessmentViewProps) {
  const { state, dispatch } = useApp();
  const [task, setTask] = useState<Task | null>(null);
  const [assessment, setAssessment] = useState<Assessment | null>(null);
  const [answers, setAnswers] = useState<Record<string, any>>({});
  const [result, setResult] = useState<AssessmentResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const t = state.tasks.find((t) => t.id === taskId);
    if (t) setTask(t);
  }, [taskId, state.tasks]);

  useEffect(() => {
    async function loadAssessment() {
      if (!taskId || !task) return;

      try {
        // Try to get from storage
        const stored = await getStoredContentByTypeAction(taskId, "assessment");
        if (stored.contentData) {
          setAssessment(stored.contentData as Assessment);
          setLoading(false);
          return;
        }

        // Generate new
        const roadmapGoal = state.activeRoadmap?.goal.goal || "Learning";
        const res = await generateAssessmentAction(
          task.name,
          task.subject || "",
          roadmapGoal,
        );
        if (res.error) throw new Error(res.error);
        if (res.assessment) {
          setAssessment(res.assessment);
          await saveCourseAction(taskId, res.assessment, "assessment");
        }
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    if (task) loadAssessment();
  }, [task, taskId, state.activeRoadmap]);

  const handleAnswerChange = (questionId: string, value: any) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }));
  };

  const handleSubmit = async () => {
    if (!assessment) return;
    setEvaluating(true);
    try {
      const res = await evaluateAssessmentAction(
        assessment,
        answers,
        task?.name,
      );
      if (res.error) throw new Error(res.error);
      if (res.result) {
        setResult(res.result);

        // Find roadmap node and update mastery
        const node = state.activeRoadmap?.nodes.find(
          (n) => n.taskId === taskId,
        );
        if (node) {
          dispatch({
            type: "UPDATE_NODE_MASTERY",
            payload: { nodeId: node.id, score: res.result.score },
          });
        }
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setEvaluating(false);
    }
  };

  if (loading)
    return <div className="p-8 font-mono">⊞ GENERATING ASSESSMENT...</div>;
  if (error)
    return <div className="p-8 font-mono text-red-500">Error: {error}</div>;
  if (!assessment)
    return <div className="p-8 font-mono">Assessment not found.</div>;

  return (
    <div className="max-w-3xl mx-auto p-8 font-mono">
      <header className="mb-8 border-b border-rule pb-4">
        <h1 className="text-2xl font-bold">{assessment.title}</h1>
        <p className="text-muted mt-2">{assessment.description}</p>
      </header>

      {result ? (
        <div className="space-y-8">
          <div className="p-6 border border-safe bg-safe/5">
            <h2 className="text-xl font-bold mb-2">
              RESULTS: {Math.round(result.score * 100)}%
            </h2>
            <p className="text-ink">{result.feedback}</p>
          </div>

          <div className="space-y-6">
            {assessment.questions.map((q, idx) => {
              const qResult = result.questionResults.find(
                (r) => r.questionId === q.id,
              );
              return (
                <div
                  key={q.id}
                  className={`p-4 border ${qResult?.isCorrect ? "border-safe" : "border-rule"}`}
                >
                  <div className="flex justify-between mb-2">
                    <span className="text-xs text-muted">
                      QUESTION {idx + 1}
                    </span>
                    <span
                      className={`text-xs font-bold ${qResult?.isCorrect ? "text-safe" : "text-watch"}`}
                    >
                      {qResult?.isCorrect ? "CORRECT" : "INCORRECT"} (
                      {Math.round((qResult?.score || 0) * 100)}%)
                    </span>
                  </div>
                  <div className="mb-4">{q.question}</div>
                  <div className="text-sm text-ink opacity-80 italic">
                    Feedback: {qResult?.feedback}
                  </div>
                </div>
              );
            })}
          </div>

          <button
            className="btn btn-primary w-full justify-center"
            onClick={() => setResult(null)}
          >
            RETRY ASSESSMENT
          </button>
        </div>
      ) : (
        <div className="space-y-8">
          {assessment.questions.map((q, idx) => (
            <div key={q.id} className="space-y-4 p-4 border border-rule">
              <div className="text-xs text-muted">
                QUESTION {idx + 1} — {q.type.toUpperCase()}
              </div>
              <div className="font-bold text-lg">{q.question}</div>

              {q.type === "mcq" && (
                <div className="grid gap-2">
                  {q.options.map((opt, optIdx) => (
                    <button
                      key={optIdx}
                      className={`text-left p-3 border transition-colors ${
                        answers[q.id] === optIdx
                          ? "border-ink bg-ink text-bg"
                          : "border-rule hover:border-muted"
                      }`}
                      onClick={() => handleAnswerChange(q.id, optIdx)}
                    >
                      {opt}
                    </button>
                  ))}
                </div>
              )}

              {q.type === "short_answer" && (
                <input
                  type="text"
                  className="w-full bg-transparent border border-rule p-3 focus:border-ink outline-none"
                  placeholder="Type your answer here..."
                  value={answers[q.id] || ""}
                  onChange={(e) => handleAnswerChange(q.id, e.target.value)}
                />
              )}

              {q.type === "long_answer" && (
                <textarea
                  className="w-full bg-transparent border border-rule p-3 focus:border-ink outline-none min-h-[150px]"
                  placeholder="Describe your answer in detail..."
                  value={answers[q.id] || ""}
                  onChange={(e) => handleAnswerChange(q.id, e.target.value)}
                />
              )}

              {q.type === "coding" && (
                <div className="space-y-2">
                  <div className="text-xs text-muted">
                    LANGUAGE: {q.language.toUpperCase()}
                  </div>
                  <textarea
                    className="w-full bg-card-bg text-ink border border-rule p-3 font-mono focus:border-ink outline-none min-h-[200px]"
                    placeholder={`// Your ${q.language} code here...`}
                    value={answers[q.id] || q.starterCode || ""}
                    onChange={(e) => handleAnswerChange(q.id, e.target.value)}
                  />
                  <div className="text-xs text-muted">
                    Test cases will be evaluated on submission.
                  </div>
                </div>
              )}
            </div>
          ))}

          <button
            className="btn btn-primary w-full justify-center text-lg py-4"
            disabled={evaluating}
            onClick={handleSubmit}
          >
            {evaluating ? "EVALUATING..." : "SUBMIT ASSESSMENT"}
          </button>
        </div>
      )}
    </div>
  );
}
