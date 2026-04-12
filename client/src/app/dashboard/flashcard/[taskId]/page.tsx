"use client";

import { useRouter } from "next/navigation";
import React, { useEffect, useState } from "react";
import { getTasks } from "@/app/actions/tasks";
import { getStoredContentByTypeAction, saveCourseAction } from "@/app/actions/courses";
import FlashcardStack from "@/components/FlashcardStack";
import Header from "@/components/Header";
import { useApp } from "@/lib/store";
import { generateFlashcardsForTask } from "./actions";
import { useToast } from "@/components/ui/ToastProvider";

type FlashcardSet = {
  title: string;
  cards: Array<{ id: string; front: string; back: string }>;
};

export default function TaskFlashcardPage({
  params,
}: {
  params: Promise<{ taskId: string }>;
}) {
  const unwrappedParams = React.use(params);
  const router = useRouter();
  const { state } = useApp();
  const { addToast } = useToast();

  const [task, setTask] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FlashcardSet | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function initTask() {
      if (state.tasks && state.tasks.length > 0) {
        const foundTask = state.tasks.find(
          (t: any) => t.id === unwrappedParams.taskId,
        );
        if (foundTask) {
          setTask(foundTask);
        } else {
          setError("Task not found.");
        }
      } else {
        try {
          const res = await getTasks();
          if (res.tasks) {
            const foundTask = res.tasks.find(
              (t: any) => t.id === unwrappedParams.taskId,
            );
            if (foundTask) {
              setTask(foundTask);
            } else {
              setError("Task not found.");
            }
          } else {
            setError("No tasks found in database.");
          }
        } catch (err: any) {
          setError(`Error loading tasks: ${err.message}`);
        }
      }
    }
    initTask();
  }, [state.tasks, unwrappedParams.taskId]);

  useEffect(() => {
    async function loadSavedFlashcards() {
      if (!task) return;
      const res = await getStoredContentByTypeAction(unwrappedParams.taskId, "flashcards");
      if (res.contentData) {
        setResult(res.contentData as FlashcardSet);
      }
    }

    if (!result) {
      loadSavedFlashcards();
    }
  }, [task, unwrappedParams.taskId, result]);

  async function handleGenerate() {
    if (!task) return;

    setLoading(true);
    setError(null);
    addToast(`Flashcard generation started for ${task.name}.`, "info");

    try {
      const response = await generateFlashcardsForTask({
        title: task.name,
        description: task.subject || task.name,
        additionalInstructions:
          "Use concise, exam-focused cards for this roadmap task.",
      });

      if (!response.success || !response.data) {
        setResult(null);
        const message = response.error || "Failed to generate flashcards.";
        setError(message);
        addToast(message, "error");
        return;
      }

      setResult(response.data);
      await saveCourseAction(unwrappedParams.taskId, response.data, "flashcards");
      addToast("Flashcard generation completed.", "success");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected error";
      setResult(null);
      setError(message);
      addToast(message, "error");
    } finally {
      setLoading(false);
    }
  }

  if (!task && !error) {
    return (
      <>
        <Header />
        <div
          style={{
            padding: 40,
            fontFamily: "var(--mono)",
            color: "var(--muted)",
          }}
        >
          Loading task details...
        </div>
      </>
    );
  }

  return (
    <>
      <Header />
      <div
        style={{
          paddingTop: 60,
          minHeight: "100vh",
          backgroundColor: "var(--bg)",
        }}
      >
        <section
          className="container section-rule"
          style={{ paddingTop: 40, paddingBottom: 28 }}
        >
          <button
            type="button"
            onClick={() => router.back()}
            style={{
              background: "none",
              border: "none",
              color: "var(--muted)",
              textDecoration: "underline",
              cursor: "pointer",
              fontFamily: "var(--mono)",
              fontSize: 12,
              padding: 0,
              marginBottom: 24,
            }}
          >
            ← BACK TO ROADMAP
          </button>

          <h1
            style={{
              fontFamily: "var(--font-playfair)",
              color: "var(--ink)",
              marginBottom: 8,
            }}
          >
            {task?.name}
          </h1>
          <div
            className="meta-text"
            style={{ color: "var(--muted)", marginBottom: 20 }}
          >
            Auto-generated flashcards from this scheduled roadmap task.
          </div>

          <button
            type="button"
            className="btn btn-primary"
            disabled={loading}
            onClick={handleGenerate}
            style={{ opacity: loading ? 0.6 : 1 }}
          >
            {loading
              ? "GENERATING..."
              : result
                ? "REGENERATE FLASHCARDS"
                : "GENERATE FLASHCARDS"}
          </button>

          {error && (
            <div
              className="meta-text"
              style={{ color: "var(--vermillion)", marginTop: 14 }}
            >
              ERR: {error}
            </div>
          )}
        </section>

        {result && (
          <section
            className="section-rule"
            style={{ marginTop: 20, paddingBottom: 16 }}
          >
            <FlashcardStack
              cards={result.cards}
              heading={result.title || "Generated Flashcards"}
            />
          </section>
        )}
      </div>
    </>
  );
}
