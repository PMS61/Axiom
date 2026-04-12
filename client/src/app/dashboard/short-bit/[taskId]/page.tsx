"use client";

import { useRouter } from "next/navigation";
import React, { useEffect, useState } from "react";
import { getTasks } from "@/app/actions/tasks";
import { getStoredContentByTypeAction, saveCourseAction } from "@/app/actions/courses";
import Header from "@/components/Header";
import ShortBitStack from "@/components/ShortBitStack";
import { useApp } from "@/lib/store";
import { generateShortBitsForTask } from "./actions";
import { useToast } from "@/components/ui/ToastProvider";

type ShortBitSet = {
  title: string;
  items: Array<{ id: string; imageKeyword: string; blogContent: string }>;
};

export default function TaskShortBitPage({
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
  const [result, setResult] = useState<ShortBitSet | null>(null);
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
    async function loadSavedShortBits() {
      if (!task) return;
      const res = await getStoredContentByTypeAction(unwrappedParams.taskId, "short_bits");
      if (res.contentData) {
        setResult(res.contentData as ShortBitSet);
      }
    }

    if (!result) {
      loadSavedShortBits();
    }
  }, [task, unwrappedParams.taskId, result]);

  async function handleGenerate() {
    if (!task) return;

    setLoading(true);
    setError(null);
    addToast(`Short-bits generation started for ${task.name}.`, "info");

    try {
      const response = await generateShortBitsForTask({
        title: task.name,
        description: task.subject || task.name,
        additionalInstructions:
          "Keep outputs concise and actionable for revision.",
      });

      if (!response.success || !response.data) {
        setResult(null);
        const message = response.error || "Failed to generate short bits.";
        setError(message);
        addToast(message, "error");
        return;
      }

      setResult(response.data);
      await saveCourseAction(unwrappedParams.taskId, response.data, "short_bits");
      addToast("Short-bits generation completed.", "success");
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
            Auto-generated short bits from this scheduled roadmap task.
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
                ? "REGENERATE SHORT BITS"
                : "GENERATE SHORT BITS"}
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
            <ShortBitStack
              items={result.items}
              heading={result.title || "Generated Short Bits"}
            />
          </section>
        )}
      </div>
    </>
  );
}
