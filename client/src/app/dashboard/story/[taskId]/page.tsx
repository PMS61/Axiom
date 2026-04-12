/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Task Story Page
   Render a personalised educational story for a specific task.
   Uses ReactMarkdown for beautiful editorial formatting.
   ═══════════════════════════════════════════════════════════ */

"use client";

import { useRouter } from "next/navigation";
import React, { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getTasks } from "@/app/actions/tasks";
import { getStoredContentByTypeAction, saveCourseAction } from "@/app/actions/courses";
import Header from "@/components/Header";
import { useApp } from "@/lib/store";
import { generateStoryForTask } from "./actions";
import { useToast } from "@/components/ui/ToastProvider";

type StoryResult = {
  title: string;
  markdown: string;
  fileName: string;
};

function markdownToPdfLines(markdown: string): string[] {
  const lines = markdown.split("\n");
  const output: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();

    if (trimmed.startsWith("## ")) {
      output.push("");
      output.push(trimmed.slice(3).trim().toUpperCase());
      output.push("-");
      continue;
    }

    if (trimmed.startsWith("# ")) {
      output.push(trimmed.slice(2).trim().toUpperCase());
      output.push("=");
      continue;
    }

    if (trimmed.startsWith("- ")) {
      output.push(`• ${trimmed.slice(2).trim()}`);
      continue;
    }

    output.push(line);
  }

  return output;
}

async function downloadMarkdownAsPdf(markdown: string, fileName: string) {
  const { jsPDF } = await import("jspdf");
  const doc = new jsPDF({ unit: "pt", format: "a4" });

  const marginX = 48;
  const marginY = 56;
  const maxWidth = doc.internal.pageSize.getWidth() - marginX * 2;
  const pageBottom = doc.internal.pageSize.getHeight() - marginY;

  let cursorY = marginY;
  const lines = markdownToPdfLines(markdown);

  const writeLine = (text: string, size = 11, spacing = 16, isBold = false) => {
    const font = isBold ? "helvetica" : "times";
    doc.setFont(font, isBold ? "bold" : "normal");
    doc.setFontSize(size);

    const wrapped = doc.splitTextToSize(text || " ", maxWidth);
    for (const part of wrapped) {
      if (cursorY > pageBottom) {
        doc.addPage();
        cursorY = marginY;
      }

      doc.text(part, marginX, cursorY);
      cursorY += spacing;
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();

    if (trimmed === "=") {
      cursorY += 6;
      continue;
    }

    if (trimmed === "-") {
      doc.setDrawColor(120);
      doc.line(marginX, cursorY - 8, marginX + maxWidth, cursorY - 8);
      cursorY += 2;
      continue;
    }

    if (!trimmed) {
      cursorY += 8;
      continue;
    }

    const isSectionHeader =
      /^[A-Z0-9\s().:/&-]+$/.test(trimmed) && trimmed.length < 80;
    if (isSectionHeader) {
      writeLine(trimmed, 13, 18, true);
      cursorY += 3;
      continue;
    }

    writeLine(line, 11, 15, false);
  }

  doc.save(fileName);
}

export default function TaskStoryPage({
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
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<StoryResult | null>(null);

  const wordCount = useMemo(() => {
    if (!result?.markdown) return 0;
    return result.markdown.split(/\s+/).filter(Boolean).length;
  }, [result]);

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
    async function loadSavedStory() {
      if (!task) return;
      const res = await getStoredContentByTypeAction(unwrappedParams.taskId, "story");
      if (res.contentData) {
        setResult(res.contentData as StoryResult);
      }
    }

    if (!result) {
      loadSavedStory();
    }
  }, [task, unwrappedParams.taskId, result]);

  async function handleGenerate() {
    if (!task) return;

    setLoading(true);
    setError(null);
    addToast(`Story generation started for ${task.name}.`, "info");

    try {
      const response = await generateStoryForTask({
        title: task.name,
        description: task.subject || task.name,
        storyPreference: "User like cultivation novels.",
      });

      if (!response.success || !response.data) {
        setResult(null);
        const message = response.error || "Failed to generate story chapter.";
        setError(message);
        addToast(message, "error");
        return;
      }

      setResult(response.data);
      await saveCourseAction(unwrappedParams.taskId, response.data, "story");
      addToast("Story generation completed.", "success");
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
              fontFamily: "var(--serif)",
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
            Auto-generated story chapter from this scheduled roadmap task.
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
                ? "REGENERATE STORY"
                : "GENERATE STORY"}
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
            className="container"
            style={{ paddingTop: 24, paddingBottom: 60 }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: 12,
              }}
            >
              <div className="meta-text" style={{ color: "var(--muted)" }}>
                READY · {wordCount} WORDS
              </div>
              <button
                type="button"
                className="btn"
                onClick={() =>
                  downloadMarkdownAsPdf(result.markdown, result.fileName)
                }
              >
                DOWNLOAD PDF
              </button>
            </div>
            <div
              className="story-content"
              style={{
                margin: 0,
                border: "0.5px solid var(--rule)",
                background: "var(--card-bg)",
                padding: "48px 64px",
                color: "var(--ink)",
                overflowX: "auto",
                maxHeight: "65vh",
                overflowY: "auto",
              }}
            >
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  h1: ({ node, ...props }) => (
                    <h1
                      style={{
                        fontFamily: "var(--font-playfair)",
                        fontSize: 28,
                        marginTop: 0,
                        marginBottom: 12,
                      }}
                      {...props}
                    />
                  ),
                  h2: ({ node, ...props }) => (
                    <h2
                      style={{ marginTop: 22, marginBottom: 10 }}
                      {...props}
                    />
                  ),
                  h3: ({ node, ...props }) => (
                    <h3 style={{ marginTop: 16, marginBottom: 8 }} {...props} />
                  ),
                  p: ({ node, ...props }) => (
                    <p style={{ margin: "0 0 10px" }} {...props} />
                  ),
                  ul: ({ node, ...props }) => (
                    <ul
                      style={{ paddingLeft: 20, margin: "0 0 10px" }}
                      {...props}
                    />
                  ),
                  ol: ({ node, ...props }) => (
                    <ol
                      style={{ paddingLeft: 20, margin: "0 0 10px" }}
                      {...props}
                    />
                  ),
                  code: ({ node, ...props }) => (
                    <code
                      style={{
                        background: "var(--bg)",
                        border: "0.5px solid var(--rule)",
                        padding: "1px 4px",
                      }}
                      {...props}
                    />
                  ),
                  pre: ({ node, ...props }) => (
                    <pre
                      style={{
                        background: "var(--bg)",
                        border: "0.5px solid var(--rule)",
                        padding: 12,
                        overflowX: "auto",
                      }}
                      {...props}
                    />
                  ),
                }}
              >
                {result.markdown}
              </ReactMarkdown>
            </div>
          </section>
        )}
      </div>
    </>
  );
}
