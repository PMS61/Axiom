/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Story Generator Page
   One-shot next chapter generation with beautiful markdown.
   ═══════════════════════════════════════════════════════════ */

"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import Header from "@/components/Header";
import { generateStoryChapter } from "./actions";

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

export default function StoryPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<StoryResult | null>(null);

  const wordCount = useMemo(() => {
    if (!result?.markdown) return 0;
    return result.markdown.split(/\s+/).filter(Boolean).length;
  }, [result]);

  async function handleGenerate(formData: FormData) {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await generateStoryChapter(formData);
      if (!response.success) {
        setError(response.error || "Failed to generate chapter.");
        return;
      }

      if (!response.data) {
        setError("Story output missing.");
        return;
      }

      setResult(response.data);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected error";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <Header />
      <div
        style={{
          minHeight: "100vh",
          padding: "40px 24px 64px",
          backgroundColor: "var(--bg)",
        }}
      >
        <div
          className="container"
          style={{ maxWidth: 900, margin: "0 auto", display: "grid", gap: 28 }}
        >
          <div>
            <button
              type="button"
              onClick={() => router.push("/dashboard")}
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
              ← BACK TO DASHBOARD
            </button>
            <h1
              style={{
                fontFamily: "var(--serif)",
                color: "var(--ink)",
                marginBottom: 8,
              }}
            >
              Story Generator
            </h1>
            <div className="meta-text" style={{ color: "var(--muted)" }}>
              One-shot next chapter generation from your title, description, and
              style preferences.
            </div>
          </div>

          <form
            action={handleGenerate}
            style={{
              display: "grid",
              gap: 16,
              background: "var(--card-bg)",
              border: "0.5px solid var(--rule)",
              padding: 24,
            }}
          >
            <div style={{ display: "grid", gap: 8 }}>
              <label htmlFor="title" className="meta-text">
                STORY TITLE
              </label>
              <input
                id="title"
                name="title"
                type="text"
                required
                placeholder="e.g., Ashes Over Caligo"
              />
            </div>

            <div style={{ display: "grid", gap: 8 }}>
              <label htmlFor="description" className="meta-text">
                STORY DESCRIPTION
              </label>
              <textarea
                id="description"
                name="description"
                rows={3}
                required
                placeholder="e.g., A resistance group fights to reclaim a city swallowed by living fog."
              />
            </div>

            <div style={{ display: "grid", gap: 8 }}>
              <label htmlFor="storyPreference" className="meta-text">
                USER STORY PREFERENCE
              </label>
              <textarea
                id="storyPreference"
                name="storyPreference"
                rows={3}
                placeholder="e.g., Dark fantasy tone, cinematic pacing, morally gray characters."
              />
            </div>

            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={loading}
                style={{ opacity: loading ? 0.6 : 1 }}
              >
                {loading ? "GENERATING..." : "GENERATE NEXT CHAPTER"}
              </button>
            </div>
          </form>

          {error && (
            <div
              className="meta-text"
              style={{
                color: "var(--vermillion)",
                border: "0.5px solid var(--rule)",
                padding: 12,
              }}
            >
              ERR: {error}
            </div>
          )}

          {result && (
            <section style={{ display: "grid", gap: 16 }}>
              <div
                style={{
                  display: "flex",
                  gap: 12,
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <div className="meta-text" style={{ color: "var(--muted)" }}>
                  READY · {wordCount} WORDS · MARKDOWN PREVIEW
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
                  maxHeight: "75vh",
                }}
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {result.markdown}
                </ReactMarkdown>
              </div>
            </section>
          )}
        </div>
      </div>
    </>
  );
}
