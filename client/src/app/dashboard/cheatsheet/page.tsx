"use client";

import { useMemo, useState } from "react";
import { generateCheatSheet } from "./actions";

type CheatSheetResult = {
  title: string;
  markdown: string;
  fileName: string;
};

function markdownToPdfLines(markdown: string): string[] {
  const lines = markdown.split("\n");
  const output: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();

    if (trimmed.startsWith("### ")) {
      output.push(`• ${trimmed.slice(4).trim()}`);
      continue;
    }

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

export default function CheatSheetPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CheatSheetResult | null>(null);

  const wordCount = useMemo(() => {
    if (!result?.markdown) return 0;
    return result.markdown.split(/\s+/).filter(Boolean).length;
  }, [result]);

  async function handleGenerate(formData: FormData) {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await generateCheatSheet(formData);
      if (!response.success) {
        setError(response.error || "Failed to generate cheat sheet.");
        return;
      }

      if (!response.data) {
        setError("Cheat sheet output missing.");
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
    <div
      style={{
        minHeight: "100vh",
        padding: "32px 24px",
        backgroundColor: "var(--bg)",
      }}
    >
      <div
        className="container"
        style={{ maxWidth: 900, margin: "0 auto", display: "grid", gap: 28 }}
      >
        <div>
          <h1
            style={{
              fontFamily: "var(--font-playfair)",
              color: "var(--ink)",
              marginBottom: 8,
            }}
          >
            Cheat Sheet Generator
          </h1>
          <div className="meta-text" style={{ color: "var(--muted)" }}>
            One-shot generation pipeline: text → markdown → downloadable PDF.
          </div>
        </div>

        <form
          action={handleGenerate}
          style={{
            display: "grid",
            gap: 16,
            background: "var(--card-bg)",
            border: "0.5px solid var(--rule)",
            padding: 20,
          }}
        >
          <div style={{ display: "grid", gap: 8 }}>
            <label htmlFor="title" className="meta-text">
              COURSE TITLE
            </label>
            <input
              id="title"
              name="title"
              type="text"
              required
              placeholder="e.g., Data Structures and Algorithms"
            />
          </div>

          <div style={{ display: "grid", gap: 8 }}>
            <label htmlFor="description" className="meta-text">
              COURSE DESCRIPTION
            </label>
            <textarea
              id="description"
              name="description"
              rows={4}
              required
              placeholder="e.g., Complexity analysis, arrays, linked lists, trees, graphs, dynamic programming."
            />
          </div>

          <div style={{ display: "grid", gap: 8 }}>
            <label htmlFor="additionalInstructions" className="meta-text">
              ADDITIONAL INSTRUCTIONS (OPTIONAL)
            </label>
            <textarea
              id="additionalInstructions"
              name="additionalInstructions"
              rows={3}
              placeholder="e.g., Prioritize exam tricks, include common mistakes, keep formulas prominent."
            />
          </div>

          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading}
              style={{ opacity: loading ? 0.6 : 1 }}
            >
              {loading ? "GENERATING..." : "GENERATE CHEAT SHEET"}
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

            <pre
              style={{
                margin: 0,
                border: "0.5px solid var(--rule)",
                background: "var(--card-bg)",
                padding: 18,
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                color: "var(--ink)",
                lineHeight: 1.5,
                overflowX: "auto",
                maxHeight: "65vh",
              }}
            >
              {result.markdown}
            </pre>
          </section>
        )}
      </div>
    </div>
  );
}
